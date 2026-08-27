"""Outcome-independent stratified sampler over frozen predecessor campaigns."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest, file_sha256, publish_json, publish_jsonl, read_json
from .migration import TokenMigration, load_migration


LANE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LANE_ROOT.parent
PREDECESSOR = (
    REPO_ROOT
    / "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1"
)
RESULTS = PREDECESSOR / "results/confirmation"
SOURCE_PATHS = {
    "spectral_public": RESULTS / "spectral_public.json.gz",
    "spectral_held_out": RESULTS / "spectral_owner.json.gz",
    "fd_public": RESULTS / "fd_public.json.gz",
    "fd_held_out": RESULTS / "fd_owner.json.gz",
    "official_24": RESULTS / "llm/official_campaign.json.gz",
}
SAMPLER_SALT = "evidence-view-benchmark-v1-seed-rank"
NOISE_MODELS = ("iid", "correlated")
NOISE_LEVELS = (0.0, 0.001, 0.005)
COMMON_ACTION_KEYS = ("action_token", "nodes", "temporal_modes")
FORBIDDEN_PUBLIC_FRAGMENTS = (
    "truth",
    "held_out",
    "correct",
    "false",
    "owner",
    "oracle",
    "reference",
    "error_to_49",
)


def selected_seeds() -> tuple[int, ...]:
    candidates = range(10001, 10050)
    ranked = sorted(
        candidates,
        key=lambda seed: hashlib.sha256(
            f"{SAMPLER_SALT}|{seed}".encode("ascii")
        ).hexdigest(),
    )
    return (10000, *sorted(ranked[:5]))


def _validate_campaign(payload: Mapping[str, Any], label: str) -> None:
    body = dict(payload)
    claimed = body.pop("campaign_sha256", None)
    if claimed != digest(body):
        raise ValueError(f"{label} campaign hash mismatch")
    if payload.get("episode_count", len(payload.get("cases", []))) not in {24, 1800}:
        raise ValueError(f"{label} has unexpected size")


def _by_episode(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload["episodes"]
    output = {str(row["episode_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("duplicate episode id")
    return output


def _official_by_episode(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    output = {}
    for row in payload["cases"]:
        spectral = row["bindings"]["spectral_etdrk4"]["episode_id"]
        fd = row["bindings"]["fd_bdf"]["episode_id"]
        if spectral != fd or spectral in output:
            raise ValueError("official producer bindings do not identify one unique episode")
        output[spectral] = row
    if len(output) != 24:
        raise ValueError("official subset must contain 24 cases")
    return output


def _assert_public(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_PUBLIC_FRAGMENTS):
                raise ValueError(f"held-out field in public case at {path}.{key}")
            _assert_public(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_public(nested, f"{path}[{index}]")


def _pair_rows(
    spectral_rows: list[Mapping[str, Any]],
    fd_rows: list[Mapping[str, Any]],
    migration: TokenMigration,
) -> list[dict[str, Any]]:
    if len(spectral_rows) != 12 or len(fd_rows) != 12:
        raise ValueError("each producer must supply twelve actions")
    output = []
    for spectral, fd in zip(spectral_rows, fd_rows):
        for key in COMMON_ACTION_KEYS:
            if spectral[key] != fd[key]:
                raise ValueError(f"producer action mismatch for {key}")
        migrated_spectral = migration.migrate(dict(spectral))
        migrated_fd = migration.migrate(dict(fd))
        row = {key: migrated_spectral.pop(key) for key in COMMON_ACTION_KEYS}
        for key in COMMON_ACTION_KEYS:
            migrated_fd.pop(key)
        row["spectral_etdrk4"] = migrated_spectral
        row["fd_bdf"] = migrated_fd
        output.append(row)
    return output


def _restore_producer_rows(
    paired_rows: list[Mapping[str, Any]],
    producer: str,
    migration: TokenMigration,
) -> list[dict[str, Any]]:
    rows = []
    for paired in paired_rows:
        row = {key: paired[key] for key in COMMON_ACTION_KEYS}
        row.update(paired[producer])
        rows.append(migration.restore_legacy(row))
    return rows


def _prompt_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted((LANE_ROOT / "prompts").glob("*.txt")):
        payload = path.read_bytes()
        rows.append(
            {
                "path": str(path.relative_to(LANE_ROOT)),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if len(rows) != 8:
        raise ValueError("expected eight protocol prompt components")
    return {
        "schema_version": "evidence_view_benchmark.prompt_manifest.v1",
        "files": rows,
        "manifest_sha256": digest(rows),
    }


def build() -> dict[str, Any]:
    migration = load_migration(LANE_ROOT / "schema/token_migration.json")
    payloads = {name: read_json(path) for name, path in SOURCE_PATHS.items()}
    for name, payload in payloads.items():
        _validate_campaign(payload, name)
    spectral_public = _by_episode(payloads["spectral_public"])
    spectral_held_out = _by_episode(payloads["spectral_held_out"])
    fd_public = _by_episode(payloads["fd_public"])
    fd_held_out = _by_episode(payloads["fd_held_out"])
    if not (
        set(spectral_public)
        == set(spectral_held_out)
        == set(fd_public)
        == set(fd_held_out)
    ):
        raise ValueError("producer/public/held-out episode coverage differs")
    official = _official_by_episode(payloads["official_24"])

    family_order: list[str] = []
    selected: list[tuple[str, int, str, float, str]] = []
    seed_set = set(selected_seeds())
    for episode_id, row in spectral_held_out.items():
        family, seed, noise_model, noise_level = row["owner_episode_key"]
        if family not in family_order:
            family_order.append(family)
        if seed in seed_set:
            selected.append((family, int(seed), noise_model, float(noise_level), episode_id))
    if len(family_order) != 6:
        raise ValueError("source must contain six families")
    family_rank = {family: index for index, family in enumerate(family_order)}
    noise_rank = {name: index for index, name in enumerate(NOISE_MODELS)}
    level_rank = {level: index for index, level in enumerate(NOISE_LEVELS)}
    selected.sort(
        key=lambda row: (
            family_rank[row[0]],
            row[1],
            noise_rank[row[2]],
            level_rank[row[3]],
        )
    )
    if len(selected) != 216:
        raise ValueError(f"expected 216 selected cases, got {len(selected)}")

    next_new_id = 24
    public_cases: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    full_strata: Counter[tuple[str, str, float, bool]] = Counter()
    collapsed_strata: Counter[tuple[str, float, bool]] = Counter()
    official_exact = 0
    for family, seed, noise_model, noise_level, episode_id in selected:
        official_row = official.get(episode_id)
        if official_row is None:
            case_id = f"case_{next_new_id:03d}"
            next_new_id += 1
        else:
            case_id = str(official_row["case_id"])
        public_s = spectral_public[episode_id]
        public_f = fd_public[episode_id]
        held_s = spectral_held_out[episode_id]
        held_f = fd_held_out[episode_id]
        if held_s["owner_episode_key"] != held_f["owner_episode_key"]:
            raise ValueError("producer held-out keys differ")
        if held_s["truth_support"] != held_f["truth_support"]:
            raise ValueError("producer truth supports differ")
        rows = _pair_rows(public_s["actions"], public_f["actions"], migration)
        body: dict[str, Any] = {
            "schema_version": "evidence_view_benchmark.public_case.v1",
            "case_id": case_id,
            "family": f"F{family_rank[family]:02d}",
            "noise_model": noise_model,
            "noise_level": noise_level,
            "seed": seed,
            "original_24_subset": official_row is not None,
            "legacy_case_id": None if official_row is None else official_row["case_id"],
            "legacy_prompt_sha256": (
                None if official_row is None else official_row["prompt_sha256"]
            ),
            "public_rows": rows,
            "source_bindings": {
                "spectral_etdrk4": {
                    "episode_id": episode_id,
                    "public_episode_sha256": public_s["episode_sha256"],
                },
                "fd_bdf": {
                    "episode_id": episode_id,
                    "public_episode_sha256": public_f["episode_sha256"],
                },
            },
        }
        _assert_public(body)
        body["case_sha256"] = digest(body)
        public_cases.append(body)

        truth_support = held_s["truth_support"]
        in_library = truth_support is not None
        family_current = migration.migrate(family)
        sidecar_body: dict[str, Any] = {
            "schema_version": "evidence_view_benchmark.held_out_case.v1",
            "case_id": case_id,
            "public_case_sha256": body["case_sha256"],
            "held_out": {
                "family": family_current,
                "family_id": body["family"],
                "truth_support": truth_support,
                "in_library": in_library,
            },
            "reference": {
                "spectral_etdrk4": {
                    "nodes": 49,
                    "temporal_modes": 5,
                    "final": migration.migrate(held_s["reference_final"]),
                    "held_out_correct": held_s["reference_correct"],
                    "false_promotion": held_s["reference_false_promotion"],
                },
                "fd_bdf": {
                    "nodes": 49,
                    "temporal_modes": 5,
                    "final": migration.migrate(held_f["reference_final"]),
                    "held_out_correct": held_f["reference_correct"],
                    "false_promotion": held_f["reference_false_promotion"],
                },
            },
            "source_bindings": {
                "spectral_held_out_campaign_sha256": payloads["spectral_held_out"][
                    "campaign_sha256"
                ],
                "fd_held_out_campaign_sha256": payloads["fd_held_out"][
                    "campaign_sha256"
                ],
                "episode_id": episode_id,
            },
        }
        sidecar_body["sidecar_sha256"] = digest(sidecar_body)
        sidecars.append(sidecar_body)
        full_strata[(body["family"], noise_model, noise_level, in_library)] += 1
        collapsed_strata[(noise_model, noise_level, in_library)] += 1

        if official_row is not None:
            if official_row["bindings"] != body["source_bindings"]:
                raise ValueError(f"official source binding mismatch for {case_id}")
            restored_s = _restore_producer_rows(rows, "spectral_etdrk4", migration)
            restored_f = _restore_producer_rows(rows, "fd_bdf", migration)
            request = official_row["request"]
            if restored_s != request["spectral_etdrk4_actions"]:
                raise ValueError(f"official spectral rows differ for {case_id}")
            if restored_f != request["fd_bdf_actions"]:
                raise ValueError(f"official FD rows differ for {case_id}")
            official_exact += 1
    if next_new_id != 216 or official_exact != 24:
        raise ValueError("case IDs or official subset coverage are incomplete")
    if len({row["case_id"] for row in public_cases}) != 216:
        raise ValueError("case IDs are not unique")

    full_rows = []
    for family_id in sorted({row[0] for row in full_strata}):
        for noise_model in NOISE_MODELS:
            for noise_level in NOISE_LEVELS:
                full_rows.append(
                    {
                        "family": family_id,
                        "noise_model": noise_model,
                        "noise_level": noise_level,
                        "in_library": full_strata[
                            (family_id, noise_model, noise_level, True)
                        ],
                        "hyperdiffusion": full_strata[
                            (family_id, noise_model, noise_level, False)
                        ],
                        "total": sum(
                            full_strata[(family_id, noise_model, noise_level, label)]
                            for label in (True, False)
                        ),
                    }
                )
    collapsed_rows = []
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:
            collapsed_rows.append(
                {
                    "noise_model": noise_model,
                    "noise_level": noise_level,
                    "in_library": collapsed_strata[(noise_model, noise_level, True)],
                    "hyperdiffusion": collapsed_strata[
                        (noise_model, noise_level, False)
                    ],
                    "total": sum(
                        collapsed_strata[(noise_model, noise_level, label)]
                        for label in (True, False)
                    ),
                }
            )
    counts = {
        "schema_version": "evidence_view_benchmark.strata_counts.v1",
        "selected_seeds": list(selected_seeds()),
        "cases": 216,
        "in_library": sum(row["in_library"] for row in collapsed_rows),
        "hyperdiffusion": sum(row["hyperdiffusion"] for row in collapsed_rows),
        "original_24_subset": official_exact,
        "family_noise_level_strata": full_rows,
        "noise_level_strata": collapsed_rows,
    }
    prompt_manifest = _prompt_manifest()
    source_bindings = {
        "schema_version": "evidence_view_benchmark.source_bindings.v1",
        "sampler_salt": SAMPLER_SALT,
        "selected_seeds": list(selected_seeds()),
        "inputs": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "file_sha256": file_sha256(path),
                "campaign_sha256": payloads[name]["campaign_sha256"],
            }
            for name, path in SOURCE_PATHS.items()
        },
    }
    cases_path = LANE_ROOT / "cases/cases.jsonl"
    sidecar_path = LANE_ROOT / "cases/held_out_sidecar.jsonl"
    publish_jsonl(cases_path, public_cases)
    publish_jsonl(sidecar_path, sidecars)
    publish_json(LANE_ROOT / "cases/strata_counts.json", counts)
    publish_json(LANE_ROOT / "cases/source_bindings.json", source_bindings)
    publish_json(LANE_ROOT / "prompts/manifest.json", prompt_manifest)
    dataset_manifest = {
        "schema_version": "evidence_view_benchmark.dataset_manifest.v1",
        "cases_file_sha256": file_sha256(cases_path),
        "held_out_sidecar_file_sha256": file_sha256(sidecar_path),
        "strata_counts_sha256": file_sha256(LANE_ROOT / "cases/strata_counts.json"),
        "source_bindings_sha256": file_sha256(LANE_ROOT / "cases/source_bindings.json"),
        "prompt_manifest_sha256": file_sha256(LANE_ROOT / "prompts/manifest.json"),
        "case_count": len(public_cases),
        "sidecar_count": len(sidecars),
    }
    publish_json(LANE_ROOT / "cases/dataset_manifest.json", dataset_manifest)
    return dataset_manifest

