#!/usr/bin/env python3
"""Independent Gate-1 artifact validator."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


LANE_ROOT = Path(__file__).resolve().parents[1]
SRC = LANE_ROOT / "src"
sys.path.insert(0, str(SRC))

from evidence_view_benchmark.canonical import (  # noqa: E402
    digest,
    file_sha256,
    publish_json,
    read_json,
    read_jsonl,
)
from evidence_view_benchmark.migration import load_migration  # noqa: E402


REPO_ROOT = LANE_ROOT.parent
PREDECESSOR = (
    REPO_ROOT
    / "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1"
)
CURRENT_STATUSES = {"IDENTIFIED", "OUTSIDE_LIBRARY", "INCONCLUSIVE"}
FORBIDDEN = (
    "truth",
    "held_out",
    "correct",
    "false",
    "owner",
    "oracle",
    "reference",
    "error_to_49",
)
COMMON = ("action_token", "nodes", "temporal_modes")


def _assert_no_forbidden(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN):
                raise ValueError(f"forbidden public key {path}.{key}")
            _assert_no_forbidden(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_forbidden(nested, f"{path}[{index}]")


def _validate_source_manifest() -> str:
    payload = read_json(LANE_ROOT / "SOURCE_MANIFEST.json")
    for row in payload["files"]:
        path = LANE_ROOT / row["path"]
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"source closure mismatch: {path}")
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"source size mismatch: {path}")
    if digest(payload["files"]) != payload["closure_sha256"]:
        raise ValueError("source closure digest mismatch")
    return payload["closure_sha256"]


def _validate_prompt_manifest() -> str:
    payload = read_json(LANE_ROOT / "prompts/manifest.json")
    rows = payload["files"]
    if len(rows) != 8 or digest(rows) != payload["manifest_sha256"]:
        raise ValueError("prompt manifest malformed")
    for row in rows:
        path = LANE_ROOT / row["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["bytes"]:
            raise ValueError(f"prompt digest mismatch: {path}")
        text = raw.decode("utf-8")
        if "OUT_OF_GRAMMAR" in text or "UNIDENTIFIED" in text:
            raise ValueError(f"legacy status in current prompt: {path}")
    return payload["manifest_sha256"]


def _restore_rows(case: Mapping[str, Any], producer: str, migration) -> list[dict]:
    output = []
    for paired in case["public_rows"]:
        row = {key: paired[key] for key in COMMON}
        row.update(paired[producer])
        output.append(migration.restore_legacy(row))
    return output


def _official_cases() -> Mapping[str, Any]:
    path = PREDECESSOR / "results/confirmation/llm/official_campaign.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    body = dict(payload)
    claimed = body.pop("campaign_sha256")
    if claimed != digest(body):
        raise ValueError("official campaign digest mismatch")
    return {row["case_id"]: row for row in payload["cases"]}


def validate() -> dict[str, Any]:
    source_closure = _validate_source_manifest()
    prompt_digest = _validate_prompt_manifest()
    migration = load_migration(LANE_ROOT / "schema/token_migration.json")
    cases = read_jsonl(LANE_ROOT / "cases/cases.jsonl")
    sidecars = read_jsonl(LANE_ROOT / "cases/held_out_sidecar.jsonl")
    if len(cases) != 216 or len(sidecars) != 216:
        raise ValueError("Gate-1 case/sidecar count must be 216")
    cases_by_id = {row["case_id"]: row for row in cases}
    sidecars_by_id = {row["case_id"]: row for row in sidecars}
    if len(cases_by_id) != 216 or set(cases_by_id) != set(sidecars_by_id):
        raise ValueError("case IDs are duplicated or sidecar coverage differs")
    full_counts: Counter[tuple[str, str, float]] = Counter()
    class_counts: Counter[bool] = Counter()
    for case_id, case in cases_by_id.items():
        _assert_no_forbidden(case)
        body = dict(case)
        claimed = body.pop("case_sha256")
        if claimed != digest(body):
            raise ValueError(f"public case digest mismatch: {case_id}")
        if not isinstance(case["family"], str) or not case["family"].startswith("F"):
            raise ValueError("public family must be opaque")
        if len(case["public_rows"]) != 12:
            raise ValueError("public case must contain twelve paired rows")
        for index, row in enumerate(case["public_rows"]):
            if row["action_token"] != f"A{index:02d}":
                raise ValueError("actions are not in canonical order")
            for producer in ("spectral_etdrk4", "fd_bdf"):
                status = row[producer]["terminal_status"]
                if status not in CURRENT_STATUSES:
                    raise ValueError(f"legacy or unknown public status: {status}")
        sidecar = sidecars_by_id[case_id]
        sidecar_body = dict(sidecar)
        sidecar_claimed = sidecar_body.pop("sidecar_sha256")
        if sidecar_claimed != digest(sidecar_body):
            raise ValueError(f"sidecar digest mismatch: {case_id}")
        if sidecar["public_case_sha256"] != case["case_sha256"]:
            raise ValueError(f"case/sidecar binding mismatch: {case_id}")
        in_library = bool(sidecar["held_out"]["in_library"])
        class_counts[in_library] += 1
        full_counts[(case["family"], case["noise_model"], case["noise_level"])] += 1
    if class_counts != Counter({True: 180, False: 36}):
        raise ValueError(f"incorrect class counts: {class_counts}")
    if len(full_counts) != 36 or set(full_counts.values()) != {6}:
        raise ValueError("36 strata must each contain six cases")

    official = _official_cases()
    exact = 0
    for case_id, old in official.items():
        case = cases_by_id.get(case_id)
        if case is None or case["original_24_subset"] is not True:
            raise ValueError(f"official case is missing or unflagged: {case_id}")
        if case["source_bindings"] != old["bindings"]:
            raise ValueError(f"official binding changed: {case_id}")
        if case["legacy_prompt_sha256"] != old["prompt_sha256"]:
            raise ValueError(f"official prompt binding changed: {case_id}")
        if _restore_rows(case, "spectral_etdrk4", migration) != old["request"][
            "spectral_etdrk4_actions"
        ]:
            raise ValueError(f"official spectral public rows changed: {case_id}")
        if _restore_rows(case, "fd_bdf", migration) != old["request"]["fd_bdf_actions"]:
            raise ValueError(f"official FD public rows changed: {case_id}")
        exact += 1
    if exact != 24 or sum(row["original_24_subset"] for row in cases) != 24:
        raise ValueError("official subset count differs")

    counts = read_json(LANE_ROOT / "cases/strata_counts.json")
    if counts["cases"] != 216 or counts["in_library"] != 180:
        raise ValueError("strata summary disagrees")
    if counts["hyperdiffusion"] != 36 or counts["original_24_subset"] != 24:
        raise ValueError("strata class/subset counts disagree")
    bindings = read_json(LANE_ROOT / "cases/source_bindings.json")
    for row in bindings["inputs"].values():
        path = REPO_ROOT / row["path"]
        if file_sha256(path) != row["file_sha256"]:
            raise ValueError(f"predecessor input changed: {path}")
    replay = read_json(LANE_ROOT / "results/legacy_replay_report.json")
    if replay["status"] != "PASS" or not replay["all_file_hashes_unchanged"]:
        raise ValueError("legacy replay did not pass")
    dataset = read_json(LANE_ROOT / "cases/dataset_manifest.json")
    checks = {
        "cases_file_sha256": file_sha256(LANE_ROOT / "cases/cases.jsonl"),
        "held_out_sidecar_file_sha256": file_sha256(
            LANE_ROOT / "cases/held_out_sidecar.jsonl"
        ),
        "strata_counts_sha256": file_sha256(LANE_ROOT / "cases/strata_counts.json"),
        "source_bindings_sha256": file_sha256(LANE_ROOT / "cases/source_bindings.json"),
        "prompt_manifest_sha256": file_sha256(LANE_ROOT / "prompts/manifest.json"),
    }
    for key, value in checks.items():
        if dataset[key] != value:
            raise ValueError(f"dataset manifest mismatch: {key}")
    report = {
        "schema_version": "evidence_view_benchmark.gate1_validation.v1",
        "status": "PASS",
        "source_closure_sha256": source_closure,
        "prompt_component_manifest_digest": prompt_digest,
        "cases": 216,
        "held_out_sidecars": 216,
        "in_library": 180,
        "hyperdiffusion": 36,
        "balanced_strata": 36,
        "cases_per_stratum": 6,
        "original_24_exact_public_row_replays": exact,
        "legacy_artifacts_roundtripped": replay["stored_json_artifacts_roundtripped"],
        "legacy_file_hashes_unchanged": replay["all_file_hashes_unchanged"],
        "dataset_manifest_sha256": file_sha256(
            LANE_ROOT / "cases/dataset_manifest.json"
        ),
    }
    publish_json(LANE_ROOT / "results/gate1_validation.json", report)
    return report


def main() -> None:
    report = validate()
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

