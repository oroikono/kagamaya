"""Fail-closed structural and replay validation for the release package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .data import (
    DATA_ROOT,
    canonical_bytes,
    digest,
    file_sha256,
    load_cases,
    load_labels,
    load_observations,
    load_view_bank,
    read_jsonl,
)
from .environment import Action, EvidenceViewEnv, forbidden_paths
from .scoring import score_trace


SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)anthropic_api_key\s*[:=]\s*[^\s]+"),
    re.compile(r"(?i)openai_api_key\s*[:=]\s*[^\s]+"),
    re.compile(r"(?i)google_api_key\s*[:=]\s*[^\s]+"),
)


def _validate_content_hash(record: dict[str, Any], field: str) -> None:
    body = dict(record)
    claimed = body.pop(field, None)
    if claimed != digest(body):
        raise ValueError(f"content hash mismatch for {record.get('case_id')}:{field}")


def _index(records: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate {name} case: {case_id}")
        result[case_id] = record
    return result


def _validate_trace_release(
    release_root: Path,
    case_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt_root = release_root / "prompts"
    trace_root = release_root / "traces"
    if not prompt_root.is_dir() and not trace_root.is_dir():
        return {
            "present": False,
            "prompt_inventory_rows": 0,
            "raw_trace_rows": 0,
            "prompt_join_missing": 0,
            "invalid_internal_hashes": 0,
        }
    if not prompt_root.is_dir() or not trace_root.is_dir():
        raise ValueError("prompt and trace directories must be published together")

    prompt_hashes: set[str] = set()
    prompt_hashes_by_file: dict[str, set[str]] = {}
    inventory_rows = 0
    for path in sorted(prompt_root.glob("*.jsonl")):
        file_hashes: set[str] = set()
        for row in read_jsonl(path):
            inventory_rows += 1
            text = row.get("prompt_text")
            claimed = row.get("prompt_sha256")
            if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != claimed:
                raise ValueError(f"prompt text hash mismatch in {path}")
            case_id = str(row["case_id"])
            if case_id not in case_index or row.get("case_sha256") != case_index[case_id]["case_sha256"]:
                raise ValueError(f"prompt/case binding mismatch in {path}:{case_id}")
            prompt_hashes.add(str(claimed))
            file_hashes.add(str(claimed))
        prompt_hashes_by_file[path.name] = file_hashes

    raw_rows = 0
    prompt_join_missing = 0
    raw_files = (
        ("claude_goal_conditioned_normalized.jsonl", "claude_prompt_inventory.jsonl"),
        ("claude_row_order_raw.jsonl", "claude_row_order_prompt_inventory.jsonl"),
        ("claude_m5_repair_raw.jsonl", "claude_m5_repair_prompt_inventory.jsonl"),
    )
    for filename, inventory_name in raw_files:
        path = trace_root / filename
        if not path.is_file():
            raise ValueError(f"missing published trace file: {filename}")
        for row in read_jsonl(path):
            raw_rows += 1
            _validate_content_hash(row, "record_sha256")
            if row.get("prompt_sha256") not in prompt_hashes_by_file.get(inventory_name, set()):
                prompt_join_missing += 1
            case_id = str(row["case_id"])
            if case_id not in case_index or row.get("case_sha256") != case_index[case_id]["case_sha256"]:
                raise ValueError(f"trace/case binding mismatch in {filename}:{case_id}")
            if row.get("credential_recorded_or_hashed") is not False:
                raise ValueError(f"credential material flag is not false in {filename}")
    if prompt_join_missing:
        raise ValueError(f"{prompt_join_missing} raw trace prompts are absent from inventories")

    checked_json_hashes = 0
    for path in sorted(trace_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "content_sha256" not in payload:
            continue
        body = dict(payload)
        claimed = body.pop("content_sha256")
        # This validation record binds the raw Qwen payload in content_sha256;
        # its own digest is carried by validation_sha256.
        if path.name == "qwen_multistep_validation.json":
            raw = json.loads((trace_root / "qwen_multistep_raw.json").read_text(encoding="utf-8"))
            if claimed != raw.get("content_sha256"):
                raise ValueError("Qwen validation/raw binding mismatch")
            continue
        if claimed != digest(body):
            raise ValueError(f"JSON content hash mismatch in {path}")
        checked_json_hashes += 1

    return {
        "present": True,
        "prompt_inventory_rows": inventory_rows,
        "unique_prompt_hashes": len(prompt_hashes),
        "raw_trace_rows": raw_rows,
        "prompt_join_missing": 0,
        "invalid_internal_hashes": 0,
        "checked_json_content_hashes": checked_json_hashes,
    }


def validate_release(root: str | Path | None = None) -> dict[str, Any]:
    release_root = Path(root).resolve() if root is not None else DATA_ROOT.parent
    data_root = release_root / "data" if root is not None else DATA_ROOT
    cases_path = data_root / "dev_cases.jsonl"
    observations_path = data_root / "dev_observations.jsonl"
    labels_path = data_root / "dev_labels.jsonl"
    bank_path = data_root / "view_bank.jsonl"
    cases = load_cases(cases_path)
    observations = load_observations(observations_path)
    labels = load_labels(labels_path)
    banks = load_view_bank(bank_path)
    if not (len(cases) == len(observations) == len(labels) == len(banks) == 216):
        raise ValueError("expected 216 cases, observations, labels, and view-bank rows")

    case_index = _index(cases, "case")
    observation_index = _index(observations, "observation")
    label_index = _index(labels, "label")
    bank_index = _index(banks, "view-bank")
    if not (set(case_index) == set(observation_index) == set(label_index) == set(bank_index)):
        raise ValueError("case/observation/label/view-bank key sets differ")

    for case_id, case in case_index.items():
        _validate_content_hash(case, "case_sha256")
        observation = observation_index[case_id]
        label = label_index[case_id]
        bank = bank_index[case_id]
        _validate_content_hash(observation, "selection_case_sha256")
        _validate_content_hash(label, "sidecar_sha256")
        _validate_content_hash(bank, "acquisition_case_sha256")
        if len(case.get("public_rows", [])) != 12 or len(observation.get("public_rows", [])) != 12:
            raise ValueError("each case and observation must contain 12 public rows")
        if canonical_bytes(case["public_rows"]) != canonical_bytes(observation["public_rows"]):
            raise ValueError("observation/public-row mismatch")
        if set(observation) & {"family", "noise_model", "noise_level", "seed", "source_bindings"}:
            raise ValueError("default observation exposes analysis metadata")
        if observation["parent_public_case_sha256"] != case["case_sha256"]:
            raise ValueError("observation/case binding mismatch")
        if label["public_case_sha256"] != case["case_sha256"]:
            raise ValueError("label/case binding mismatch")
        if bank["public_case_sha256"] != case["case_sha256"]:
            raise ValueError("bank/case binding mismatch")
        if len(bank.get("views", [])) != 4:
            raise ValueError("each acquisition row must add four 49-node views")

    if root is not None:
        resource_root = release_root / "evidenceviewbench" / "resources"
        for filename in ("dev_cases.jsonl", "dev_observations.jsonl", "dev_labels.jsonl", "view_bank.jsonl"):
            if file_sha256(data_root / filename) != file_sha256(resource_root / filename):
                raise ValueError(f"source/package resource mismatch: {filename}")

    # Exhaust every selector-visible case, producer, and view through the public API.
    view_checks = 0
    finite_scalars = 0
    trace_replays = 0
    terminal_counts: Counter[str] = Counter()
    smoke_scores: list[dict[str, Any]] = []
    coordinates = [(nodes, modes) for nodes in (7, 13, 25, 49) for modes in (2, 3, 4, 5)]
    first_case_id = next(iter(observation_index))
    for case_id, observation in observation_index.items():
        for producer in ("spectral_etdrk4", "fd_bdf"):
            for nodes, modes in coordinates:
                env = EvidenceViewEnv(
                    observation,
                    budget=nodes,
                    visibility="acquire_only",
                    producer=producer,
                    acquisition_bank_path=bank_path,
                )
                reset = env.reset()
                if forbidden_paths(reset) or set(reset) & {"family", "noise_model", "noise_level", "seed"}:
                    raise ValueError("metadata-blind reset leaks analysis or held-out material")
                acquired, _ = env.step(Action.acquire_view(nodes, modes))
                for field in ("residual", "score_gap", "gram_condition"):
                    if not math.isfinite(float(acquired[field])):
                        raise ValueError(f"non-finite {field} at {case_id}/{producer}/{nodes}/{modes}")
                    finite_scalars += 1
                _, done = env.step(Action.commit(nodes, modes))
                if not done:
                    raise ValueError("single-view trace did not terminate")
                trace = env.export_trace()
                replay = EvidenceViewEnv.replay_trace(
                    observation,
                    trace,
                    acquisition_bank_path=bank_path,
                )
                if canonical_bytes(trace) != canonical_bytes(replay):
                    raise ValueError("trace replay mismatch")
                terminal_counts[str(acquired["terminal"])] += 1
                view_checks += 1
                trace_replays += 1
            if case_id == first_case_id:
                smoke_scores.append(score_trace(trace, label_index[case_id]))

    unique_evidence_tables = len({digest(case["public_rows"]) for case in cases})
    seed_clusters = len({int(case["seed"]) for case in cases})
    zero_noise_pairs = 0
    families = sorted({str(case["family"]) for case in cases})
    seeds = sorted({int(case["seed"]) for case in cases})
    for family in families:
        for seed in seeds:
            pair = [
                case
                for case in cases
                if str(case["family"]) == family and int(case["seed"]) == seed and float(case["noise_level"]) == 0.0
            ]
            if len(pair) != 2 or digest(pair[0]["public_rows"]) != digest(pair[1]["public_rows"]):
                raise ValueError("zero-noise iid/correlated duplicate contract does not hold")
            zero_noise_pairs += 1

    trace_report = _validate_trace_release(release_root, case_index)

    scanned_files = 0
    for path in release_root.rglob("*"):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        if path.suffix.lower() not in {".md", ".py", ".toml", ".txt", ".json", ".jsonl", ".yml", ".yaml"}:
            continue
        scanned_files += 1
        text = path.read_text(encoding="utf-8", errors="strict")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"credential-like material found in {path}")

    return {
        "schema_version": "evidenceviewbench.release_validation.v2",
        "valid": True,
        "case_count": len(cases),
        "factorial_case_records": len(cases),
        "selection_observation_count": len(observations),
        "label_count": len(labels),
        "view_bank_count": len(banks),
        "seed_cluster_count": seed_clusters,
        "unique_public_evidence_tables": unique_evidence_tables,
        "zero_noise_iid_correlated_duplicate_pairs": zero_noise_pairs,
        "single_view_commit_checks": view_checks,
        "finite_public_scalar_checks": finite_scalars,
        "byte_identical_trace_replays": trace_replays,
        "metadata_blind_smokes": 2,
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "smoke_scores": smoke_scores,
        "published_trace_validation": trace_report,
        "scanned_text_files": scanned_files,
        "files": {
            "dev_cases_sha256": file_sha256(cases_path),
            "dev_observations_sha256": file_sha256(observations_path),
            "dev_labels_sha256": file_sha256(labels_path),
            "view_bank_sha256": file_sha256(bank_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate installed resources or a complete release checkout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="release checkout root; omit to validate only installed package resources",
    )
    args = parser.parse_args()
    print(json.dumps(validate_release(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
