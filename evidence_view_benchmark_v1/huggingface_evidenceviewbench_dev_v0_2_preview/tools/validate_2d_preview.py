#!/usr/bin/env python3
"""Fail-closed structural and scientific-summary checks for the 2-D preview."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate() -> dict[str, Any]:
    errors: list[str] = []
    data = ROOT / "data"
    public = load_jsonl_gz(data / "two_d_design_public.jsonl.gz")
    owner = load_jsonl_gz(data / "two_d_design_owner.jsonl.gz")
    convergence = load_jsonl_gz(data / "two_d_design_convergence.jsonl.gz")
    manifest = json.loads((data / "two_d_design_manifest.json").read_text(encoding="utf-8"))

    for name in ("two_d_design_public.jsonl.gz", "two_d_design_owner.jsonl.gz"):
        source = data / name
        resource = ROOT / "evidenceviewbench" / "resources" / name
        if not resource.is_file() or resource.read_bytes() != source.read_bytes():
            errors.append(f"package_resource_mismatch:{name}")

    if len(public) != 180 or len(owner) != 180 or len(convergence) != 30:
        errors.append("row_counts")
    if any(len(row.get("views", [])) != 72 for row in public):
        errors.append("view_count")
    forbidden = {"family", "truth_kind", "truth_support", "omitted_mechanism", "convergence_error"}
    if any(forbidden & set(row) for row in public):
        errors.append("owner_fields_in_public")
    for row in public:
        claimed = row.get("public_row_sha256")
        body = dict(row)
        body.pop("public_row_sha256", None)
        if claimed != digest(body):
            errors.append(f"public_digest:{row.get('episode_id')}")
    for row in owner:
        claimed = row.get("owner_row_sha256")
        body = dict(row)
        body.pop("owner_row_sha256", None)
        if claimed != digest(body):
            errors.append(f"owner_digest:{row.get('episode_id')}")
    public_keys = {(row["origin_component"], row["episode_id"]) for row in public}
    owner_keys = {(row["origin_component"], row["episode_id"]) for row in owner}
    if public_keys != owner_keys:
        errors.append("public_owner_join")
    if manifest.get("primary_unique_observation_rows_per_origin") != 75:
        errors.append("primary_denominator")

    results = ROOT / "experiments/maintrack_2d_coupled_r2/results"
    step0 = json.loads((results / "step0_analysis_v1r2.json").read_text(encoding="utf-8"))
    agent = json.loads((results / "agent_eval_v1r2.json").read_text(encoding="utf-8"))
    probe = json.loads((results / "diagnostic_submit_probe_v1.json").read_text(encoding="utf-8"))
    if step0["gates"]["feedback_gate"] is not False:
        errors.append("feedback_gate_not_blocked")
    if step0["learning_status"] != "POLICY_LEARNING_BLOCKED_FIXED_DOMINANCE_OR_SCOPE":
        errors.append("learning_decision")
    if agent["hidden_label_mutation_trace_invariant"] is not True:
        errors.append("hidden_label_mutation")
    if probe["positive_both_directions"] is not False:
        errors.append("exploratory_probe_promoted")

    registry = json.loads((ROOT / "PREVIEW_EXPERIMENT_REGISTRY.json").read_text(encoding="utf-8"))
    evidence_count = 0
    for claim in registry.get("claims", []):
        for item in claim.get("evidence", []):
            evidence_count += 1
            path = ROOT / item["path"]
            if not path.is_file():
                errors.append(f"missing_evidence:{item['path']}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != item["file_sha256"]:
                errors.append(f"evidence_hash:{item['path']}")

    return {
        "valid": not errors,
        "errors": errors,
        "public_rows": len(public),
        "owner_rows": len(owner),
        "convergence_rows": len(convergence),
        "primary_cells": 150,
        "preview_claims": len(registry.get("claims", [])),
        "preview_evidence_bindings": evidence_count,
        "agent_advantage_demonstrated": False,
    }


def main() -> None:
    report = validate()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
