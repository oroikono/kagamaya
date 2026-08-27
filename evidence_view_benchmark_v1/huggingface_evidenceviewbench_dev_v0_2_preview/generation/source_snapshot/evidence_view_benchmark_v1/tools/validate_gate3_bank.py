#!/usr/bin/env python3
"""Independent structural validation for the Gate-3 acquisition bank."""

from __future__ import annotations

import json
import sys
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


PRODUCERS = ("spectral_etdrk4", "fd_bdf")
MODES = (2, 3, 4, 5)
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


def _forbidden_paths(value: Any, location: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN):
                findings.append(f"{location}.{key}")
            findings.extend(_forbidden_paths(nested, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            findings.extend(_forbidden_paths(nested, f"{location}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in FORBIDDEN):
            findings.append(location)
    return findings


def validate() -> dict[str, Any]:
    cases = read_jsonl(LANE_ROOT / "cases/cases.jsonl")
    sidecars = read_jsonl(LANE_ROOT / "cases/held_out_sidecar.jsonl")
    bank = read_jsonl(LANE_ROOT / "cases/acquisition_49_rows.jsonl")
    manifest_path = LANE_ROOT / "cases/acquisition_bank_manifest.json"
    manifest = read_json(manifest_path)
    case_by_id = {row["case_id"]: row for row in cases}
    sidecar_by_id = {row["case_id"]: row for row in sidecars}
    bank_by_id = {row["case_id"]: row for row in bank}
    errors: list[str] = []

    if len(case_by_id) != 216 or len(sidecar_by_id) != 216 or len(bank_by_id) != 216:
        errors.append("case/sidecar/bank cardinality is not 216")
    if set(case_by_id) != set(sidecar_by_id) or set(case_by_id) != set(bank_by_id):
        errors.append("case IDs differ across artifacts")
    if manifest.get("acquisition_bank_file_sha256") != file_sha256(
        LANE_ROOT / "cases/acquisition_49_rows.jsonl"
    ):
        errors.append("bank file hash mismatch")
    if manifest.get("cases_file_sha256") != file_sha256(LANE_ROOT / "cases/cases.jsonl"):
        errors.append("case file hash mismatch")
    manifest_body = dict(manifest)
    claimed_manifest_hash = manifest_body.pop("manifest_sha256", None)
    if claimed_manifest_hash != digest(manifest_body):
        errors.append("manifest content hash mismatch")

    view_count = 0
    mode5_owner_matches = 0
    for case_id in sorted(set(case_by_id) & set(sidecar_by_id) & set(bank_by_id)):
        case, sidecar, row = case_by_id[case_id], sidecar_by_id[case_id], bank_by_id[case_id]
        if row.get("public_case_sha256") != case.get("case_sha256"):
            errors.append(f"{case_id}: public case binding mismatch")
        body = dict(row)
        claimed = body.pop("acquisition_case_sha256", None)
        if claimed != digest(body):
            errors.append(f"{case_id}: acquisition case hash mismatch")
        findings = _forbidden_paths(body)
        if findings:
            errors.append(f"{case_id}: forbidden public paths {findings[:3]}")
        views = row.get("views", [])
        coordinates = {(int(view["nodes"]), int(view["temporal_modes"])) for view in views}
        if coordinates != {(49, mode) for mode in MODES} or len(views) != 4:
            errors.append(f"{case_id}: node-49 view grid mismatch")
        view_count += len(views)
        by_mode = {int(view["temporal_modes"]): view for view in views}
        if 5 not in by_mode:
            continue
        for producer in PRODUCERS:
            public = by_mode[5][producer]
            locked = sidecar["reference"][producer]["final"]
            if public["terminal_status"] != locked["status"]:
                errors.append(f"{case_id}.{producer}: mode-5 status mismatch")
            elif public["terminal_support"] != locked["support"]:
                errors.append(f"{case_id}.{producer}: mode-5 support mismatch")
            else:
                mode5_owner_matches += 1

    report = {
        "schema_version": "evidence_view_benchmark.gate3_bank_validation.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "case_count": len(bank),
        "view_count": view_count,
        "producer_views": view_count * len(PRODUCERS),
        "mode5_locked_status_support_matches": mode5_owner_matches,
        "expected_mode5_locked_status_support_matches": 216 * len(PRODUCERS),
        "bank_file_sha256": file_sha256(LANE_ROOT / "cases/acquisition_49_rows.jsonl"),
        "manifest_file_sha256": file_sha256(manifest_path),
    }
    report["report_sha256"] = digest(report)
    publish_json(LANE_ROOT / "results/gate3_bank_validation.json", report)
    if errors:
        raise SystemExit(json.dumps(report, sort_keys=True, indent=2))
    return report


def main() -> None:
    print(json.dumps(validate(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

