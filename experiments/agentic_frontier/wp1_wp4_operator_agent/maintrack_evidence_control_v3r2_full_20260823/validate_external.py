"""Producer-free structural and owner-score validation for external transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contract import DIAGNOSE_VIEW_ID, NOISE_CELLS, VIEW_BANK
from .external_transfer import owner_score


VIEW_IDS = {view.view_id for view in VIEW_BANK}


def canonical_digest(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("content_sha256", None)
    return hashlib.sha256(json.dumps(
        clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(payload: dict, manifest: dict, receipt: dict) -> dict:
    errors: list[str] = []
    receipt_clone = dict(receipt)
    receipt_clone.pop("receipt_sha256", None)
    if receipt.get("receipt_sha256") != hashlib.sha256(json.dumps(
            receipt_clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        errors.append("receipt semantic digest")
    if payload.get("schema_version") != "maintrack.external-fhn-transfer.v1":
        errors.append("schema version")
    if payload.get("content_sha256") != canonical_digest(payload):
        errors.append("semantic digest")
    # The external artifact binds the exact manifest bytes rather than a
    # re-serialization.  That check is intentionally performed by ``main``
    # below, where the original manifest path is available.
    if payload.get("model_receipt_sha256") != receipt.get("receipt_sha256"):
        errors.append("model receipt binding")
    fixed_submit = receipt.get("matched_fixed", {}).get("submit")
    if payload.get("fixed_policy") != {
            "diagnose": DIAGNOSE_VIEW_ID, "submit": fixed_submit}:
        errors.append("fixed policy binding")

    paths = {item["path"] for item in manifest["files"]}
    expected = {
        (path, f"{group:04d}", target, noise_model, noise_level)
        for path in paths for group in range(5) for target in ("u", "v")
        for noise_model, noise_level in NOISE_CELLS
    }
    rows = payload.get("rows", [])
    observed = set()
    for row in rows:
        key = (row.get("path"), row.get("group"), row.get("target"),
               row.get("noise_model"), row.get("noise_level"))
        if key in observed:
            errors.append("duplicate external cell")
        observed.add(key)
        if row.get("split") != str(row.get("path", "")).split("/")[-2]:
            errors.append("split/path mismatch")
        if row.get("agent_view") is not None and row.get("agent_view") not in VIEW_IDS:
            errors.append("unknown agent view")
        if row.get("fixed_view") != fixed_submit:
            errors.append("fixed row binding")
        agent_correct, agent_false = owner_score(
            {"status": row.get("agent_status"), "support": row.get("agent_support")},
            row.get("target"))
        fixed_correct, fixed_false = owner_score(
            {"status": row.get("fixed_status"), "support": row.get("fixed_support")},
            row.get("target"))
        if (row.get("agent_correct"), row.get("agent_false")) != (
                agent_correct, agent_false):
            errors.append("agent owner score")
        if (row.get("fixed_correct"), row.get("fixed_false")) != (
                fixed_correct, fixed_false):
            errors.append("fixed owner score")
    if observed != expected:
        errors.append("external factorial cross")
    if payload.get("n") != len(expected) or len(rows) != len(expected):
        errors.append("external row count")
    for policy in ("agent", "fixed"):
        summary = payload.get(policy, {})
        if summary.get("correct") != sum(row[f"{policy}_correct"] for row in rows):
            errors.append(f"{policy} correct summary")
        if summary.get("false") != sum(row[f"{policy}_false"] for row in rows):
            errors.append(f"{policy} false summary")
    return {"valid": not errors, "errors": sorted(set(errors)),
            "rows": len(rows), "expected_rows": len(expected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text())
    manifest = json.loads(args.manifest.read_text())
    receipt = json.loads(args.receipt.read_text())
    report = validate(payload, manifest, receipt)
    if payload.get("manifest_sha256") != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        report["errors"].append("manifest byte binding")
        report["valid"] = False
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
