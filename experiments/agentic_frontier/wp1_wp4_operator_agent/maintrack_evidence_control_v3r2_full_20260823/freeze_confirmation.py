"""Create the one-time confirmation authorization from passing calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contract import GATES


def semantic_digest(payload: dict, digest_key: str) -> str:
    clone = dict(payload)
    clone.pop(digest_key, None)
    return hashlib.sha256(json.dumps(
        clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dominance", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-closure", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dominance = json.loads(args.dominance.read_text())
    receipt = json.loads(args.receipt.read_text())
    model_path = args.receipt.parent / "models.pkl"
    failures = []
    if dominance.get("content_sha256") != semantic_digest(dominance, "content_sha256"):
        failures.append("dominance_digest")
    if receipt.get("receipt_sha256") != semantic_digest(receipt, "receipt_sha256"):
        failures.append("receipt_digest")
    if not dominance["gate"]["learning_admitted"]:
        failures.append("dominance_learning_gate")
    chosen = next(row for row in receipt["threshold_table"]
                  if row["threshold"] == receipt["selected_risk_threshold"])
    if (chosen["false_rate"] > GATES.risk_ceiling
            or chosen["cluster_risk"]["mean_seed_false_rate_ci95"][1]
            > GATES.risk_ceiling):
        failures.append("calibration_cluster_risk")
    if not receipt["lofo_gate"]["all_passed"]:
        failures.append("nested_lofo_gate")
    if receipt.get("matched_fixed") is None:
        failures.append("matched_fixed_missing")
    if receipt["source_closure"] != args.source_closure:
        failures.append("source_closure")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != receipt["model_sha256"]:
        failures.append("model_hash")
    payload = {
        "schema_version": "maintrack.confirmation-authorization.v3r2",
        "stage": "confirmation",
        "source_closure": args.source_closure,
        "dominance_content_sha256": dominance["content_sha256"],
        "model_receipt_sha256": receipt["receipt_sha256"],
        "model_sha256": receipt["model_sha256"],
        "selected_risk_threshold": receipt["selected_risk_threshold"],
        "matched_fixed": receipt.get("matched_fixed"),
        "lofo_gate": receipt["lofo_gate"],
        "calibration_gates_passed": not failures,
        "failures": failures,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["authorization_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
