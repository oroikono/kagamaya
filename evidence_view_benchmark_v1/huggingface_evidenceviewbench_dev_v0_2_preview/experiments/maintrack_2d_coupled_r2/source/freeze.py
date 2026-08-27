"""One-time source-closure creator and read-only verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE_PATH = ROOT / "PREOUTCOME_FREEZE.json"
SOURCE_FILES = (
    "__init__.py", "contract.py", "owner_truth.py", "trajectory.py",
    "numerical_spectral.py", "numerical_fd.py", "weak_evidence.py",
    "authority.py", "environment.py", "policies.py", "dominance.py",
    "campaign.py", "validate.py", "freeze.py",
    "tests/__init__.py", "tests/test_protocol.py",
    "PROGRAMME_CONTRACT_PREIMPLEMENTATION.md", "DECISIONS.md",
    "R2_NUMERICAL_REPAIR.md",
    "docs/interface_contract.md", "docs/data_contract.md",
    "docs/evaluation_protocol.md", "docs/reproducibility_protocol.md",
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    return {name: file_sha(ROOT / name) for name in SOURCE_FILES}


def content_digest(manifest: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def create() -> dict:
    if FREEZE_PATH.exists():
        raise FileExistsError("pre-outcome freeze already exists")
    manifest = source_manifest()
    payload = {
        "schema_version": "maintrack.2d-coupled-freeze.v2",
        "source_manifest": manifest,
        "source_closure_sha256": content_digest(manifest),
        "opened_splits": ["design_preflight_only"],
        "design_preflight_executed": True,
        "scientific_campaign_outcomes_opened": False,
        "learning_authorized": False,
        "confirmation_authorized": False,
        "paid_api_authorized": False,
    }
    FREEZE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify() -> dict:
    payload = json.loads(FREEZE_PATH.read_text())
    actual = source_manifest()
    errors = []
    if payload.get("source_manifest") != actual:
        errors.append("source_manifest_mismatch")
    if payload.get("source_closure_sha256") != content_digest(actual):
        errors.append("source_closure_digest_mismatch")
    return {"valid": not errors, "errors": errors,
            "source_closure_sha256": content_digest(actual)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "verify"))
    args = parser.parse_args()
    result = create() if args.command == "create" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "verify" and not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
