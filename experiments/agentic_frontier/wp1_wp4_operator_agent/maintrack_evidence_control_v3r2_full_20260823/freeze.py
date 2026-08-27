"""Create or verify the one-time pre-outcome source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "PREOUTCOME_FREEZE.json"
INCLUDED_SUFFIXES = {".py", ".md", ".sbatch"}
EXCLUDED_PARTS = {"results", "__pycache__", ".pytest_cache"}


def files() -> list[Path]:
    return [path for path in sorted(ROOT.rglob("*"))
            if path.is_file()
            and (path.suffix in INCLUDED_SUFFIXES
                 or path.name == "EXTERNAL_NUMERICAL_MANIFEST.json")
            and path.name != FREEZE.name
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closure() -> tuple[dict[str, str], str]:
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in files()}
    digest = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest, digest = closure()
    if args.create:
        if FREEZE.exists():
            raise SystemExit("freeze already exists")
        payload = {
            "schema_version": "maintrack.preoutcome-freeze.v3r2",
            "source_closure": digest,
            "files": manifest,
            "train_seeds": [61000, 61039],
            "calibration_seeds": [62000, 62019],
            "confirmation_seeds": [63000, 63099],
            "outcomes_opened_at_creation": [],
        }
        FREEZE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"created": str(FREEZE), "source_closure": digest,
                          "files": len(manifest)}, sort_keys=True))
        return
    if args.verify:
        payload = json.loads(FREEZE.read_text())
        valid = payload["files"] == manifest and payload["source_closure"] == digest
        print(json.dumps({"valid": valid, "source_closure": digest,
                          "files": len(manifest)}, sort_keys=True))
        if not valid:
            raise SystemExit(2)
        return
    raise SystemExit("select --create or --verify")


if __name__ == "__main__":
    main()
