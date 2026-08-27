"""Scan an artifact tree for an in-memory credential without exposing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .providers import read_anthropic_key


def scan(credential: Path, artifact_root: Path) -> dict:
    if artifact_root == credential or artifact_root in credential.parents:
        raise ValueError("artifact root must not contain the credential file")
    secret = read_anthropic_key(credential).encode("ascii")
    checked, violations = 0, []
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        checked += 1
        if secret in path.read_bytes():
            violations.append(str(path.relative_to(artifact_root)))
    return {
        "schema_version": "verisigs.adversarial_verifier_gaming.v1.secret_scan",
        "artifact_root": str(artifact_root),
        "files_checked": checked,
        "credential_recorded_or_hashed": False,
        "violation_count": len(violations),
        "violation_paths": violations,
        "pass": not violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = scan(args.credential.resolve(), args.artifact_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not result["pass"]:
        raise RuntimeError("credential material was found in the artifact tree")
    print(f"PASS files_checked={result['files_checked']}")


if __name__ == "__main__":
    main()
