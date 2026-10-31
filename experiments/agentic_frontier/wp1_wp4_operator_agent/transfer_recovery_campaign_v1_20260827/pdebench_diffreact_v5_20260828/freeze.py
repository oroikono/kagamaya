"""Exact source-closure freeze for the prospective PDEBench lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

from .contract import CONTRACT_SHA256, SCHEMA

ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "PREOUTCOME_FREEZE.json"
SUFFIXES = {".py", ".json", ".md", ".sbatch"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest() -> dict[str, str]:
    result = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == FREEZE or "__pycache__" in path.parts:
            continue
        if path.suffix not in SUFFIXES:
            continue
        result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result

def closure(files: dict[str, str]) -> str:
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def create() -> dict:
    if FREEZE.exists():
        raise SystemExit("freeze already exists")
    files = manifest()
    value = {
        "schema": SCHEMA,
        "contract_sha256": CONTRACT_SHA256,
        "source_files": files,
        "source_closure_sha256": closure(files),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    FREEZE.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return value

def verify() -> dict:
    value = json.loads(FREEZE.read_text())
    if value.get("schema") != SCHEMA or value.get("contract_sha256") != CONTRACT_SHA256:
        raise SystemExit("freeze semantic binding mismatch")
    current = manifest()
    if value.get("source_files") != current:
        raise SystemExit("source manifest drift")
    if value.get("source_closure_sha256") != closure(current):
        raise SystemExit("source closure drift")
    return value

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("create", "verify"))
    args = p.parse_args(argv)
    value = create() if args.command == "create" else verify()
    print(json.dumps(value, sort_keys=True))

if __name__ == "__main__":
    main()
