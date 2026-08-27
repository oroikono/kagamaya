#!/usr/bin/env python3
"""Build the deterministic release manifest; excludes Git and generated outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", "reproduction"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUT:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        paths.append(relative)
    material = "".join(f"{digest(ROOT / relative)}  {relative.as_posix()}\n" for relative in sorted(paths))
    OUT.write_text(material)
    print(f"MANIFEST_WRITTEN files={len(paths)}")


if __name__ == "__main__":
    main()

