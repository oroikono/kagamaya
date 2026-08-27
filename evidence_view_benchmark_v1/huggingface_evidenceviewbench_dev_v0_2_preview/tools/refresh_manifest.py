"""Regenerate MANIFEST.sha256 deterministically for the release tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", "build", "dist"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != MANIFEST
        and not any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts)
    )


def main() -> None:
    paths = sorted(path for path in ROOT.rglob("*") if included(path))
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(ROOT)}" for path in paths]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} entries to {MANIFEST}")


if __name__ == "__main__":
    main()
