#!/usr/bin/env python3
"""Build the deterministic release manifest; excludes Git and generated outputs."""
from __future__ import annotations

from pathlib import Path

if __package__:
    from .manifest_common import ManifestError, digest, inventory
else:
    from manifest_common import ManifestError, digest, inventory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MANIFEST.sha256"


def main() -> None:
    try:
        paths = inventory(ROOT, OUT.relative_to(ROOT).as_posix())
        material = "".join(f"{digest(path)}  {relative}\n" for relative, path in sorted(paths.items()))
        # Never replace or follow a symlink/special manifest destination.
        if OUT.is_symlink() or (OUT.exists() and not OUT.is_file()):
            raise ManifestError("unsafe manifest destination")
        OUT.write_text(material, encoding="utf-8")
    except (OSError, ValueError) as error:
        raise SystemExit(f"MANIFEST_FAIL\n{error}") from error
    print(f"MANIFEST_WRITTEN files={len(paths)}")


if __name__ == "__main__":
    main()
