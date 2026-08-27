"""Content-addressed downloader for the frozen external numerical manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


REPO = "DabbyOWL/PDE_Inverse_Problem_Benchmarking"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        target = args.output_dir / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == item["bytes"] \
                and sha256(target) == item["sha256"]:
            print(json.dumps({"cached": str(target), "sha256": item["sha256"]}), flush=True)
            continue
        if target.exists():
            raise RuntimeError(f"existing mismatched external file: {target}")
        encoded = urllib.parse.quote(item["path"], safe="/")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{encoded}?download=true"
        temporary = target.with_suffix(target.suffix + ".partial")
        if temporary.exists():
            temporary.unlink()
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
            raise RuntimeError(f"download hash/size mismatch: {item['path']}")
        temporary.replace(target)
        print(json.dumps({"downloaded": str(target), "sha256": item["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
