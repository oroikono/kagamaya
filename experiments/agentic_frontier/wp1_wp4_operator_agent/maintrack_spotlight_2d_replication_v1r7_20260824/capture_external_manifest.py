"""Mechanically capture pinned The Well and generator provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


DATASET_ID = "polymathic-ai/gray_scott_reaction_diffusion"
DATASET_REVISION = "913ef5679d5c4b4ccd6fcd52c055509dfbf190f9"
GENERATOR_REPOSITORY = "danfortunato/spectral-gray-scott"
GENERATOR_REVISION = "528fde26d54f2c05456fed78e479b88651647090"
GENERATOR_FILES = ("gen_gs.m", "init_fourier.m", "random_gaussians.m", "LICENSE")
EXPECTED_SPLITS = ("valid", "test")
EXPECTED_REGIMES = ("bubbles", "gliders", "maze", "spirals", "spots", "worms")


def _read(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def capture() -> dict:
    api_url = (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/"
        f"{DATASET_REVISION}?blobs=true"
    )
    api_raw = _read(api_url)
    api = json.loads(api_raw)
    if api.get("sha") != DATASET_REVISION:
        raise RuntimeError("dataset revision drift")
    if "license:cc-by-4.0" not in api.get("tags", []):
        raise RuntimeError("dataset license drift")
    files = []
    for item in api.get("siblings", []):
        path = str(item.get("rfilename", ""))
        if not path.endswith(".hdf5"):
            continue
        parts = path.split("/")
        if len(parts) != 3 or parts[1] not in EXPECTED_SPLITS:
            continue
        regime = next((name for name in EXPECTED_REGIMES if f"_{name}_" in path), None)
        if regime is None or "lfs" not in item:
            raise RuntimeError(f"unrecognized external file {path}")
        files.append({
            "path": path,
            "split": parts[1],
            "regime": regime,
            "bytes": int(item["lfs"]["size"]),
            "sha256": str(item["lfs"]["sha256"]),
            "blob_id": str(item["blobId"]),
        })
    files.sort(key=lambda item: (item["split"], item["regime"], item["path"]))
    expected = {(split, regime) for split in EXPECTED_SPLITS for regime in EXPECTED_REGIMES}
    if {(item["split"], item["regime"]) for item in files} != expected:
        raise RuntimeError("external split/regime cross mismatch")

    metadata_url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/gray_scott_reaction_diffusion.yaml"
    )
    metadata_raw = _read(metadata_url)
    generator = []
    for name in GENERATOR_FILES:
        raw_url = (
            f"https://raw.githubusercontent.com/{GENERATOR_REPOSITORY}/"
            f"{GENERATOR_REVISION}/{name}"
        )
        raw = _read(raw_url)
        generator.append({"path": name, "sha256": _sha(raw), "bytes": len(raw)})
    return {
        "schema_version": "maintrack.external-well-manifest.v1",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": "CC-BY-4.0",
        "dataset_api_url": api_url,
        "dataset_api_sha256": _sha(api_raw),
        "dataset_metadata_path": "gray_scott_reaction_diffusion.yaml",
        "dataset_metadata_sha256": _sha(metadata_raw),
        "external_role": "independently_generated_numerical_transfer",
        "scientific_authority": "published_equation_support_and_library_interventions",
        "generator_repository": GENERATOR_REPOSITORY,
        "generator_revision": GENERATOR_REVISION,
        "generator_license": "MIT",
        "generator_files": generator,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "selection_rule": "all_six_publisher_valid_and_all_six_publisher_test_shards",
        "claim_ceiling": (
            "independent numerical data origin; local weak-form adapter and evaluator; "
            "not independent evaluator or physical validation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = capture()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "files": len(payload["files"]),
        "total_bytes": payload["total_bytes"],
        "dataset_api_sha256": payload["dataset_api_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
