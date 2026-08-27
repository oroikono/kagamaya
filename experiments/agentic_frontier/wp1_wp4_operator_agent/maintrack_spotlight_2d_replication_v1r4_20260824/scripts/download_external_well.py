"""Resume-safe, manifest-verified downloader for the pinned external split.

This script never decodes HDF5 outcomes.  Nevertheless, both publisher splits
require their stage-specific authorization, and TEST additionally binds the
passing publisher-VALID aggregate before any network request is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from ..authorization import validate as validate_authorization
from ..external_well import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_REGIMES,
    EXPECTED_SPLITS,
    authorization_bindings,
    file_sha256,
    load_manifest,
    manifest_entry,
)


def _download(url: str, partial: Path) -> None:
    subprocess.run([
        "curl", "--location", "--fail", "--show-error",
        "--retry", "8", "--retry-all-errors", "--continue-at", "-",
        "--output", str(partial), url,
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=EXPECTED_SPLITS, required=True)
    parser.add_argument("--regime", choices=("all",) + EXPECTED_REGIMES,
                        default="all")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--internal-confirmation-aggregate-sha256", required=True)
    parser.add_argument("--external-validation-aggregate-sha256")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    stage, bindings = authorization_bindings(
        split=args.split,
        internal_confirmation_aggregate_sha256=(
            args.internal_confirmation_aggregate_sha256
        ),
        external_validation_aggregate_sha256=(
            args.external_validation_aggregate_sha256
        ),
    )
    authorization = validate_authorization(
        args.authorization, stage=stage, input_bindings=bindings
    )
    manifest, manifest_digest = load_manifest(args.manifest)
    regimes = EXPECTED_REGIMES if args.regime == "all" else (args.regime,)
    completed = []
    for regime in regimes:
        entry = manifest_entry(manifest, split=args.split, regime=regime)
        destination = args.data_root / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (destination.stat().st_size != entry["bytes"]
                    or file_sha256(destination) != entry["sha256"]):
                raise RuntimeError(f"existing publisher file fails manifest: {destination}")
        else:
            partial = destination.with_suffix(destination.suffix + ".part")
            url = (
                f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
                f"{DATASET_REVISION}/{entry['path']}"
            )
            _download(url, partial)
            if partial.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"downloaded publisher file size mismatch: {regime}")
            observed = file_sha256(partial)
            if observed != entry["sha256"]:
                raise RuntimeError(f"downloaded publisher file hash mismatch: {regime}")
            # Same-filesystem hard link gives no-overwrite publication semantics.
            try:
                os.link(partial, destination)
            finally:
                partial.unlink(missing_ok=True)
        completed.append({
            "regime": regime,
            "path": entry["path"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        })
    body = {
        "schema_version": "maintrack.external-well-download-receipt.v1",
        "split": args.split,
        "stage": stage,
        "stage_authorization_sha256": authorization.authorization_digest,
        "external_manifest_sha256": manifest_digest,
        "files": completed,
        "outcomes_decoded": False,
    }
    body["content_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.receipt.parent, prefix=args.receipt.name + ".tmp.",
        delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, args.receipt)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "files": len(completed),
        "split": args.split,
        "content_sha256": body["content_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
