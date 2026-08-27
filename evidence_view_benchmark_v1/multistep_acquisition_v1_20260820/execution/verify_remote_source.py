"""Verify staged bytes against a manifest created at another absolute root."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
manifest_path = root / "multistep_acquisition_v1_20260820/SOURCE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
marker = "/evidence_view_benchmark_v1/"
checked = 0
for row in manifest["files"]:
    original = str(row["path"])
    if marker not in original:
        raise SystemExit(f"manifest path lacks benchmark marker: {original}")
    staged = root / original.split(marker, 1)[1]
    if not staged.is_file():
        raise SystemExit(f"staged source absent: {staged}")
    actual = hashlib.sha256(staged.read_bytes()).hexdigest()
    if actual != row["sha256"]:
        raise SystemExit(f"staged source hash mismatch: {staged}")
    checked += 1
print(json.dumps({"valid": True, "files": checked, "source_closure_sha256": manifest["source_closure_sha256"]}))

