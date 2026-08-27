"""Regenerate the published data from the bundled frozen source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "generation/source_snapshot"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(skip_numerical_bank: bool) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="evidenceviewbench_regen_") as directory:
        work = Path(directory) / "source_snapshot"
        shutil.copytree(SNAPSHOT, work)
        lane = work / "evidence_view_benchmark_v1"
        generated = lane / "cases"
        for relative in (
            "cases.jsonl",
            "held_out_sidecar.jsonl",
            "strata_counts.json",
            "source_bindings.json",
            "dataset_manifest.json",
            "acquisition_49_rows.jsonl",
            "acquisition_bank_manifest.json",
        ):
            path = generated / relative
            if path.exists():
                path.unlink()
        prompt_manifest = lane / "prompts/manifest.json"
        if prompt_manifest.exists():
            prompt_manifest.unlink()

        pythonpath = os.pathsep.join((str(work), str(lane / "src")))
        env = {**os.environ, "PYTHONPATH": pythonpath}
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from evidence_view_benchmark.case_sampler import build; build()",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        comparisons = {
            "dev_cases": (generated / "cases.jsonl", ROOT / "data/dev_cases.jsonl"),
            "dev_labels": (generated / "held_out_sidecar.jsonl", ROOT / "data/dev_labels.jsonl"),
            "dataset_manifest": (
                generated / "dataset_manifest.json",
                ROOT / "data/upstream_dataset_manifest.json",
            ),
        }
        if not skip_numerical_bank:
            subprocess.run(
                [sys.executable, str(lane / "tools/build_gate3_bank.py")],
                cwd=ROOT,
                env=env,
                check=True,
            )
            comparisons.update(
                {
                    "view_bank": (
                        generated / "acquisition_49_rows.jsonl",
                        ROOT / "data/view_bank.jsonl",
                    ),
                    "view_bank_manifest": (
                        generated / "acquisition_bank_manifest.json",
                        ROOT / "data/upstream_view_bank_manifest.json",
                    ),
                }
            )
        rows = {}
        for name, (actual, expected) in comparisons.items():
            actual_hash = sha256(actual)
            expected_hash = sha256(expected)
            if actual_hash != expected_hash:
                raise ValueError(f"regenerated {name} differs from the published file")
            rows[name] = {"sha256": actual_hash, "byte_identical": True}
        return {
            "schema_version": "evidenceviewbench.regeneration_validation.v1",
            "valid": True,
            "numerical_bank_regenerated": not skip_numerical_bank,
            "artifacts": rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-numerical-bank",
        action="store_true",
        help="regenerate only cases and labels; the full bank replay takes about two minutes",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.skip_numerical_bank), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
