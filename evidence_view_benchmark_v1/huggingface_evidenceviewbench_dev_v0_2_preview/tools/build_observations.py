"""Build the metadata-blind Hugging Face selection table deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidenceviewbench.data import canonical_bytes, digest, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/dev_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "data/dev_observations.jsonl"


def records() -> list[dict]:
    output = []
    for case in read_jsonl(SOURCE):
        row = {
            "schema_version": "evidence_view_benchmark.selection_case.v1",
            "case_id": case["case_id"],
            "parent_public_case_sha256": case["case_sha256"],
            "public_rows": case["public_rows"],
        }
        row["selection_case_sha256"] = digest(row)
        output.append(row)
    if len(output) != 216 or len({row["case_id"] for row in output}) != 216:
        raise ValueError("selection table cardinality mismatch")
    return output


def serialized() -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in records())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = serialized()
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != expected:
            raise SystemExit("observation table does not match deterministic replay")
        print(json.dumps({"valid": True, "rows": 216, "path": str(args.output)}))
        return
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(expected.decode("utf-8"))
    print(json.dumps({"written": True, "rows": 216, "path": str(args.output)}))


if __name__ == "__main__":
    main()
