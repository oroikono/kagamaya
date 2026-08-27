"""Run the frozen owner join after binding it to the stronger R2 raw validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..src import score as frozen_score
from ..src.contract import canonical_json, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--raw-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.raw_validation.read_text())
    if report.get("valid") is not True or report.get("schedule_keys") != 3240:
        raise ValueError("R2 raw validation did not pass")
    if report.get("normalized_sha256") != file_sha256(args.normalized):
        raise ValueError("normalized file is not bound to R2 validation")

    def validated(path: Path, stage: str) -> dict:
        if path != args.normalized or stage != "full":
            raise ValueError("unexpected scorer validation target")
        return {
            "content_sha256": report["content_sha256"],
            "valid": True,
            "rows": 3240,
        }

    frozen_score.validate = validated
    value = frozen_score.owner_join(args.normalized)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
    print(json.dumps({"content_sha256": value["content_sha256"], "rows": len(value["rows"])}, indent=2))


if __name__ == "__main__":
    main()
