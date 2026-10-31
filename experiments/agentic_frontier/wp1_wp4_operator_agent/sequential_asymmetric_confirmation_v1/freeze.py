"""Create the one-time pre-outcome freeze receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..adversarial_verifier_gaming_v1.io_utils import write_json
from ..sequential_consensus_controller_v1.contract import digest
from .contract import FREEZE_PATH, expected_freeze_body, validate_calibration_eligibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FREEZE_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    validate_calibration_eligibility()
    body = expected_freeze_body()
    payload = {**body, "freeze_sha256": digest(body)}
    write_json(args.output, payload)
    print(payload["freeze_sha256"])


if __name__ == "__main__":
    main()
