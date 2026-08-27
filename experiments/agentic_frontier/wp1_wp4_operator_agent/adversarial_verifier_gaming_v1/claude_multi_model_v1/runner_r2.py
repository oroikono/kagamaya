"""R2 registry for matched temperature-zero Sonnet/Opus execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import runner as base


MODEL_SPECS_R2: dict[str, dict[str, float | int | str]] = {
    "claude-sonnet-4-6": {
        "policy": "claude_sonnet_4_6",
        "input_usd_per_million": 3.0,
        "output_usd_per_million": 15.0,
        "expected_total_input_tokens": 100_333,
        "maximum_authorized_cost_usd": 0.35,
    },
    "claude-opus-4-6": {
        "policy": "claude_opus_4_6",
        "input_usd_per_million": 5.0,
        "output_usd_per_million": 25.0,
        "expected_total_input_tokens": 100_333,
        "maximum_authorized_cost_usd": 0.57,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--credential", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS_R2))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authorization-reference", required=True)
    args = parser.parse_args()
    original = base.MODEL_SPECS
    base.MODEL_SPECS = MODEL_SPECS_R2
    try:
        base.run(
            campaign_path=args.campaign,
            credential_path=args.credential,
            model=args.model,
            output_root=args.output_root,
            authorization_reference=args.authorization_reference,
        )
    finally:
        base.MODEL_SPECS = original


if __name__ == "__main__":
    main()

