"""Validate bound demo artifacts without trusting their summary fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .serialization import content_digest, file_digest


EXPECTED = {
    "multi_mode": {
        "status": "VERIFIED_WITHIN_FRAGMENT",
        "identified": "reaction_diffusion",
    },
    "single_mode": {"status": "UNIDENTIFIED", "identified": None},
    "multi_mode_truth_support_removed": {"status": "REJECTED", "identified": None},
}


def _without(payload: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(key)
    return result


def validate_demo(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if content_digest(_without(payload, "artifact_sha256")) != payload["artifact_sha256"]:
        raise ValueError(f"{path}: invalid artifact_sha256")
    manifest = payload["manifest"]
    if content_digest(_without(manifest, "manifest_sha256")) != manifest["manifest_sha256"]:
        raise ValueError(f"{path}: invalid SUSANO manifest digest")

    package_root = Path(__file__).resolve().parent
    implementation = payload["implementation"]
    for relative, expected_sha in implementation["file_sha256"].items():
        if file_digest(package_root / relative) != expected_sha:
            raise ValueError(f"{path}: stale implementation binding for {relative}")
    if (
        content_digest(implementation["file_sha256"])
        != implementation["implementation_sha256"]
    ):
        raise ValueError(f"{path}: invalid implementation digest")

    scientific: dict[str, str] = {}
    for case_name, expected in EXPECTED.items():
        case = payload["cases"][case_name]
        scientific_body = _without(case, "scientific_sha256")
        scientific_body.pop("proposal_audit")
        if content_digest(scientific_body) != case["scientific_sha256"]:
            raise ValueError(f"{path}: invalid scientific digest for {case_name}")
        if case["terminal_status"] != expected["status"]:
            raise ValueError(f"{path}: unexpected terminal status for {case_name}")
        if case["scientifically_identified_candidate_id"] != expected["identified"]:
            raise ValueError(f"{path}: unexpected scientific identification for {case_name}")
        if case["proposal_audit"]["authority_boundary"] != (
            "proposal order excluded from scientific digest and terminal decision"
        ):
            raise ValueError(f"{path}: proposal authority boundary changed")
        scientific[case_name] = case["scientific_sha256"]
    return {
        "artifact_sha256": payload["artifact_sha256"],
        "proposal_kind": payload["proposal_kind"],
        "scientific_sha256": scientific,
    }


def validate_demo_pair(static_path: Path, evolution_path: Path) -> dict[str, Any]:
    static = validate_demo(static_path)
    evolution = validate_demo(evolution_path)
    if static["scientific_sha256"] != evolution["scientific_sha256"]:
        raise ValueError("proposal mechanisms changed the scientific records")
    return {
        "schema_version": "verisigs.wp1_wp4.artifact_admission.v1",
        "valid": True,
        "static": static,
        "evolution": evolution,
        "proposal_order_invariant_scientific_sha256": static["scientific_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static",
        type=Path,
        default=Path(
            "experiments/agentic_frontier/wp1_wp4_operator_agent/results/demo_static.json"
        ),
    )
    parser.add_argument(
        "--evolution",
        type=Path,
        default=Path(
            "experiments/agentic_frontier/wp1_wp4_operator_agent/results/demo_evolution.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_demo_pair(args.static, args.evolution)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
