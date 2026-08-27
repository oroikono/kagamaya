"""Run the frozen single-/multi-mode and misspecification experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from sigs_agent.policies import TransformersQwenBackend, resolve_model_identity

from .config import FROZEN_CONFIG
from .engine import discover_operators
from .manufactured import manufactured_reaction_diffusion
from .proposals import (
    evolutionary_proposal,
    qwen_proposal,
    random_proposal,
    static_proposal,
)
from .serialization import content_digest, file_digest
from .susano_candidates import build_candidate_manifest


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _implementation_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py")) + sorted((root / "tests").glob("*.py"))
    files += [root / "DESIGN.md", root / "DEVELOPMENT_LOG.md"]
    hashes = {
        str(path.relative_to(root)): file_digest(path)
        for path in files
        if path.is_file()
    }
    return {
        "file_sha256": hashes,
        "implementation_sha256": content_digest(hashes),
    }


def _proposal(kind: str, candidates, args: argparse.Namespace):
    if kind == "static":
        return static_proposal(candidates)
    if kind == "random":
        return random_proposal(candidates, seed=args.proposal_seed)
    if kind == "evolution":
        # A proposal-only objective that places simpler SUSANO supports earlier.
        complexity = {candidate.candidate_id: candidate.complexity for candidate in candidates}
        return evolutionary_proposal(
            candidates,
            seed=args.proposal_seed,
            budget=args.evolution_budget,
            objective=lambda order: -sum(
                (index + 1) * complexity[candidate_id]
                for index, candidate_id in enumerate(order)
            ),
        )
    if kind == "qwen":
        if not args.qwen_model:
            raise ValueError("--qwen-model is required with --proposal qwen")
        backend = TransformersQwenBackend(
            args.qwen_model,
            max_new_tokens=args.qwen_max_new_tokens,
            local_files_only=args.qwen_local_files_only,
        )
        return qwen_proposal(
            candidates,
            backend=backend,
            model_identity=resolve_model_identity(args.qwen_model),
        )
    raise AssertionError(kind)


def _run_case(excitation: str, candidates, manifest, proposal) -> dict[str, Any]:
    observed = manufactured_reaction_diffusion(
        excitation=excitation,
        nx=FROZEN_CONFIG.nx,
        nt=FROZEN_CONFIG.nt,
    )
    return discover_operators(
        observed,
        candidates,
        proposal,
        manifest_sha256=manifest["manifest_sha256"],
    ).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposal",
        choices=("static", "random", "evolution", "qwen"),
        default="static",
    )
    parser.add_argument("--proposal-seed", type=int, default=42)
    parser.add_argument("--evolution-budget", type=int, default=64)
    parser.add_argument("--qwen-model")
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/agentic_frontier/wp1_wp4_operator_agent/results/demo.json"
        ),
    )
    args = parser.parse_args()

    candidates, manifest = build_candidate_manifest()
    proposal = _proposal(args.proposal, candidates, args)
    multi_mode = _run_case("multi_mode", candidates, manifest, proposal)
    single_mode = _run_case("single_mode", candidates, manifest, proposal)

    reduced, reduced_manifest = build_candidate_manifest(
        candidate_ids=[
            candidate.candidate_id
            for candidate in candidates
            if candidate.candidate_id != "reaction_diffusion"
        ]
    )
    reduced_proposal = _proposal(args.proposal, reduced, args)
    misspecified = _run_case(
        "multi_mode", reduced, reduced_manifest, reduced_proposal
    )

    payload: dict[str, Any] = {
        "schema_version": "verisigs.wp1_wp4.demo.v1",
        "verisigs_git_commit": _git_commit(),
        "implementation": _implementation_manifest(),
        "manifest": manifest,
        "proposal_kind": args.proposal,
        "cases": {
            "multi_mode": multi_mode,
            "single_mode": single_mode,
            "multi_mode_truth_support_removed": misspecified,
        },
    }
    payload["artifact_sha256"] = content_digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": payload["artifact_sha256"],
                "statuses": {
                    key: value["terminal_status"]
                    for key, value in payload["cases"].items()
                },
                "selected": {
                    key: value["selected_candidate_id"]
                    for key, value in payload["cases"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
