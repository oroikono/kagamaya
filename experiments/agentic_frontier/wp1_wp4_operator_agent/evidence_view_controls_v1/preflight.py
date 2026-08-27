"""Unreserved one-family smoke; never computes a scientific endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .campaign import _reference97_episodes, _standard_episodes
from .contract import HIGH_NOISES, NODES, PRODUCERS, STANDARD_NOISES, digest, publish_exclusive
from ..adversarial_verifier_gaming_v1.campaign import _cell
from ..maintrack_suite_v41.simulator import Family


SMOKE_SEED = 149999


def run() -> dict:
    family = Family.ADVECTION_DIFFUSION
    standard_counts = {}
    reference_counts = {}
    truth_mutation_invariant = {}
    for producer in PRODUCERS:
        standard = list(_standard_episodes(producer, family, SMOKE_SEED, HIGH_NOISES))
        if len(standard) != 6:
            raise AssertionError("standard smoke cross mismatch")
        for _model, _noise, baseline, channels, truth in standard:
            if tuple(sorted(channels)) != NODES or baseline.t.size % 2 == 0:
                raise AssertionError("standard temporal grid mismatch")
            if truth.support is None:
                raise AssertionError("smoke family unexpectedly outside dictionary")
            if any(channel.t.size != nodes for nodes, channel in channels.items()):
                raise AssertionError("standard node cardinality mismatch")
        _model, _noise, baseline, channels, truth = standard[0]
        with_truth = _cell(producer, baseline, channels[49], channels[49], truth.support, 5)
        without_truth = _cell(producer, baseline, channels[49], channels[49], None, 5)
        if with_truth["final"] != without_truth["final"]:
            raise AssertionError("public diagnostic mutated when held-out truth changed")
        truth_mutation_invariant[producer] = True
        standard_counts[producer] = len(standard)

        reference = list(_reference97_episodes(producer, family, SMOKE_SEED))
        if len(reference) != 2 * len(STANDARD_NOISES):
            raise AssertionError("reference smoke cross mismatch")
        for _model, _noise, _baseline, fine, _truth in reference:
            if fine.t.size != 97 or fine.u.shape[0] != 97:
                raise AssertionError("reference 97-node cardinality mismatch")
            if not np.all(np.isfinite(fine.u)):
                raise AssertionError("reference contains nonfinite observations")
        reference_counts[producer] = len(reference)
    body = {
        "schema_version": "verisigs.evidence_view_controls.preflight.v1",
        "status": "PASS_UNRESERVED_STRUCTURAL_SMOKE_NO_SCIENTIFIC_ENDPOINT",
        "smoke_seed": SMOKE_SEED,
        "family": family.value,
        "standard_episode_counts": standard_counts,
        "reference97_episode_counts": reference_counts,
        "diagnostic_truth_mutation_invariant": truth_mutation_invariant,
    }
    return body | {"content_sha256": digest(body)}


def main() -> None:
    output = Path(__file__).resolve().parent / "preflight/structural_smoke_r2.json"
    payload = run()
    publish_exclusive(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
