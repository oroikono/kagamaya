from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from .campaign import CampaignConfig, paired_contrasts, run, stratified, summarize
from .simulator import CELLS, matched_episode
from .validate import validate_payload
from ..maintrack_suite_v41.simulator import Family


def tiny():
    return run(
        CampaignConfig(
            (4199,),
            families=("advection_diffusion",),
            noise_models=("iid",),
            relative_noises=(0.001,),
            split="test_development",
        )
    )


def rehash(payload):
    raw = dict(payload)
    raw.pop("content_sha256", None)
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_nested_observations_are_exact_at_shared_nodes():
    _, channels, _ = matched_episode(
        Family.ADVECTION_DIFFUSION, 4198, 8675309, 0.001, "correlated"
    )
    fine = channels[49]
    for cell in CELLS:
        assert np.array_equal(channels[cell.nodes].u, fine.u[:: cell.subsample])
        assert np.array_equal(channels[cell.nodes].t, fine.t[:: cell.subsample])


def test_factorial_payload_validates():
    validate_payload(tiny())


def test_confirmation_seed_is_blocked():
    with pytest.raises(ValueError, match="reserved"):
        run(
            CampaignConfig(
                (9500,),
                families=("advection_diffusion",),
                noise_models=("iid",),
                relative_noises=(0.0,),
            )
        )


def test_mutated_cost_is_rejected():
    payload = copy.deepcopy(tiny())
    payload["rows"][0]["action_cost"] = 0.5
    rehash(payload)
    with pytest.raises(AssertionError, match="cost"):
        validate_payload(payload)


def test_missing_cell_is_rejected():
    payload = run(
        CampaignConfig(
            (4197, 4199),
            families=("advection_diffusion",),
            noise_models=("iid",),
            relative_noises=(0.001,),
            split="test_development",
        )
    )
    index = next(
        i
        for i, row in enumerate(payload["rows"])
        if row["nodes"] == 13 and row["temporal_modes"] == 2
    )
    payload["rows"].pop(index)
    payload["summary"] = summarize(payload["rows"])
    payload["paired_contrasts"] = paired_contrasts(payload["rows"])
    payload["stratified"] = stratified(payload["rows"])
    rehash(payload)
    with pytest.raises(AssertionError, match="incomplete"):
        validate_payload(payload)


def test_mutated_hash_is_rejected():
    payload = tiny()
    payload["rows"][0]["correct"] = not payload["rows"][0]["correct"]
    with pytest.raises(AssertionError, match="hash"):
        validate_payload(payload)


def test_rehashed_false_label_is_rejected():
    payload = copy.deepcopy(tiny())
    payload["rows"][0]["false_promotion"] = not payload["rows"][0]["false_promotion"]
    payload["summary"] = summarize(payload["rows"])
    payload["paired_contrasts"] = paired_contrasts(payload["rows"])
    payload["stratified"] = stratified(payload["rows"])
    rehash(payload)
    with pytest.raises(AssertionError, match="false-promotion label"):
        validate_payload(payload)


def test_rehashed_duplicate_factorial_key_is_rejected():
    payload = copy.deepcopy(tiny())
    payload["rows"][1] = copy.deepcopy(payload["rows"][0])
    rehash(payload)
    with pytest.raises(AssertionError, match="duplicate factorial"):
        validate_payload(payload)
