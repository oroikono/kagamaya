from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .campaign import _auc, _nested, _ols_slope
from .contract import (
    A10_SEEDS, A11_SEEDS, A12_SEEDS, HIGH_NOISES, NODES,
    REFERENCE_NODES, STANDARD_NOISES, RunContract, digest,
    publish_exclusive,
)
from ..maintrack_suite_v41.simulator import Trajectory


def test_reserved_seeds_are_disjoint_and_exact() -> None:
    assert not (set(A10_SEEDS) & set(A11_SEEDS))
    assert not (set(A10_SEEDS) & set(A12_SEEDS))
    assert not (set(A11_SEEDS) & set(A12_SEEDS))
    assert A10_SEEDS == tuple(range(150000, 150010))
    assert A11_SEEDS == tuple(range(151000, 151010))
    assert A12_SEEDS == tuple(range(152000, 152010))


def test_run_contract_fails_closed_on_seed_mutation() -> None:
    RunContract("a10", A10_SEEDS).validate()
    with pytest.raises(ValueError):
        RunContract("a10", A10_SEEDS[:-1]).validate()
    with pytest.raises(ValueError):
        RunContract("a10", A11_SEEDS).validate()


def test_grids_and_noise_levels_are_exact() -> None:
    assert NODES == (7, 13, 25, 49)
    assert REFERENCE_NODES == (49, 97)
    assert STANDARD_NOISES == (0.0, 0.001, 0.005)
    assert HIGH_NOISES == (0.01, 0.05, 0.10)


def test_auc_direction_and_ties() -> None:
    assert _auc([False, False, True, True], [0.0, 1.0, 2.0, 3.0]) == 1.0
    assert _auc([False, False, True, True], [3.0, 2.0, 1.0, 0.0]) == 0.0
    assert _auc([False, True], [1.0, 1.0]) == 0.5


def test_slope_sign() -> None:
    assert _ols_slope([0.0, 1.0, 2.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_nested_49_from_97_is_exact() -> None:
    t = np.linspace(0.0, 1.0, 97)
    u = np.arange(97 * 3, dtype=float).reshape(97, 3)
    fine = Trajectory("x", "a", np.arange(3.0), t, u, "iid", 0.0)
    nested = _nested(fine, 49)
    assert np.array_equal(nested.t, t[::2])
    assert np.array_equal(nested.u, u[::2])
    with pytest.raises(ValueError):
        _nested(fine, 48)


def test_publish_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    publish_exclusive(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}
    with pytest.raises(FileExistsError):
        publish_exclusive(path, {"x": 2})


def test_digest_detects_mutation() -> None:
    assert digest({"threshold": 0.20}) != digest({"threshold": 0.21})

