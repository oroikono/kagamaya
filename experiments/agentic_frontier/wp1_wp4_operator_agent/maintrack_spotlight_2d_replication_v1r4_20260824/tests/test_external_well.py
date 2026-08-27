from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ..aggregate_external import authority_admission
from ..authority import AuthorityRecord
from ..contract import (
    NOISE_CELLS,
    VIEW_BY_CONFIG,
    adapted_public_objective,
    public_objective,
)
from .. import external_well as external_module
from ..external_well import (
    CONTRACT_COLUMNS,
    EXTERNAL_REFERENCE_VIEW_ID,
    EXPECTED_REGIMES,
    ExternalTrajectory,
    ProjectionBundle,
    _full_design,
    _low_modes,
    _record_payload,
    _spatial_projection,
    atoms_for_target,
    authorization_bindings,
    build_external_shard,
    build_external_evidence,
    load_manifest,
    observed_external_arrays,
    read_open_well_trajectory,
    temporal_positions,
)


LANE = Path(__file__).resolve().parents[1]


def _trajectory(n: int = 12) -> ExternalTrajectory:
    times = np.arange(961, dtype=float) * 10.0 + 200.0
    x = np.linspace(-1.0, 1.0, n, endpoint=False)
    y = np.linspace(-1.0, 1.0, n, endpoint=False)
    tt = np.linspace(0.0, 1.0, 961)[:, None, None]
    xx = np.arange(n)[None, :, None]
    yy = np.arange(n)[None, None, :]
    A = (0.7 + 0.05 * np.cos(2 * np.pi * xx / n)
         + 0.01 * tt + 0.0 * yy).astype(np.float32)
    B = (0.2 + 0.04 * np.sin(2 * np.pi * yy / n)
         - 0.01 * tt + 0.0 * xx).astype(np.float32)
    return ExternalTrajectory(
        split="valid",
        regime="bubbles",
        trajectory_index=0,
        times=times,
        x=x,
        y=y,
        A=A,
        B=B,
        F=0.098,
        k=0.057,
        source_file_sha256="a" * 64,
    )


def test_pinned_manifest_is_exact_six_by_two() -> None:
    manifest, digest = load_manifest(LANE / "EXTERNAL_WELL_MANIFEST.json")
    assert len(digest) == 64
    assert {
        (row["split"], row["regime"]) for row in manifest["files"]
    } == {
        (split, regime)
        for split in ("valid", "test")
        for regime in EXPECTED_REGIMES
    }
    worms = next(
        row for row in manifest["files"]
        if row["split"] == "valid" and row["regime"] == "worms"
    )
    assert worms["sha256"] == (
        "f8df51aa04ca360113ad088f2a5038a95b366ffbb049526e889367e6d8a3ddfa"
    )


@pytest.mark.parametrize("count,stride", [(7, 160), (13, 80), (25, 40), (961, 1)])
def test_temporal_views_are_exact_nested_publisher_frames(
    count: int, stride: int
) -> None:
    positions = temporal_positions(count)
    assert len(positions) == count
    assert positions.tolist() == list(range(0, 961, stride))


def test_fft_projection_matches_direct_periodic_quadrature() -> None:
    n = 16
    x = np.arange(n)
    y = np.arange(n)
    x2, y2 = np.meshgrid(x, y, indexing="ij")
    values = (
        0.8 * np.cos(2 * np.pi * (2 * x2 + 3 * y2) / n + 0.31)
        + 0.1 * np.sin(2 * np.pi * (x2 + y2) / n)
    )[None, :, :]
    modes = _low_modes(values, cell_area=4.0 / n**2, chunk_size=1)
    for kind, phase in (("sin", 0.173), ("cos", -0.291)):
        z = 2 * np.pi * (2 * x2 + 3 * y2) / n + phase
        test = np.sin(z) if kind == "sin" else np.cos(z)
        direct = np.sum(values[0] * test) * 4.0 / n**2
        projected = _spatial_projection(modes, kind=kind, kx=2, ky=3)[0]
        assert projected == pytest.approx(direct, abs=1e-12)


def _bundle() -> ProjectionBundle:
    rng = np.random.default_rng(123)
    shape = (961, 6, 6)
    bundle = ProjectionBundle(
        times=np.arange(961, dtype=float) * 10.0,
        A=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        B=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        AB2=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        length_x=2.0,
        length_y=2.0,
        source_digest="b" * 64,
        spatial_nodes=128,
    )
    bundle.validate()
    return bundle


@pytest.mark.parametrize("modes,rows", [(2, 32), (3, 90), (4, 192), (5, 350)])
def test_external_row_counts_match_original_mode_bank(modes: int, rows: int) -> None:
    design, target, roles = _full_design(
        _bundle(), target="A", temporal_samples=7, test_mode_count=modes,
        F=0.098, k=0.057
    )
    assert design.shape == (rows, 3)
    assert target.shape == (rows,)
    assert len(roles) == rows
    assert set(roles) == {"train", "validate"}


@pytest.mark.parametrize("target", ["A", "B"])
def test_each_omission_removes_exactly_one_whole_scaled_mechanism(
    target: str,
) -> None:
    bundle = _bundle()
    full_components = _full_design(
        bundle, target=target, temporal_samples=7, test_mode_count=2,
        F=0.098, k=0.057
    )
    full = build_external_evidence(
        bundle, target=target, contract_id="full", episode_id="episode",
        view_id="A00", temporal_samples=7, test_mode_count=2,
        F=0.098, k=0.057, registered_cost=1.0,
        full_components=full_components,
    )
    assert full.atoms == atoms_for_target(target)
    for contract_id, columns in CONTRACT_COLUMNS.items():
        evidence = build_external_evidence(
            bundle, target=target, contract_id=contract_id,
            episode_id="episode", view_id="A00", temporal_samples=7,
            test_mode_count=2, F=0.098, k=0.057, registered_cost=1.0,
            full_components=full_components,
        )
        assert evidence.atoms == tuple(full.atoms[index] for index in columns)
        np.testing.assert_array_equal(evidence.design, full.design[:, columns])


def test_shared_noise_is_deterministic_and_view_independent() -> None:
    trajectory = _trajectory()
    first = observed_external_arrays(
        trajectory, noise_model="correlated", noise_level=0.005
    )
    second = observed_external_arrays(
        trajectory, noise_model="correlated", noise_level=0.005
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], trajectory.A)
    assert ("correlated", 0.0) not in NOISE_CELLS


class _FakeDataset:
    def __init__(self, values, **attrs):
        self.values = np.asarray(values)
        self.attrs = attrs

    def __getitem__(self, item):
        return self.values[item]


class _FakeGroup(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = {} if attrs is None else attrs


def test_fake_well_schema_reader_binds_fields_coordinates_and_parameters() -> None:
    n = 12
    field_A = np.zeros((1, 1001, n, n), dtype=np.float32)
    field_B = np.ones((1, 1001, n, n), dtype=np.float32)
    handle = _FakeGroup(attrs={
        "n_trajectories": 20,
        "n_spatial_dims": 2,
        "dataset_name": "gray_scott_reaction_diffusion",
    })
    handle["dimensions"] = _FakeGroup({
        "x": _FakeDataset(np.linspace(-1.0, 1.0, n, endpoint=False)),
        "y": _FakeDataset(np.linspace(-1.0, 1.0, n, endpoint=False)),
        "time": _FakeDataset(np.arange(1001, dtype=float) * 10.0,
                             sample_varying=False),
    }, attrs={"spatial_dims": ["x", "y"]})
    handle["t0_fields"] = _FakeGroup({
        "A": _FakeDataset(field_A),
        "B": _FakeDataset(field_B),
    }, attrs={"field_names": ["A", "B"]})
    handle["scalars"] = _FakeGroup({
        "F": _FakeDataset(np.asarray(0.098), sample_varying=False),
        "k": _FakeDataset(np.asarray(0.057), sample_varying=False),
    })
    trajectory = read_open_well_trajectory(
        handle,
        split="valid",
        regime="bubbles",
        trajectory_index=0,
        source_file_sha256="c" * 64,
        require_publisher_shape=False,
    )
    assert trajectory.A.shape == (961, n, n)
    assert trajectory.B.shape == (961, n, n)
    assert trajectory.times[[0, -1]].tolist() == [200.0, 9800.0]
    assert trajectory.F == pytest.approx(0.098)
    assert trajectory.k == pytest.approx(0.057)


def test_test_split_requires_validation_aggregate_binding() -> None:
    digest = "d" * 64
    stage, bindings = authorization_bindings(
        split="valid", internal_confirmation_aggregate_sha256=digest
    )
    assert stage == "external_validation"
    assert bindings == {"internal_confirmation_aggregate_sha256": digest}
    with pytest.raises(ValueError, match="validation aggregate"):
        authorization_bindings(
            split="test", internal_confirmation_aggregate_sha256=digest
        )
    stage, bindings = authorization_bindings(
        split="test",
        internal_confirmation_aggregate_sha256=digest,
        external_validation_aggregate_sha256="e" * 64,
    )
    assert stage == "external_confirmation"
    assert set(bindings) == {
        "internal_confirmation_aggregate_sha256",
        "external_validation_aggregate_sha256",
    }


def _admission_rows() -> list[dict]:
    rows = []
    for regime in EXPECTED_REGIMES:
        for trajectory_index in range(20):
            for target in ("A", "B"):
                rows.append({
                    "regime": regime,
                    "trajectory_index": trajectory_index,
                    "target": target,
                    "library_contract": "full",
                    "noise_model": "iid",
                    "noise_level": 0.0,
                    "reference_correct": True,
                    "reference_false_promotion": False,
                    "proposals": {
                        "fixed_25_m5": {
                            "false_promotion": False,
                            "inconclusive": False,
                        }
                    },
                })
    return rows


def test_authority_admission_is_frozen_and_fail_closed() -> None:
    rows = _admission_rows()
    result = authority_admission(rows)
    assert result["status"] == "ADMITTED_FOR_SEALED_TEST"
    assert result["n"] == 240
    rows[0] = {**rows[0], "reference_correct": False,
               "reference_false_promotion": True}
    result = authority_admission(rows)
    assert result["status"] == "AUTHORITY_NOT_VALID_ON_EXTERNAL_SOURCE"
    assert not result["checks"]["reference_false_promotion"]


def test_reference_id_is_not_in_selectable_bank() -> None:
    assert EXTERNAL_REFERENCE_VIEW_ID not in {
        VIEW_BY_CONFIG[(nt, modes)].view_id
        for nt in (7, 13, 25)
        for modes in (2, 3, 4, 5)
    }


def test_each_public_row_serializes_both_frozen_score_variants() -> None:
    view = VIEW_BY_CONFIG[(7, 2)]
    record = AuthorityRecord(
        episode_id="episode",
        view_id=view.view_id,
        origin="external",
        status="OUTSIDE_LIBRARY",
        support=(),
        coefficients=(),
        residual=0.25,
        score_gap=0.01,
        condition=5.0,
        minimum_coefficient=1.0,
        ambiguity_size=1,
        registered_cost=1.0,
        evidence_digest="f" * 64,
    )
    payload = _record_payload(record, include_objective=True)
    assert payload["public_objective"] == public_objective(record)
    assert payload["adapted_public_objective"] == adapted_public_objective(record)


def test_shard_builder_validates_authorization_before_source_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []

    def reject(*args, **kwargs):
        events.append("authorization")
        raise RuntimeError("stop before source access")

    def manifest_access(*args, **kwargs):
        events.append("manifest")
        raise AssertionError("source metadata accessed before authorization")

    monkeypatch.setattr(external_module, "validate_authorization", reject)
    monkeypatch.setattr(external_module, "load_manifest", manifest_access)
    with pytest.raises(RuntimeError, match="stop before source access"):
        build_external_shard(
            split="test",
            regime="bubbles",
            data_path=tmp_path / "sealed.hdf5",
            manifest_path=tmp_path / "manifest.json",
            source_closure="a" * 64,
            authorization_path=tmp_path / "authorization.json",
            authorization_stage="external_confirmation",
            authorization_input_bindings={
                "internal_confirmation_aggregate_sha256": "b" * 64,
                "external_validation_aggregate_sha256": "c" * 64,
            },
        )
    assert events == ["authorization"]
