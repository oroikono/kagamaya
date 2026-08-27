"""Prospectively frozen external Gray--Scott transfer on The Well.

The external arm changes the data origin, dimensionality, and coupled physics,
while retaining the public 12-action table, score-max attack, four-mode
agreement safeguard, terminal thresholds, and owner-isolated scoring used by
the internal campaign.  Publisher ``valid`` is the authority-admission split;
publisher ``test`` is sealed behind a content-addressed stage authorization.

The three library interventions remove *mechanism groups*.  Columns are scaled
by the coefficients published in the pinned generator, so the complete model
has coefficients of magnitude one.  This is fixed dimensional feature scaling,
not threshold tuning: the local authority thresholds are unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .attack_guard import POLICIES
from .authority import AuthorityRecord, evaluate
from .authorization import validate as validate_authorization
from .campaign import atomic_write_gzip_json, canonical_digest
from .contract import (
    NOISE_CELLS,
    VIEW_BANK,
    ViewSpec,
    adapted_public_objective,
    public_objective,
)
from .environment import OwnerLabel, PublicEpisode, score_decision
from .weak_evidence import WeakEvidence


EXTERNAL_SCHEMA_VERSION = "maintrack.external-well-gray-scott.v1"
EXTERNAL_MANIFEST_SCHEMA = "maintrack.external-well-manifest.v1"
EXTERNAL_REFERENCE_VIEW_ID = "EXTERNAL_REFERENCE_961_M5"
EXTERNAL_ORIGIN = "the_well_gray_scott_chebfun_etdrk4"
DATASET_ID = "polymathic-ai/gray_scott_reaction_diffusion"
DATASET_REVISION = "913ef5679d5c4b4ccd6fcd52c055509dfbf190f9"
GENERATOR_REVISION = "528fde26d54f2c05456fed78e479b88651647090"
EXPECTED_REGIMES = ("bubbles", "gliders", "maze", "spirals", "spots", "worms")
EXPECTED_SPLITS = ("valid", "test")
TRAJECTORIES_PER_FILE = 20
RAW_STEPS = 1001
SPATIAL_RESOLUTION = 128
WINDOW_FIRST_INDEX = 20
WINDOW_LAST_INDEX = 980
WINDOW_LENGTH = WINDOW_LAST_INDEX - WINDOW_FIRST_INDEX + 1
MAX_MODE = 5
DELTA_A = 2.0e-5
DELTA_B = 1.0e-5
TARGETS = ("A", "B")
LIBRARY_CONTRACTS = (
    "full",
    "omit_diffusion",
    "omit_autocatalysis",
    "omit_reservoir",
)

# Exact parameter values in the pinned generator and the pinned file names.
REGIME_PARAMETERS: dict[str, tuple[float, float]] = {
    "bubbles": (0.098, 0.057),
    "gliders": (0.014, 0.054),
    "maze": (0.029, 0.057),
    "spirals": (0.018, 0.051),
    "spots": (0.030, 0.062),
    "worms": (0.058, 0.065),
}


def atoms_for_target(target: str) -> tuple[str, ...]:
    if target == "A":
        return (
            "scaled_diffusion_A",
            "autocatalysis_AB2",
            "scaled_reservoir_A",
        )
    if target == "B":
        return (
            "scaled_diffusion_B",
            "autocatalysis_AB2",
            "scaled_reservoir_B",
        )
    raise ValueError("target must be A or B")


CONTRACT_COLUMNS: dict[str, tuple[int, ...]] = {
    "full": (0, 1, 2),
    "omit_diffusion": (1, 2),
    "omit_autocatalysis": (0, 2),
    "omit_reservoir": (0, 1),
}


def omitted_mechanism(contract_id: str) -> str | None:
    if contract_id == "full":
        return None
    if contract_id not in CONTRACT_COLUMNS:
        raise ValueError("unknown library contract")
    return contract_id.removeprefix("omit_")


def canonical_json_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    validate_manifest(value)
    return value, hashlib.sha256(raw).hexdigest()


def validate_manifest(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != EXTERNAL_MANIFEST_SCHEMA:
        raise ValueError("external manifest schema drift")
    if value.get("dataset") != DATASET_ID:
        raise ValueError("external dataset drift")
    if value.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("external dataset revision drift")
    if value.get("dataset_license") != "CC-BY-4.0":
        raise ValueError("external dataset license drift")
    if value.get("generator_revision") != GENERATOR_REVISION:
        raise ValueError("external generator revision drift")
    if value.get("generator_license") != "MIT":
        raise ValueError("external generator license drift")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 12:
        raise ValueError("external manifest must contain twelve publisher files")
    expected = {
        (split, regime) for split in EXPECTED_SPLITS for regime in EXPECTED_REGIMES
    }
    observed = {(item.get("split"), item.get("regime")) for item in files}
    if observed != expected:
        raise ValueError("external publisher split/regime cross drift")
    for item in files:
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            raise ValueError("invalid external file size")
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid external file SHA-256")
        if str(item.get("path", "")).split("/")[:2] != ["data", item["split"]]:
            raise ValueError("external path does not bind publisher split")


def manifest_entry(manifest: Mapping[str, Any], *, split: str,
                   regime: str) -> dict:
    validate_manifest(manifest)
    matches = [
        dict(item) for item in manifest["files"]
        if item["split"] == split and item["regime"] == regime
    ]
    if len(matches) != 1:
        raise ValueError("manifest entry is not unique")
    return matches[0]


def file_sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_publisher_file(path: Path, entry: Mapping[str, Any]) -> str:
    if path.name != Path(str(entry["path"])).name:
        raise ValueError("publisher file name does not match pinned manifest")
    if path.stat().st_size != int(entry["bytes"]):
        raise ValueError("publisher file size mismatch")
    observed = file_sha256(path)
    if observed != entry["sha256"]:
        raise ValueError("publisher file SHA-256 mismatch")
    return observed


def temporal_positions(temporal_samples: int) -> np.ndarray:
    """Exact nested publisher-frame positions inside indices 20..980."""

    strides = {7: 160, 13: 80, 25: 40, WINDOW_LENGTH: 1}
    if temporal_samples not in strides:
        raise ValueError("external temporal resolution is outside the freeze")
    result = np.arange(0, WINDOW_LENGTH, strides[temporal_samples], dtype=int)
    if len(result) != temporal_samples or result[0] != 0 or result[-1] != 960:
        raise AssertionError("external temporal nesting invariant failed")
    return result


def external_registered_cost(view: ViewSpec, spatial_nodes: int = SPATIAL_RESOLUTION
                             ) -> float:
    return float(
        spatial_nodes**2 * view.temporal_samples
        + 0.1 * view.test_mode_count * spatial_nodes**2
    )


def external_reference_cost(spatial_nodes: int = SPATIAL_RESOLUTION) -> float:
    return float(spatial_nodes**2 * WINDOW_LENGTH + 0.1 * 5 * spatial_nodes**2)


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _as_text_tuple(values: Any) -> tuple[str, ...]:
    return tuple(_as_text(value) for value in values)


def _read_scalar(group: Any, name: str, trajectory_index: int) -> float:
    dataset = group[name]
    sample_varying = bool(dataset.attrs.get("sample_varying", False))
    raw = dataset[trajectory_index] if sample_varying else dataset[()]
    array = np.asarray(raw)
    if array.size != 1:
        raise ValueError(f"scalar {name} is not scalar-valued")
    return float(array.reshape(-1)[0])


@dataclass(frozen=True)
class ExternalTrajectory:
    split: str
    regime: str
    trajectory_index: int
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    B: np.ndarray
    F: float
    k: float
    source_file_sha256: str

    @property
    def source_id(self) -> str:
        payload = (
            f"{DATASET_REVISION}|{self.source_file_sha256}|{self.split}|"
            f"{self.regime}|{self.trajectory_index}|{self.F:.12g}|{self.k:.12g}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate(self, *, require_publisher_shape: bool = False) -> None:
        if self.split not in EXPECTED_SPLITS or self.regime not in EXPECTED_REGIMES:
            raise ValueError("external source identity outside manifest")
        if not (0 <= self.trajectory_index < TRAJECTORIES_PER_FILE):
            raise ValueError("external trajectory index outside publisher file")
        if self.A.shape != self.B.shape or self.A.ndim != 3:
            raise ValueError("external A/B field shape mismatch")
        if self.A.shape[0] != WINDOW_LENGTH:
            raise ValueError("external trajectory must contain indices 20..980")
        if self.A.shape[1:] != (len(self.x), len(self.y)):
            raise ValueError("external coordinate/field shape mismatch")
        if require_publisher_shape and self.A.shape[1:] != (
            SPATIAL_RESOLUTION, SPATIAL_RESOLUTION
        ):
            raise ValueError("external publisher spatial resolution drift")
        if self.times.shape != (WINDOW_LENGTH,):
            raise ValueError("external time shape drift")
        for values, name in ((self.times, "time"), (self.x, "x"), (self.y, "y")):
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid external {name} coordinates")
            if len(values) > 1 and not np.all(np.diff(values) > 0):
                raise ValueError(f"external {name} coordinates are not increasing")
        if not np.allclose(np.diff(self.times), np.diff(self.times)[0], rtol=1e-6,
                           atol=1e-9):
            raise ValueError("external publisher time grid is not uniform")
        for values, name in ((self.x, "x"), (self.y, "y")):
            if not np.allclose(np.diff(values), np.diff(values)[0], rtol=1e-6,
                               atol=1e-9):
                raise ValueError(f"external publisher {name} grid is not uniform")
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.B)):
            raise ValueError("external fields contain nonfinite values")
        expected_F, expected_k = REGIME_PARAMETERS[self.regime]
        if not math.isclose(self.F, expected_F, rel_tol=0.0, abs_tol=5e-7):
            raise ValueError("external F parameter disagrees with pinned generator")
        if not math.isclose(self.k, expected_k, rel_tol=0.0, abs_tol=5e-7):
            raise ValueError("external k parameter disagrees with pinned generator")


def read_open_well_trajectory(handle: Any, *, split: str, regime: str,
                              trajectory_index: int,
                              source_file_sha256: str,
                              require_publisher_shape: bool = True
                              ) -> ExternalTrajectory:
    """Read one trajectory from an already-open HDF5-like object.

    Keeping schema validation separate from ``h5py.File`` allows deterministic
    fake-schema tests without installing or downloading the 30 GiB corpus.
    """

    if split not in EXPECTED_SPLITS or regime not in EXPECTED_REGIMES:
        raise ValueError("unknown external split or regime")
    if int(handle.attrs["n_trajectories"]) != TRAJECTORIES_PER_FILE:
        raise ValueError("publisher trajectory count drift")
    if int(handle.attrs["n_spatial_dims"]) != 2:
        raise ValueError("publisher spatial dimension drift")
    if _as_text(handle.attrs["dataset_name"]) != "gray_scott_reaction_diffusion":
        raise ValueError("publisher dataset name drift")
    dimensions = handle["dimensions"]
    spatial_dims = _as_text_tuple(dimensions.attrs["spatial_dims"])
    if len(spatial_dims) != 2:
        raise ValueError("publisher spatial-dimension metadata drift")
    x = np.asarray(dimensions[spatial_dims[0]][:], dtype=float)
    y = np.asarray(dimensions[spatial_dims[1]][:], dtype=float)
    time_dataset = dimensions["time"]
    if bool(time_dataset.attrs.get("sample_varying", False)):
        all_times = np.asarray(time_dataset[trajectory_index, :], dtype=float)
    else:
        all_times = np.asarray(time_dataset[:], dtype=float)
    if all_times.shape != (RAW_STEPS,):
        raise ValueError("publisher raw time length drift")
    fields = handle["t0_fields"]
    names = set(_as_text_tuple(fields.attrs["field_names"]))
    if names != {"A", "B"}:
        raise ValueError("publisher Gray--Scott field manifest drift")
    field_slice = (trajectory_index, slice(WINDOW_FIRST_INDEX, WINDOW_LAST_INDEX + 1),
                   slice(None), slice(None))
    A = np.asarray(fields["A"][field_slice], dtype=np.float32)
    B = np.asarray(fields["B"][field_slice], dtype=np.float32)
    F = _read_scalar(handle["scalars"], "F", trajectory_index)
    k = _read_scalar(handle["scalars"], "k", trajectory_index)
    trajectory = ExternalTrajectory(
        split=split,
        regime=regime,
        trajectory_index=trajectory_index,
        times=all_times[WINDOW_FIRST_INDEX:WINDOW_LAST_INDEX + 1],
        x=x,
        y=y,
        A=A,
        B=B,
        F=F,
        k=k,
        source_file_sha256=source_file_sha256,
    )
    trajectory.validate(require_publisher_shape=require_publisher_shape)
    return trajectory


def read_well_trajectory(path: Path, *, split: str, regime: str,
                         trajectory_index: int,
                         source_file_sha256: str,
                         sealed_test_authorized: bool = False
                         ) -> ExternalTrajectory:
    if split == "test" and not sealed_test_authorized:
        raise PermissionError("publisher TEST remains sealed")
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on Euler
        raise RuntimeError("h5py is required to decode The Well files") from exc
    with h5py.File(path, "r") as handle:
        return read_open_well_trajectory(
            handle,
            split=split,
            regime=regime,
            trajectory_index=trajectory_index,
            source_file_sha256=source_file_sha256,
        )


def noise_seed(source_id: str, noise_model: str, noise_level: float) -> int:
    payload = (
        f"{source_id}|{noise_model}|{noise_level:.9g}|"
        "external-well-shared-noise-v1"
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def observed_external_arrays(trajectory: ExternalTrajectory, *, noise_model: str,
                             noise_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Create one shared noisy record from which every view is nested."""

    trajectory.validate()
    if (noise_model, noise_level) not in NOISE_CELLS:
        raise ValueError("external noise cell outside prospective contract")
    if noise_level == 0.0:
        return trajectory.A, trajectory.B
    rng = np.random.default_rng(noise_seed(
        trajectory.source_id, noise_model, noise_level
    ))

    def perturb(values: np.ndarray) -> np.ndarray:
        scale = np.float32(noise_level * max(float(np.std(values)), 1e-12))
        try:
            eps = rng.standard_normal(values.shape, dtype=np.float32)
        except TypeError:  # NumPy < 1.20 fallback
            eps = rng.standard_normal(values.shape).astype(np.float32)
        if noise_model == "correlated":
            eps = (
                eps + np.roll(eps, 1, axis=1) + np.roll(eps, -1, axis=1)
                + np.roll(eps, 1, axis=2) + np.roll(eps, -1, axis=2)
            ) / np.float32(math.sqrt(5.0))
            eps = np.float32(0.75) * eps + np.float32(0.25) * np.roll(
                eps, 1, axis=0
            )
        elif noise_model != "iid":
            raise ValueError("noise model must be iid or correlated")
        result = np.array(values, dtype=np.float32, copy=True)
        result += scale * eps
        return result

    return perturb(trajectory.A), perturb(trajectory.B)


@dataclass(frozen=True)
class ProjectionBundle:
    times: np.ndarray
    A: np.ndarray
    B: np.ndarray
    AB2: np.ndarray
    length_x: float
    length_y: float
    source_digest: str
    spatial_nodes: int

    def validate(self) -> None:
        expected = (len(self.times), MAX_MODE + 1, MAX_MODE + 1)
        if self.A.shape != expected or self.B.shape != expected or self.AB2.shape != expected:
            raise ValueError("low-mode projection shape mismatch")
        if not (self.length_x > 0 and self.length_y > 0):
            raise ValueError("invalid periodic domain lengths")
        if not all(np.all(np.isfinite(value)) for value in (self.A, self.B, self.AB2)):
            raise ValueError("nonfinite low-mode projection")


def _periodic_length(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        raise ValueError("periodic coordinate has fewer than two nodes")
    spacing = float(np.diff(coordinates)[0])
    if spacing <= 0 or not np.allclose(np.diff(coordinates), spacing,
                                       rtol=1e-6, atol=1e-9):
        raise ValueError("periodic coordinate is not uniform")
    return spacing * len(coordinates)


def _low_modes(values: np.ndarray, *, cell_area: float,
               chunk_size: int = 32) -> np.ndarray:
    if values.ndim != 3 or values.shape[1] < 2 * MAX_MODE + 1 \
            or values.shape[2] < 2 * MAX_MODE + 1:
        raise ValueError("field is too small for the frozen low-mode bank")
    result = np.empty(
        (values.shape[0], MAX_MODE + 1, MAX_MODE + 1), dtype=np.complex128
    )
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        transformed = np.fft.fft2(values[start:stop], axes=(-2, -1))
        result[start:stop] = transformed[:, :MAX_MODE + 1, :MAX_MODE + 1]
    result *= cell_area
    return result


def build_projection_bundle(trajectory: ExternalTrajectory,
                            observed_A: np.ndarray, observed_B: np.ndarray,
                            *, chunk_size: int = 32) -> ProjectionBundle:
    if observed_A.shape != trajectory.A.shape or observed_B.shape != trajectory.B.shape:
        raise ValueError("observed external field shape mismatch")
    length_x = _periodic_length(trajectory.x)
    length_y = _periodic_length(trajectory.y)
    cell_area = length_x * length_y / (len(trajectory.x) * len(trajectory.y))
    A_modes = _low_modes(observed_A, cell_area=cell_area, chunk_size=chunk_size)
    B_modes = _low_modes(observed_B, cell_area=cell_area, chunk_size=chunk_size)
    AB2_modes = np.empty_like(A_modes)
    for start in range(0, len(observed_A), chunk_size):
        stop = min(start + chunk_size, len(observed_A))
        nonlinear = observed_A[start:stop] * observed_B[start:stop] ** 2
        AB2_modes[start:stop] = _low_modes(
            nonlinear, cell_area=cell_area, chunk_size=chunk_size
        )
    digest = hashlib.sha256()
    digest.update(trajectory.source_id.encode())
    digest.update(np.ascontiguousarray(A_modes).tobytes())
    digest.update(np.ascontiguousarray(B_modes).tobytes())
    digest.update(np.ascontiguousarray(AB2_modes).tobytes())
    bundle = ProjectionBundle(
        times=np.asarray(trajectory.times, dtype=float),
        A=A_modes,
        B=B_modes,
        AB2=AB2_modes,
        length_x=length_x,
        length_y=length_y,
        source_digest=digest.hexdigest(),
        spatial_nodes=len(trajectory.x),
    )
    bundle.validate()
    return bundle


def _spatial_projection(coefficients: np.ndarray, *, kind: str,
                        kx: int, ky: int) -> np.ndarray:
    # numpy's DFT uses exp(-i theta); real data imply the positive-exponent
    # coefficient is its conjugate.  Phases match the internal test bank.
    positive = np.conjugate(coefficients[:, kx, ky])
    phase = 0.173 if kind == "sin" else -0.291
    phased = np.exp(1j * phase) * positive
    if kind == "sin":
        return np.imag(phased)
    if kind == "cos":
        return np.real(phased)
    raise ValueError("spatial test kind must be sin or cos")


def _temporal_test(q: int, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = float(times[0])
    end = float(times[-1])
    if not end > start:
        raise ValueError("external time window has nonpositive duration")
    s = (times - start) / (end - start)
    value = s ** (q + 2) * (1.0 - s) ** 2
    derivative_s = (
        (q + 2) * s ** (q + 1) * (1.0 - s) ** 2
        - 2.0 * s ** (q + 2) * (1.0 - s)
    )
    return value, derivative_s / (end - start)


def _full_design(bundle: ProjectionBundle, *, target: str,
                 temporal_samples: int, test_mode_count: int,
                 F: float, k: float) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    positions = temporal_positions(temporal_samples)
    times = bundle.times[positions]
    field_modes = bundle.A if target == "A" else bundle.B
    if target == "A":
        atom_mode_series = (bundle.A, bundle.AB2, bundle.A)
    elif target == "B":
        atom_mode_series = (bundle.B, bundle.AB2, bundle.B)
    else:
        raise ValueError("target must be A or B")
    rows: list[list[float]] = []
    targets: list[float] = []
    roles: list[str] = []
    for q in range(test_mode_count):
        temporal, temporal_dt = _temporal_test(q, times)
        for kx in range(test_mode_count + 1):
            for ky in range(test_mode_count + 1):
                if kx == 0 and ky == 0:
                    continue
                physical_k2 = (
                    (2.0 * math.pi * kx / bundle.length_x) ** 2
                    + (2.0 * math.pi * ky / bundle.length_y) ** 2
                )
                for kind_index, kind in enumerate(("sin", "cos")):
                    target_series = _spatial_projection(
                        field_modes, kind=kind, kx=kx, ky=ky
                    )[positions]
                    target_row = float(np.trapezoid(-target_series * temporal_dt, times))
                    base_series = [
                        _spatial_projection(series, kind=kind, kx=kx, ky=ky)[positions]
                        for series in atom_mode_series
                    ]
                    if target == "A":
                        mechanism_series = (
                            DELTA_A * (-physical_k2) * base_series[0],
                            base_series[1],
                            -F * base_series[2],
                        )
                    else:
                        mechanism_series = (
                            DELTA_B * (-physical_k2) * base_series[0],
                            base_series[1],
                            (F + k) * base_series[2],
                        )
                    row = [
                        float(np.trapezoid(series * temporal, times))
                        for series in mechanism_series
                    ]
                    rows.append(row)
                    targets.append(target_row)
                    roles.append(
                        "train" if (q + kx + 2 * ky + kind_index) % 2 == 0
                        else "validate"
                    )
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), tuple(roles)


def _episode_id(trajectory: ExternalTrajectory, *, target: str,
                contract_id: str, noise_model: str, noise_level: float) -> str:
    payload = (
        f"{trajectory.source_id}|{target}|{contract_id}|{noise_model}|"
        f"{noise_level:.9g}|external-well-public-v1"
    )
    return "W-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def build_external_evidence(bundle: ProjectionBundle, *, target: str,
                            contract_id: str, episode_id: str,
                            view_id: str, temporal_samples: int,
                            test_mode_count: int, F: float, k: float,
                            registered_cost: float,
                            full_components: tuple[
                                np.ndarray, np.ndarray, tuple[str, ...]
                            ] | None = None) -> WeakEvidence:
    if contract_id not in CONTRACT_COLUMNS:
        raise ValueError("unknown library contract")
    if full_components is None:
        full_design, target_vector, roles = _full_design(
            bundle,
            target=target,
            temporal_samples=temporal_samples,
            test_mode_count=test_mode_count,
            F=F,
            k=k,
        )
    else:
        full_design, target_vector, roles = full_components
        if full_design.shape[1] != 3 or len(target_vector) != len(full_design):
            raise ValueError("invalid cached external full-library components")
    columns = CONTRACT_COLUMNS[contract_id]
    atoms = tuple(atoms_for_target(target)[index] for index in columns)
    design = np.ascontiguousarray(full_design[:, columns])
    digest = hashlib.sha256()
    digest.update(bundle.source_digest.encode())
    digest.update(target.encode())
    digest.update(contract_id.encode())
    digest.update(view_id.encode())
    digest.update(np.ascontiguousarray(design).tobytes())
    digest.update(np.ascontiguousarray(target_vector).tobytes())
    return WeakEvidence(
        episode_id=episode_id,
        view_id=view_id,
        origin=EXTERNAL_ORIGIN,
        atoms=atoms,
        design=design,
        target=target_vector,
        row_roles=roles,
        registered_cost=registered_cost,
        evidence_digest=digest.hexdigest(),
    )


def owner_label(episode_id: str, *, target: str,
                contract_id: str) -> OwnerLabel:
    return OwnerLabel(
        episode_id=episode_id,
        family=f"gray_scott_{target}",
        truth_kind="in_library" if contract_id == "full" else "outside_library",
        truth_support=atoms_for_target(target),
        omitted_mechanism=omitted_mechanism(contract_id),
    )


def _record_payload(record: AuthorityRecord, *, include_objective: bool) -> dict:
    payload = record.public_dict()
    payload["coefficients"] = [list(item) for item in record.coefficients]
    if include_objective:
        payload["public_objective"] = public_objective(record)
        payload["adapted_public_objective"] = adapted_public_objective(record)
    return payload


def _proposal_payload(episode: PublicEpisode, label: OwnerLabel,
                      reference: AuthorityRecord) -> list[dict]:
    proposals = []
    for policy_id, policy in POLICIES.items():
        proposal = policy(episode)
        score = score_decision(proposal.decision, label)
        signature = (proposal.decision.status, proposal.decision.support)
        reference_signature = (reference.status, reference.support)
        decisive = proposal.decision.status in {"IDENTIFIED", "OUTSIDE_LIBRARY"}
        proposals.append({
            "policy_id": policy_id,
            "selected_view_id": proposal.selected_view_id,
            "status": proposal.decision.status,
            "support": list(proposal.decision.support),
            "reason": proposal.decision.reason,
            "witnessed_view_ids": list(proposal.witnessed_view_ids),
            "realized_cost": proposal.decision.realized_cost,
            "selected_public_objective": proposal.public_objective,
            "correct": score.correct,
            "false_promotion": score.false_promotion,
            "inconclusive": score.inconclusive,
            "reference_signature_agreement": signature == reference_signature,
            "goodhart_event": decisive and signature != reference_signature,
        })
    return proposals


def build_external_episodes(trajectory: ExternalTrajectory) -> tuple[list[dict], list[dict]]:
    """Evaluate all 40 episode cells for one publisher trajectory."""

    trajectory.validate(require_publisher_shape=True)
    episodes: list[dict] = []
    labels: list[dict] = []
    for noise_model, noise_level in NOISE_CELLS:
        observed_A, observed_B = observed_external_arrays(
            trajectory, noise_model=noise_model, noise_level=noise_level
        )
        bundle = build_projection_bundle(trajectory, observed_A, observed_B)
        for target in TARGETS:
            full_components_by_view = {
                view.view_id: _full_design(
                    bundle,
                    target=target,
                    temporal_samples=view.temporal_samples,
                    test_mode_count=view.test_mode_count,
                    F=trajectory.F,
                    k=trajectory.k,
                )
                for view in VIEW_BANK
            }
            full_components_by_view[EXTERNAL_REFERENCE_VIEW_ID] = _full_design(
                bundle,
                target=target,
                temporal_samples=WINDOW_LENGTH,
                test_mode_count=5,
                F=trajectory.F,
                k=trajectory.k,
            )
            for contract_id in LIBRARY_CONTRACTS:
                episode_id = _episode_id(
                    trajectory,
                    target=target,
                    contract_id=contract_id,
                    noise_model=noise_model,
                    noise_level=noise_level,
                )
                records: dict[str, AuthorityRecord] = {}
                for view in VIEW_BANK:
                    evidence = build_external_evidence(
                        bundle,
                        target=target,
                        contract_id=contract_id,
                        episode_id=episode_id,
                        view_id=view.view_id,
                        temporal_samples=view.temporal_samples,
                        test_mode_count=view.test_mode_count,
                        F=trajectory.F,
                        k=trajectory.k,
                        registered_cost=external_registered_cost(
                            view, bundle.spatial_nodes
                        ),
                        full_components=full_components_by_view[view.view_id],
                    )
                    records[view.view_id] = evaluate(evidence)
                reference = evaluate(build_external_evidence(
                    bundle,
                    target=target,
                    contract_id=contract_id,
                    episode_id=episode_id,
                    view_id=EXTERNAL_REFERENCE_VIEW_ID,
                    temporal_samples=WINDOW_LENGTH,
                    test_mode_count=5,
                    F=trajectory.F,
                    k=trajectory.k,
                    registered_cost=external_reference_cost(bundle.spatial_nodes),
                    full_components=full_components_by_view[
                        EXTERNAL_REFERENCE_VIEW_ID
                    ],
                ))
                public = PublicEpisode(
                    episode_id=episode_id,
                    origin=EXTERNAL_ORIGIN,
                    noise_model=noise_model,
                    noise_level=noise_level,
                    view_records=records,
                )
                public.validate()
                label = owner_label(
                    episode_id, target=target, contract_id=contract_id
                )
                episodes.append({
                    "episode_id": episode_id,
                    "trajectory_index": trajectory.trajectory_index,
                    "target": target,
                    "library_contract": contract_id,
                    "noise_model": noise_model,
                    "noise_level": noise_level,
                    "views": [
                        _record_payload(records[view.view_id], include_objective=True)
                        for view in VIEW_BANK
                    ],
                    "reference": _record_payload(reference, include_objective=False),
                    "proposals": _proposal_payload(public, label, reference),
                })
                labels.append(asdict(label))
    return episodes, labels


def build_external_shard(*, split: str, regime: str, data_path: Path,
                         manifest_path: Path, source_closure: str,
                         authorization_path: Path,
                         authorization_stage: str,
                         authorization_input_bindings: Mapping[str, str]) -> dict:
    if split not in EXPECTED_SPLITS or regime not in EXPECTED_REGIMES:
        raise ValueError("external split/regime outside prospective contract")
    if len(source_closure) != 64:
        raise ValueError("source closure must be a SHA-256 digest")
    expected_stage = (
        "external_validation" if split == "valid" else "external_confirmation"
    )
    if authorization_stage != expected_stage:
        raise ValueError("authorization stage does not match publisher split")
    # This is deliberately the first filesystem-dependent action. In
    # particular, it precedes manifest loading, publisher path stat/hash, h5py
    # import, and every HDF5 value read.
    authorization = validate_authorization(
        authorization_path,
        stage=authorization_stage,
        input_bindings=authorization_input_bindings,
    )
    manifest, manifest_digest = load_manifest(manifest_path)
    entry = manifest_entry(manifest, split=split, regime=regime)
    verified_file_digest = verify_publisher_file(data_path, entry)
    episodes: list[dict] = []
    labels: list[dict] = []
    for trajectory_index in range(TRAJECTORIES_PER_FILE):
        trajectory = read_well_trajectory(
            data_path,
            split=split,
            regime=regime,
            trajectory_index=trajectory_index,
            source_file_sha256=verified_file_digest,
            sealed_test_authorized=(split == "test"),
        )
        trajectory_episodes, trajectory_labels = build_external_episodes(trajectory)
        episodes.extend(trajectory_episodes)
        labels.extend(trajectory_labels)
    payload = {
        "schema_version": EXTERNAL_SCHEMA_VERSION + ".shard",
        "source_closure_sha256": source_closure,
        "external_manifest_sha256": manifest_digest,
        "config": {
            "split": split,
            "regime": regime,
            "publisher_path": entry["path"],
            "publisher_file_sha256": verified_file_digest,
            "stage_authorization_sha256": authorization.authorization_digest,
            "stage_input_bindings": dict(authorization_input_bindings),
        },
        "metadata": {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "generator_revision": GENERATOR_REVISION,
            "generator_parameters": {
                "delta_A": DELTA_A,
                "delta_B": DELTA_B,
                "F": REGIME_PARAMETERS[regime][0],
                "k": REGIME_PARAMETERS[regime][1],
            },
            "feature_scaling": (
                "published-parameter mechanism groups; expected complete-model "
                "coefficients have magnitude one; authority thresholds unchanged"
            ),
            "targets": list(TARGETS),
            "library_contracts": list(LIBRARY_CONTRACTS),
            "noise_cells": [list(cell) for cell in NOISE_CELLS],
            "view_ids": [view.view_id for view in VIEW_BANK],
            "policy_ids": list(POLICIES),
            "window_indices_inclusive": [WINDOW_FIRST_INDEX, WINDOW_LAST_INDEX],
            "reference": {
                "view_id": EXTERNAL_REFERENCE_VIEW_ID,
                "temporal_samples": WINDOW_LENGTH,
                "test_mode_count": 5,
                "selectable": False,
            },
            "claim_ceiling": manifest["claim_ceiling"],
        },
        "episode_count": len(episodes),
        "view_record_count": len(episodes) * len(VIEW_BANK),
        "episodes": episodes,
        "owner_labels": labels,
    }
    payload["content_sha256"] = canonical_digest(payload)
    return payload


def authorization_bindings(*, split: str,
                           internal_confirmation_aggregate_sha256: str,
                           external_validation_aggregate_sha256: str | None = None
                           ) -> tuple[str, dict[str, str]]:
    if split == "valid":
        return "external_validation", {
            "internal_confirmation_aggregate_sha256": (
                internal_confirmation_aggregate_sha256
            )
        }
    if split == "test":
        if not external_validation_aggregate_sha256:
            raise ValueError("sealed TEST requires the validation aggregate digest")
        return "external_confirmation", {
            "internal_confirmation_aggregate_sha256": (
                internal_confirmation_aggregate_sha256
            ),
            "external_validation_aggregate_sha256": (
                external_validation_aggregate_sha256
            ),
        }
    raise ValueError("external split must be valid or test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=EXPECTED_SPLITS, required=True)
    parser.add_argument("--regime", choices=EXPECTED_REGIMES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-closure", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--internal-confirmation-aggregate-sha256", required=True)
    parser.add_argument("--external-validation-aggregate-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage, bindings = authorization_bindings(
        split=args.split,
        internal_confirmation_aggregate_sha256=(
            args.internal_confirmation_aggregate_sha256
        ),
        external_validation_aggregate_sha256=(
            args.external_validation_aggregate_sha256
        ),
    )
    # Validate before even resolving the publisher path. The builder validates
    # the same receipt again at its own boundary before any HDF5 value read.
    validate_authorization(
        args.authorization, stage=stage, input_bindings=bindings
    )
    pinned_name = next(
        str(item["path"])
        for item in json.loads(args.manifest.read_text())["files"]
        if item["split"] == args.split and item["regime"] == args.regime
    )
    data_path = args.data_root / pinned_name
    payload = build_external_shard(
        split=args.split,
        regime=args.regime,
        data_path=data_path,
        manifest_path=args.manifest,
        source_closure=args.source_closure,
        authorization_path=args.authorization,
        authorization_stage=stage,
        authorization_input_bindings=bindings,
    )
    atomic_write_gzip_json(args.output, payload)
    print(json.dumps({
        "content_sha256": payload["content_sha256"],
        "episodes": payload["episode_count"],
        "views": payload["view_record_count"],
        "regime": args.regime,
        "split": args.split,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
