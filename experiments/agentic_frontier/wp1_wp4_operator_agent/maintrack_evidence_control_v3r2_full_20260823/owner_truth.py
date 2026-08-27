"""Owner-only analytic truth for in-library and genuinely omitted mechanisms.

The public policy/environment modules must not import this module.  The
campaign owner joins its labels only after public action traces are locked.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .vendor.exact_2d_oracle import (
    FAMILIES,
    analytic_fields,
)
from .vendor.exact_2d_coupled_oracle import (
    COUPLED_FAMILIES,
    analytic_coupled_fields,
)

from .contract import COUPLED_ATOMS, SCALAR_ATOMS


@dataclass(frozen=True)
class OwnerCase:
    case_id: str
    family: str
    target: str
    library_kind: str
    truth_kind: str
    coefficients: tuple[tuple[str, float], ...]
    omitted_mechanism: str | None
    matched_control: str | None

    @property
    def coefficient_map(self) -> dict[str, float]:
        return dict(self.coefficients)

    @property
    def declared_atoms(self) -> tuple[str, ...]:
        return SCALAR_ATOMS if self.library_kind == "scalar" else COUPLED_ATOMS

    @property
    def truth_support(self) -> tuple[str, ...]:
        if self.truth_kind == "outside_library":
            return ()
        return tuple(atom for atom in self.declared_atoms if atom in self.coefficient_map)


def _pairs(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(k), float(v)) for k, v in values.items()))


def owner_cases() -> tuple[OwnerCase, ...]:
    cases: list[OwnerCase] = []
    for family in FAMILIES:
        cases.append(OwnerCase(
            case_id=f"S-{family.name}", family=family.name, target="u",
            library_kind="scalar", truth_kind="in_library",
            coefficients=_pairs(family.coefficients), omitted_mechanism=None,
            matched_control=None,
        ))
    for family in COUPLED_FAMILIES:
        for target, coefficients in (("u", family.u_equation), ("v", family.v_equation)):
            cases.append(OwnerCase(
                case_id=f"C-{family.name}-{target}", family=family.name,
                target=target, library_kind="coupled", truth_kind="in_library",
                coefficients=_pairs(coefficients), omitted_mechanism=None,
                matched_control=None,
            ))

    # Four mechanisms absent from the declared finite dictionaries.  Each is
    # paired with a nearby included family but is not labelled by that control.
    cases.extend((
        OwnerCase(
            case_id="O-variable-divergence-diffusion", family="variable_diffusion",
            target="u", library_kind="scalar", truth_kind="outside_library",
            coefficients=(), omitted_mechanism="divergence_variable_coefficient",
            matched_control="S-anisotropic_adr",
        ),
        OwnerCase(
            case_id="O-fractional-diffusion", family="fractional_diffusion_1p5",
            target="u", library_kind="scalar", truth_kind="outside_library",
            coefficients=(), omitted_mechanism="fractional_laplacian_alpha_1p5",
            matched_control="S-fourth_order",
        ),
        OwnerCase(
            case_id="O-hidden-field", family="hidden_field_coupling",
            target="u", library_kind="scalar", truth_kind="outside_library",
            coefficients=(), omitted_mechanism="unobserved_v_coupling",
            matched_control="C-coupled_reaction_diffusion-u",
        ),
        OwnerCase(
            case_id="O-nonlocal-convolution", family="nonlocal_convolution",
            target="u", library_kind="scalar", truth_kind="outside_library",
            coefficients=(), omitted_mechanism="gaussian_periodic_convolution",
            matched_control="S-allen_cahn",
        ),
    ))
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("owner case IDs must be unique")
    return tuple(cases)


OWNER_CASES = owner_cases()
OWNER_CASE_BY_ID = {case.case_id: case for case in OWNER_CASES}


def opaque_episode_id(case_id: str, seed: int, origin: str) -> str:
    payload = f"{case_id}|{seed}|{origin}|owner-v2".encode()
    return "E-" + hashlib.sha256(payload).hexdigest()[:20]


def seed_transform(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    return (float(rng.uniform(0.0, 2.0 * np.pi)),
            float(rng.uniform(0.0, 2.0 * np.pi)),
            float(rng.uniform(0.88, 1.12)))


def analytic_state(seed: int, x: np.ndarray, y: np.ndarray,
                   t: np.ndarray) -> dict[str, np.ndarray]:
    dx, dy, scale = seed_transform(seed)
    scalar = analytic_fields((x + dx) % (2.0 * np.pi),
                             (y + dy) % (2.0 * np.pi), t)
    coupled = analytic_coupled_fields((x + dx) % (2.0 * np.pi),
                                      (y + dy) % (2.0 * np.pi), t)
    # Amplitude transport is explicit. First-order and linear higher-order
    # derivatives scale once; polynomial atoms are recomputed.
    scalar = {key: scale * value for key, value in scalar.items()
              if key not in {"u2", "u3", "uux", "uuy"}}
    scalar["u2"] = scalar["u"] ** 2
    scalar["u3"] = scalar["u"] ** 3
    scalar["uux"] = scalar["u"] * scalar["ux"]
    scalar["uuy"] = scalar["u"] * scalar["uy"]

    u = scale * coupled["u"]
    v = scale * coupled["v"]
    out = dict(scalar)
    out.update({
        "v": v,
        "vt": scale * coupled["vt"],
        "vx": scale * coupled["vx"],
        "vy": scale * coupled["vy"],
        "vxx": scale * coupled["vxx"],
        "vyy": scale * coupled["vyy"],
        "vxy": scale * coupled["vxy"],
        "uv": u * v,
        "v2": v**2,
    })
    return out


def _fractional_laplacian_exact(seed: int, x: np.ndarray, y: np.ndarray,
                                t: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    # This owner implementation deliberately uses an independent closed-form
    # mode sum, not either numerical producer's derivative routine.
    from .vendor.exact_2d_oracle import (
        MODES,
        _poly_and_derivative,
    )
    dx, dy, scale = seed_transform(seed)
    out = np.zeros(np.broadcast_shapes(x.shape, y.shape, t.shape), dtype=float)
    for mode in MODES:
        p, _ = _poly_and_derivative(mode.time_coefficients, t)
        z = mode.kx * ((x + dx) % (2.0 * np.pi)) + mode.ky * ((y + dy) % (2.0 * np.pi)) + mode.phase
        base = np.sin(z) if mode.trig == "sin" else np.cos(z)
        multiplier = -float(mode.kx**2 + mode.ky**2) ** (alpha / 2.0)
        out += scale * mode.amplitude * p * multiplier * base
    return out


def _gaussian_convolution_exact(seed: int, x: np.ndarray, y: np.ndarray,
                                t: np.ndarray, sigma2: float = 0.12) -> np.ndarray:
    from .vendor.exact_2d_oracle import (
        MODES,
        _poly_and_derivative,
    )
    dx, dy, scale = seed_transform(seed)
    out = np.zeros(np.broadcast_shapes(x.shape, y.shape, t.shape), dtype=float)
    for mode in MODES:
        p, _ = _poly_and_derivative(mode.time_coefficients, t)
        z = mode.kx * ((x + dx) % (2.0 * np.pi)) + mode.ky * ((y + dy) % (2.0 * np.pi)) + mode.phase
        base = np.sin(z) if mode.trig == "sin" else np.cos(z)
        multiplier = math.exp(-0.5 * sigma2 * (mode.kx**2 + mode.ky**2)) - 1.0
        out += scale * mode.amplitude * p * multiplier * base
    return out


def owner_rhs(case: OwnerCase, seed: int, fields: dict[str, np.ndarray],
              x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if case.truth_kind == "in_library":
        rhs = np.zeros_like(fields[case.target])
        for atom, coefficient in case.coefficients:
            rhs += coefficient * fields[atom]
        return rhs

    if case.omitted_mechanism == "divergence_variable_coefficient":
        a = 1.0 + 0.25 * np.sin(x) + 0.15 * np.cos(y)
        ax = 0.25 * np.cos(x)
        ay = -0.15 * np.sin(y)
        return 0.055 * (a * (fields["uxx"] + fields["uyy"])
                        + ax * fields["ux"] + ay * fields["uy"])
    if case.omitted_mechanism == "fractional_laplacian_alpha_1p5":
        return 0.048 * _fractional_laplacian_exact(seed, x, y, t)
    if case.omitted_mechanism == "unobserved_v_coupling":
        return -0.22 * fields["u"] + 0.37 * fields["v"] + 0.04 * (fields["uxx"] + fields["uyy"])
    if case.omitted_mechanism == "gaussian_periodic_convolution":
        return 0.31 * _gaussian_convolution_exact(seed, x, y, t) - 0.12 * fields["u"]
    raise ValueError(f"unknown owner mechanism: {case.omitted_mechanism}")


def owner_forcing(case: OwnerCase, seed: int, fields: dict[str, np.ndarray],
                  x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return fields[f"{case.target}t"] - owner_rhs(case, seed, fields, x, y, t)
