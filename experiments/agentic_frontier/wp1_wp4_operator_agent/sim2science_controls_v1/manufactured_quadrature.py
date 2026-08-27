"""Manufactured control for temporal weak-form Simpson underintegration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad, simpson


NODES = (7, 13, 25, 49, 97)
MODES = (1, 2, 3, 4, 5)


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def design_sha() -> str:
    return hashlib.sha256(
        Path(__file__).with_name("DESIGN_FREEZE_PREEXECUTION.md").read_bytes()
    ).hexdigest()


def u(s: np.ndarray | float) -> np.ndarray | float:
    return np.exp(0.7 * np.asarray(s)) + 0.2 * np.sin(3.0 * np.pi * np.asarray(s))


def du(s: np.ndarray | float) -> np.ndarray | float:
    return 0.7 * np.exp(0.7 * np.asarray(s)) + 0.6 * np.pi * np.cos(3.0 * np.pi * np.asarray(s))


def _exact(mode: int) -> tuple[float, float]:
    omega = mode * np.pi
    left = -quad(
        lambda value: float(u(value) * omega * np.cos(omega * value)),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=200,
    )[0]
    right = quad(
        lambda value: float(du(value) * np.sin(omega * value)),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=200,
    )[0]
    return float(left), float(right)


def _relative_error(estimate: float, reference: float) -> float:
    return abs(estimate - reference) / max(abs(reference), 1e-15)


def run() -> dict:
    rows: list[dict] = []
    for mode in MODES:
        exact_left, exact_right = _exact(mode)
        for nodes in NODES:
            grid = np.linspace(0.0, 1.0, nodes)
            omega = mode * np.pi
            eta = np.sin(omega * grid)
            deta = omega * np.cos(omega * grid)
            left = -float(simpson(u(grid) * deta, x=grid))
            right = float(simpson(du(grid) * eta, x=grid))
            rows.append(
                {
                    "mode": mode,
                    "nodes": nodes,
                    "exact_left": exact_left,
                    "exact_right": exact_right,
                    "simpson_left": left,
                    "simpson_right": right,
                    "left_relative_error": _relative_error(left, exact_left),
                    "right_relative_error": _relative_error(right, exact_right),
                    "discrete_weak_identity_defect": abs(left - right),
                }
            )

    by_key = {(row["mode"], row["nodes"]): row for row in rows}
    orders: list[dict] = []
    for mode in MODES:
        for coarse, fine in zip(NODES[:-1], NODES[1:]):
            coarse_error = by_key[(mode, coarse)]["left_relative_error"]
            fine_error = by_key[(mode, fine)]["left_relative_error"]
            order = (
                math.log(coarse_error / fine_error, 2.0)
                if coarse_error > 0.0 and fine_error > 0.0
                else None
            )
            orders.append(
                {"mode": mode, "coarse_nodes": coarse, "fine_nodes": fine, "observed_order": order}
            )

    ratios: list[dict] = []
    for nodes in NODES:
        e2 = by_key[(2, nodes)]["left_relative_error"]
        e5 = by_key[(5, nodes)]["left_relative_error"]
        ratios.append(
            {
                "nodes": nodes,
                "mode5_to_mode2_left_error_ratio": e5 / max(e2, 1e-300),
                "classical_frequency_fourth_power_ratio": (5.0 / 2.0) ** 4,
            }
        )

    payload = {
        "status": "FROZEN_MANUFACTURED_CONTROL",
        "design_freeze_sha256": design_sha(),
        "trajectory": "exp(0.7*s)+0.2*sin(3*pi*s)",
        "quadrature": "composite_simpson",
        "reference": "scipy.integrate.quad epsabs=epsrel=1e-13",
        "rows": rows,
        "refinement_orders": orders,
        "mode_ratios": ratios,
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload["mode_ratios"], indent=2))
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()

