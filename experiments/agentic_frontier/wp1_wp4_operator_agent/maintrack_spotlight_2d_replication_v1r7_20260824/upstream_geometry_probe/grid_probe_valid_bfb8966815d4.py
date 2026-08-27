#!/usr/bin/env python3
"""Read-only, geometry-only probe for The Well Gray--Scott VALID files.

This script deliberately opens only ``data/valid`` paths and reads one
trajectory at five predeclared time indices.  It reports coordinate/shape
metadata and boundary-neighbour differences needed to distinguish duplicated
periodic faces from an endpoint-exclusive periodic lattice.  It does not
decode terminals, supports, library labels, or any TEST file.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(
    "/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4/"
    "data/the_well/data/valid"
)
TIME_INDICES = np.asarray([0, 20, 500, 980, 1000], dtype=np.int64)


def summaries(field: np.ndarray) -> dict[str, dict[str, float]]:
    """Compare wrap-neighbour and ordinary one-cell differences."""

    pairs = {
        "x_wrap_first_last": np.abs(field[:, 0, :] - field[:, -1, :]),
        "x_first_second": np.abs(field[:, 0, :] - field[:, 1, :]),
        "x_last_penultimate": np.abs(field[:, -1, :] - field[:, -2, :]),
        "y_wrap_first_last": np.abs(field[:, :, 0] - field[:, :, -1]),
        "y_first_second": np.abs(field[:, :, 0] - field[:, :, 1]),
        "y_last_penultimate": np.abs(field[:, :, -1] - field[:, :, -2]),
    }
    result: dict[str, dict[str, float]] = {}
    for name, values in pairs.items():
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        result[name] = {
            "min": float(np.min(flat)),
            "q25": float(np.quantile(flat, 0.25)),
            "median": float(np.median(flat)),
            "q75": float(np.quantile(flat, 0.75)),
            "q95": float(np.quantile(flat, 0.95)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "nonzero_fraction": float(np.count_nonzero(flat) / flat.size),
        }
    result["ratios"] = {
        "x_wrap_mean_over_ordinary_mean": (
            result["x_wrap_first_last"]["mean"]
            / max(
                0.5
                * (
                    result["x_first_second"]["mean"]
                    + result["x_last_penultimate"]["mean"]
                ),
                np.finfo(np.float64).tiny,
            )
        ),
        "y_wrap_mean_over_ordinary_mean": (
            result["y_wrap_first_last"]["mean"]
            / max(
                0.5
                * (
                    result["y_first_second"]["mean"]
                    + result["y_last_penultimate"]["mean"]
                ),
                np.finfo(np.float64).tiny,
            )
        ),
    }
    return result


def inspect(path: Path) -> dict[str, object]:
    if path.parent.name != "valid" or "test" in path.parts:
        raise RuntimeError(f"refusing non-VALID path: {path}")
    with h5py.File(path, "r", swmr=True) as handle:
        x = np.asarray(handle["dimensions/x"][:])
        y = np.asarray(handle["dimensions/y"][:])
        fields = {}
        for name in ("A", "B"):
            dataset = handle[f"t0_fields/{name}"]
            # h5py requires monotonic fancy indices; these are predeclared.
            values = np.asarray(dataset[0, TIME_INDICES, :, :])
            fields[name] = {
                "dataset_shape": list(dataset.shape),
                "dataset_dtype": str(dataset.dtype),
                "dataset_chunks": list(dataset.chunks) if dataset.chunks else None,
                "probe_shape": list(values.shape),
                "boundary_differences": summaries(values),
            }
        return {
            "file": path.name,
            "split": "valid",
            "trajectory_index": 0,
            "time_indices": TIME_INDICES.tolist(),
            "root_attrs": {
                key: np.asarray(value).tolist()
                for key, value in handle.attrs.items()
            },
            "x": {
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "first": float(x[0]),
                "second": float(x[1]),
                "penultimate": float(x[-2]),
                "last": float(x[-1]),
                "unique": int(np.unique(x).size),
                "step_min": float(np.min(np.diff(x))),
                "step_max": float(np.max(np.diff(x))),
            },
            "y": {
                "dtype": str(y.dtype),
                "shape": list(y.shape),
                "first": float(y[0]),
                "second": float(y[1]),
                "penultimate": float(y[-2]),
                "last": float(y[-1]),
                "unique": int(np.unique(y).size),
                "step_min": float(np.min(np.diff(y))),
                "step_max": float(np.max(np.diff(y))),
            },
            "fields": fields,
        }


def main() -> None:
    paths = sorted(ROOT.glob("*.hdf5"))
    if len(paths) != 6:
        raise RuntimeError(f"expected six VALID files, found {len(paths)}")
    print(json.dumps({"probe": "geometry_only", "files": [inspect(p) for p in paths]}, sort_keys=True))


if __name__ == "__main__":
    main()
