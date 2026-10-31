"""Source validation, finite authority, threshold selection, and stage runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from . import contract
from .evidence import Evidence, build_from_projection, project

@dataclass(frozen=True)
class Diagnostics:
    support: tuple[str, ...]
    residual: float
    gap: float
    condition: float
    minimum: float

def md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_source(path: Path) -> None:
    if path.name != contract.FILENAME or path.stat().st_size != contract.FILE_BYTES:
        raise ValueError("PDEBench source name/size mismatch")
    if md5(path) != contract.FILE_MD5:
        raise ValueError("PDEBench source MD5 mismatch")

def publisher_axis() -> np.ndarray:
    fine_index = (
        contract.PUBLISHER_FIRST_FINE_INDEX
        + contract.PUBLISHER_DECIMATION_STRIDE * np.arange(contract.SPATIAL_NODES)
    )
    return -1.0 + (2.0 * fine_index + 1.0) / contract.PUBLISHER_FINE_NODES

def read_source(handle, source_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = f"{source_id:04d}"
    if key not in handle:
        raise KeyError(f"missing publisher source {key}")
    group = handle[key]
    data = np.asarray(group["data"][:], dtype=np.float64)
    x = np.asarray(group["grid/x"][:], dtype=np.float64)
    y = np.asarray(group["grid/y"][:], dtype=np.float64)
    t = np.asarray(group["grid/t"][:], dtype=np.float64)
    if data.shape != (101, 128, 128, 2) or x.shape != (128,) or y.shape != (128,) or t.shape != (101,):
        raise ValueError("publisher schema drift")
    expected = publisher_axis()
    if not np.allclose(x, expected, rtol=0.0, atol=2.0e-7) or not np.allclose(y, expected, rtol=0.0, atol=2.0e-7):
        raise ValueError("publisher decimated spatial grid drift")
    expected_time = np.linspace(0.0, 5.0, 101)
    if not np.allclose(t, expected_time, rtol=0.0, atol=3.0e-7):
        raise ValueError("publisher time grid drift")
    if not np.all(np.diff(t) > 0) or not np.all(np.isfinite(data)):
        raise ValueError("publisher time/data invalid")
    return data, x, y, t

def fit(e: Evidence) -> Diagnostics:
    roles = np.asarray(e.roles)
    train, valid = roles == "train", roles == "validate"
    candidates = []
    for size in range(1, len(e.atoms) + 1):
        for cols in combinations(range(len(e.atoms)), size):
            xt = e.design[train][:, cols]
            norms = np.linalg.norm(xt, axis=0)
            if np.any(norms <= 1.0e-13):
                continue
            beta_n, _, rank, _ = np.linalg.lstsq(xt / norms, e.target[train], rcond=None)
            if rank != len(cols):
                continue
            beta = beta_n / norms
            residual = np.linalg.norm(e.target[valid] - e.design[valid][:, cols] @ beta)
            score = float(residual / max(np.linalg.norm(e.target[valid]), 1.0e-15) + 2.0e-5 * size)
            candidates.append((score, size, cols))
    if not candidates:
        return Diagnostics((), math.inf, 0.0, math.inf, 0.0)
    candidates.sort(key=lambda z: (z[0], z[1], tuple(e.atoms[i] for i in z[2])))
    best = candidates[0]
    winner = set(best[2])
    competitors = [c for c in candidates[1:] if not winner < set(c[2])]
    second = competitors[0][0] if competitors else math.inf
    cols = best[2]
    full = e.design[:, cols]
    norms = np.linalg.norm(full, axis=0)
    beta_n, _, rank, _ = np.linalg.lstsq(full / norms, e.target, rcond=None)
    beta = beta_n / norms
    residual = float(np.linalg.norm(e.target - full @ beta) / max(np.linalg.norm(e.target), 1.0e-15))
    condition = float(np.linalg.cond(full / norms)) if rank == len(cols) else math.inf
    return Diagnostics(
        tuple(e.atoms[i] for i in cols), residual,
        float(second - best[0]) if math.isfinite(second) else math.inf,
        condition, float(np.min(np.abs(beta))) if len(beta) else 0.0,
    )

def terminal(d: Diagnostics, th: contract.Thresholds) -> tuple[str, tuple[str, ...]]:
    if d.residual >= th.outside_library_residual_min:
        return "OUTSIDE_LIBRARY", ()
    if (d.residual <= th.identified_residual_max and d.condition <= th.condition_max
            and d.minimum >= th.coefficient_floor and d.gap >= th.support_gap_min):
        return "IDENTIFIED", d.support
    return "INCONCLUSIVE", ()

def correct(status: str, support: tuple[str, ...], contract_id: str) -> bool:
    target_name, _, truth_kind = contract.CONTRACTS[contract_id]
    if truth_kind == "outside_library":
        return status == "OUTSIDE_LIBRARY"
    return status == "IDENTIFIED" and support == contract.ATOMS[target_name]

def aggregate_reference(rows: list[dict], th: contract.Thresholds) -> dict:
    by_contract = {}
    for contract_id in contract.CONTRACTS:
        subset = [r for r in rows if r["contract_id"] == contract_id]
        c = f = inc = 0
        for row in subset:
            status, support = terminal(Diagnostics(**row["diagnostics"]), th)
            if status == "INCONCLUSIVE":
                inc += 1
            elif correct(status, support, contract_id):
                c += 1
            else:
                f += 1
        by_contract[contract_id] = {"n": len(subset), "correct": c, "false": f, "inconclusive": inc}
    return by_contract

def feasible(summary: dict) -> bool:
    full = [summary["full_u"], summary["full_v"]]
    if sum(x["correct"] for x in full) / sum(x["n"] for x in full) < 0.95:
        return False
    if sum(x["false"] for x in full) != 0:
        return False
    return all(
        v["correct"] / v["n"] >= 0.90
        for k, v in summary.items() if k.startswith("omit_")
    )

def choose_threshold(rows: list[dict]) -> tuple[contract.Thresholds, dict]:
    candidates = []
    for th in contract.THRESHOLD_GRID:
        summary = aggregate_reference(rows, th)
        if not feasible(summary):
            continue
        omission_rates = [v["correct"] / v["n"] for k, v in summary.items() if k.startswith("omit_")]
        full = [summary["full_u"], summary["full_v"]]
        full_rate = sum(x["correct"] for x in full) / sum(x["n"] for x in full)
        coverage = sum(v["correct"] + v["false"] for v in summary.values()) / sum(v["n"] for v in summary.values())
        candidates.append((min(omission_rates), full_rate, coverage, th, summary))
    if not candidates:
        raise SystemExit("AUTHORITY_DEVELOPMENT_BLOCKED_NO_FEASIBLE_THRESHOLD")
    best = min(candidates, key=lambda z: (-z[0], -z[1], -z[2], z[3]))
    return best[3], best[4]

def reference_rows(path: Path, source_ids: tuple[int, ...]) -> list[dict]:
    import h5py

    rows = []
    with h5py.File(path, "r") as handle:
        for source_id in source_ids:
            data, x, y, times = read_source(handle, source_id)
            for noise in contract.NOISE_LEVELS:
                bundle = project(source_id=source_id, data=data, x=x, y=y,
                                 times=times, noise_level=noise)
                for contract_id, (target_name, atoms, _) in contract.CONTRACTS.items():
                    evidence = build_from_projection(
                        source_id=source_id, bundle=bundle,
                        target_name=target_name, allowed_atoms=atoms,
                        temporal_samples=101, test_modes=5, noise_level=noise,
                    )
                    rows.append({
                        "source_id": source_id, "noise": noise,
                        "contract_id": contract_id,
                        "diagnostics": asdict(fit(evidence)),
                    })
    return rows

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)

def runtime_info() -> dict[str, str]:
    import h5py

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("development", "audit"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold-file", type=Path)
    args = p.parse_args(argv)
    contract.validate()
    validate_source(args.data)
    if args.stage == "development":
        ids = contract.DEVELOPMENT_IDS
        rows = reference_rows(args.data, ids)
        th, summary = choose_threshold(rows)
        value = {
            "schema": contract.SCHEMA, "stage": "development",
            "contract_sha256": contract.CONTRACT_SHA256,
            "source_md5": contract.FILE_MD5, "source_ids": ids,
            "runtime": runtime_info(),
            "selected_thresholds": asdict(th), "summary": summary,
            "status": "DEVELOPMENT_AUTHORITY_FEASIBLE",
        }
    else:
        if args.threshold_file is None:
            raise SystemExit("audit requires --threshold-file")
        frozen = json.loads(args.threshold_file.read_text())
        if frozen.get("contract_sha256") != contract.CONTRACT_SHA256:
            raise SystemExit("threshold contract binding mismatch")
        th = contract.Thresholds(**frozen["selected_thresholds"])
        ids = contract.AUDIT_IDS
        rows = reference_rows(args.data, ids)
        summary = aggregate_reference(rows, th)
        passed = feasible(summary)
        value = {
            "schema": contract.SCHEMA, "stage": "audit",
            "contract_sha256": contract.CONTRACT_SHA256,
            "source_md5": contract.FILE_MD5, "source_ids": ids,
            "runtime": runtime_info(),
            "selected_thresholds": asdict(th), "summary": summary,
            "all_authority_gates_passed": passed,
            "status": "AUTHORITY_AUDIT_PASSED" if passed else "AUTHORITY_AUDIT_FAILED",
            "confirmation_remains_sealed": True,
        }
    write_json(args.output, value)
    print(json.dumps(value, sort_keys=True))
    if args.stage == "audit" and not value["all_authority_gates_passed"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
