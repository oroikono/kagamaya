from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = {
    "cases": "cda68d3da3b74edc0f998e651f4d23c0c5c9add2d18ff55294db38db86451f6d",
    "labels": "66e048f1aed71764797eea7a6306de8fcc7e800857caffc6eed692f0d337aaa9",
}
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
MULTIPLIERS = tuple(10.0 ** (k / 4.0) for k in range(-8, 9))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def margin(row: dict[str, Any], identified_scale: float, omitted_scale: float) -> float:
    residual = float(row["best_residual"])
    gap = float(row["score_gap"])
    condition = float(row["best_condition"])
    identified = min(
        1.0 - residual / identified_scale,
        gap / 0.006 - 1.0,
        1.0 - condition / 12.0,
    )
    omitted = residual / omitted_scale - 1.0
    status = str(row["terminal_status"])
    if status == "IDENTIFIED":
        value = identified
    elif status == "OUTSIDE_LIBRARY":
        value = omitted
    elif status == "INCONCLUSIVE":
        value = max(identified, omitted)
    else:
        raise ValueError(f"unknown status {status}")
    return max(-4.0, min(4.0, value))


def objective(row: dict[str, Any], nodes: int, identified_scale: float, omitted_scale: float) -> float:
    terminal = row["terminal_status"] != "INCONCLUSIVE"
    return 10.0 * int(terminal) + margin(row, identified_scale, omitted_scale) - 1e-3 * nodes / 49.0


def choose(case: dict[str, Any], producer: str, identified_scale: float, omitted_scale: float) -> dict[str, Any]:
    scored = []
    for action in case["public_rows"]:
        token = str(action["action_token"])
        row = action[producer]
        score = objective(row, int(action["nodes"]), identified_scale, omitted_scale)
        scored.append((score, token, action, row))
    best_score = max(item[0] for item in scored)
    _, token, action, row = min((item for item in scored if item[0] == best_score), key=lambda item: item[1])
    return {
        "case_id": case["case_id"],
        "producer": producer,
        "action_token": token,
        "nodes": int(action["nodes"]),
        "temporal_modes": int(action["temporal_modes"]),
        "terminal_status": row["terminal_status"],
        "terminal_support": row.get("terminal_support"),
        "objective": best_score,
    }


def correct(choice: dict[str, Any], label: dict[str, Any]) -> bool:
    held = label["held_out"]
    if bool(held["in_library"]):
        return choice["terminal_status"] == "IDENTIFIED" and choice["terminal_support"] == held["truth_support"]
    return choice["terminal_status"] == "OUTSIDE_LIBRARY"


def summarize(choices: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inlib = [row for row in choices if labels[row["case_id"]]["held_out"]["in_library"]]
    oog = [row for row in choices if not labels[row["case_id"]]["held_out"]["in_library"]]
    false = sum(row["terminal_status"] != "INCONCLUSIVE" and not correct(row, labels[row["case_id"]]) for row in inlib)
    detected = sum(correct(row, labels[row["case_id"]]) for row in oog)
    return {
        "in_library_n": len(inlib),
        "in_library_false": false,
        "false_promotion_rate": false / len(inlib),
        "omitted_n": len(oog),
        "omitted_detected": detected,
        "omission_detection_rate": detected / len(oog),
        "mean_nodes": sum(row["nodes"] for row in choices) / len(choices),
        "action_histogram": {
            token: sum(row["action_token"] == token for row in choices)
            for token in sorted({row["action_token"] for row in choices})
        },
    }


def frontier(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cell in cells:
        fp = cell["false_promotion_rate"]
        det = cell["omission_detection_rate"]
        dominated = any(
            other["false_promotion_rate"] <= fp
            and other["omission_detection_rate"] >= det
            and (other["false_promotion_rate"] < fp or other["omission_detection_rate"] > det)
            for other in cells
        )
        if not dominated:
            result.append(cell)
    return sorted(result, key=lambda row: (row["false_promotion_rate"], -row["omission_detection_rate"], row["identified_scale"], row["omitted_scale"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.cases) != EXPECTED["cases"] or sha256(args.labels) != EXPECTED["labels"]:
        raise SystemExit("bound input hash mismatch")
    cases = load_jsonl(args.cases)
    if len(cases) != 216 or any(len(case["public_rows"]) != 12 for case in cases):
        raise SystemExit("unexpected public table shape")

    # Public selection is completed before the held-out sidecar is opened.
    public_choices: dict[str, list[dict[str, Any]]] = {producer: [] for producer in PRODUCERS}
    cell_choices: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for producer in PRODUCERS:
        for i, mi in enumerate(MULTIPLIERS):
            for j, mo in enumerate(MULTIPLIERS):
                choices = [choose(case, producer, 0.05 * mi, 0.20 * mo) for case in cases]
                cell_choices[(producer, i, j)] = choices
                public_choices[producer].extend(choices)

    label_rows = load_jsonl(args.labels)
    labels = {row["case_id"]: row for row in label_rows}
    if set(labels) != {case["case_id"] for case in cases}:
        raise SystemExit("case/sidecar key mismatch")

    producer_results: dict[str, Any] = {}
    for producer in PRODUCERS:
        cells = []
        for i, mi in enumerate(MULTIPLIERS):
            for j, mo in enumerate(MULTIPLIERS):
                summary = summarize(cell_choices[(producer, i, j)], labels)
                cells.append({
                    "identified_multiplier": mi,
                    "omitted_multiplier": mo,
                    "identified_scale": 0.05 * mi,
                    "omitted_scale": 0.20 * mo,
                    **summary,
                })
        original = next(cell for cell in cells if math.isclose(cell["identified_scale"], 0.05) and math.isclose(cell["omitted_scale"], 0.20))
        regions = {
            "fp_le_1pct_oog_ge_75pct": sum(cell["false_promotion_rate"] <= 0.01 and cell["omission_detection_rate"] >= 0.75 for cell in cells),
            "fp_le_5pct_oog_ge_75pct": sum(cell["false_promotion_rate"] <= 0.05 and cell["omission_detection_rate"] >= 0.75 for cell in cells),
            "fp_le_10pct_oog_ge_90pct": sum(cell["false_promotion_rate"] <= 0.10 and cell["omission_detection_rate"] >= 0.90 for cell in cells),
        }
        producer_results[producer] = {
            "original": original,
            "operating_region_cell_counts": regions,
            "frontier": frontier(cells),
            "cells": cells,
        }

    body: dict[str, Any] = {
        "schema_version": "evidence_view_benchmark.objective_constants_sweep.v1",
        "retrospective": True,
        "case_count": len(cases),
        "grid_shape": [len(MULTIPLIERS), len(MULTIPLIERS)],
        "input_sha256": {"cases": sha256(args.cases), "labels": sha256(args.labels)},
        "producers": producer_results,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
