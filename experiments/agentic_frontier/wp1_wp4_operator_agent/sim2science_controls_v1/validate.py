"""Validators for the frozen Sim2Science mechanism controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .denested_bank import CONFIRMATION_SEEDS, NODES, source_sha
from .manufactured_quadrature import MODES as QUADRATURE_MODES
from .manufactured_quadrature import NODES as QUADRATURE_NODES


def _digest_without_content(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_manufactured(path: Path) -> None:
    payload = json.loads(path.read_text())
    assert payload["status"] == "FROZEN_MANUFACTURED_CONTROL"
    assert payload["content_sha256"] == _digest_without_content(payload)
    rows = payload["rows"]
    assert len(rows) == len(QUADRATURE_MODES) * len(QUADRATURE_NODES)
    assert {(row["mode"], row["nodes"]) for row in rows} == {
        (mode, nodes) for mode in QUADRATURE_MODES for nodes in QUADRATURE_NODES
    }
    for row in rows:
        assert abs(row["exact_left"] - row["exact_right"]) < 1e-11
        assert row["left_relative_error"] >= 0.0
        assert row["right_relative_error"] >= 0.0
        assert row["discrete_weak_identity_defect"] >= 0.0


def validate_denested(path: Path) -> None:
    payload = json.loads(path.read_text())
    assert payload["status"] == "FROZEN_RESERVED_SEED_CONFIRMATION"
    assert payload["content_sha256"] == _digest_without_content(payload)
    assert payload["source_closure_sha256"] == source_sha()
    assert tuple(payload["config"]["seeds"]) == CONFIRMATION_SEEDS
    episodes = payload["episodes"]
    assert len(episodes) == 720
    keys = [tuple(episode["episode_key"]) for episode in episodes]
    assert len(set(keys)) == 720
    hidden = {(key[0], key[1]) for key in keys}
    assert len(hidden) == 120
    for episode in episodes:
        assert set(episode["policies"]) == {
            "nested_gate", "disjoint_gate", "fixed_fine_cumulative"
        }
        for row in episode["policies"].values():
            assert row["nodes"] in NODES
            assert sum((row["correct"], row["false_terminal"], row["unidentified"])) == 1
    for policy, summary in payload["summary"].items():
        rows = [episode["policies"][policy] for episode in episodes]
        assert summary["factorial_conditions"] == 720
        assert summary["hidden_trajectory_realizations"] == 120
        assert summary["correct"] == sum(row["correct"] for row in rows)
        assert summary["false_terminals"] == sum(row["false_terminal"] for row in rows)
        assert summary["unidentified"] == sum(row["unidentified"] for row in rows)
        assert summary["correct"] + summary["false_terminals"] + summary["unidentified"] == 720
        assert abs(summary["mean_terminal_nodes"] - sum(row["nodes"] for row in rows) / 720) < 1e-12
        assert sum(summary["stop_nodes"].values()) == 720
    assert sum(payload["transitions"].values()) == 720
    assert payload["gate_verdict"]["all_pass"] == all(
        payload["gate_verdict"]["checks"].values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manufactured", type=Path)
    parser.add_argument("--denested", type=Path)
    args = parser.parse_args()
    if not args.manufactured and not args.denested:
        raise SystemExit("provide --manufactured and/or --denested")
    if args.manufactured:
        validate_manufactured(args.manufactured)
        print(f"PASS manufactured {args.manufactured}")
    if args.denested:
        validate_denested(args.denested)
        print(f"PASS denested {args.denested}")


if __name__ == "__main__":
    main()

