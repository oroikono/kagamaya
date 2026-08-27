"""Deployable and diagnostic policies for the agentic harness."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Protocol

import numpy as np

from .contract import VIEW_BANK
from .environment import AgentAction


class Policy(Protocol):
    policy_id: str

    def reset(self, public_seed: int) -> None: ...
    def act(self, public_state: dict) -> AgentAction: ...


@dataclass
class FixedOpenLoopPolicy:
    policy_id: str
    view_sequence: tuple[str, ...]

    def reset(self, public_seed: int) -> None:
        self._position = 0

    def act(self, public_state: dict) -> AgentAction:
        if self._position < len(self.view_sequence):
            view_id = self.view_sequence[self._position]
            self._position += 1
            return AgentAction("ACQUIRE" if self._position == 1 else "CHALLENGE", view_id)
        return AgentAction("COMMIT")


@dataclass
class CheapestThenChallengePolicy:
    policy_id: str = "cheapest_then_max_tests"

    def reset(self, public_seed: int) -> None:
        self._position = 0
        cheapest = min(VIEW_BANK, key=lambda view: (view.registered_cost, view.view_id))
        challenger = max(VIEW_BANK, key=lambda view: (
            view.test_mode_count,
            view.spatial_nodes,
            view.temporal_samples,
            -view.registered_cost,
        ))
        self._sequence = (cheapest.view_id, challenger.view_id)

    def act(self, public_state: dict) -> AgentAction:
        if self._position < 2:
            view_id = self._sequence[self._position]
            self._position += 1
            return AgentAction("ACQUIRE" if self._position == 1 else "CHALLENGE", view_id)
        return AgentAction("COMMIT")


@dataclass
class DiagnosticAdaptivePolicy:
    """Truth-blind adaptive challenge based only on the first public verdict."""

    policy_id: str = "diagnostic_adaptive"

    def reset(self, public_seed: int) -> None:
        self._position = 0
        self._first = min(VIEW_BANK, key=lambda view: (
            abs(view.spatial_nodes - 18), abs(view.temporal_samples - 13),
            abs(view.test_mode_count - 2), view.registered_cost,
        ))

    def act(self, public_state: dict) -> AgentAction:
        acquired = public_state["acquired"]
        if not acquired:
            return AgentAction("ACQUIRE", self._first.view_id)
        if len(acquired) == 1:
            record = acquired[0]
            # High residual calls for spatial/temporal fidelity. Small gap or
            # large ambiguity calls for test-space diversity. This is a frozen
            # heuristic comparator, not a learned claim.
            if record["residual"] > 0.08:
                candidates = sorted(VIEW_BANK, key=lambda view: (
                    -view.spatial_nodes, -view.temporal_samples,
                    view.registered_cost, view.view_id,
                ))
            else:
                candidates = sorted(VIEW_BANK, key=lambda view: (
                    -view.test_mode_count,
                    0 if view.window_start != self._first.window_start else 1,
                    view.registered_cost, view.view_id,
                ))
            second = next(view for view in candidates if view.view_id != self._first.view_id)
            return AgentAction("CHALLENGE", second.view_id)
        return AgentAction("COMMIT")


@dataclass
class RandomOpenLoopPolicy:
    policy_id: str
    length: int = 2

    def reset(self, public_seed: int) -> None:
        digest = hashlib.sha256(f"{self.policy_id}|{public_seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        self._sequence = tuple(rng.choice(
            [view.view_id for view in VIEW_BANK], size=self.length, replace=False))
        self._position = 0

    def act(self, public_state: dict) -> AgentAction:
        if self._position < self.length:
            view_id = self._sequence[self._position]
            self._position += 1
            return AgentAction("ACQUIRE" if self._position == 1 else "CHALLENGE", str(view_id))
        return AgentAction("COMMIT")


@dataclass(frozen=True)
class LinearPolicyReceipt:
    """Content-addressable future learner; fitting is gate-controlled."""

    view_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    training_digest: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps({
            "view_ids": self.view_ids,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "training_digest": self.training_digest,
        }, sort_keys=True).encode()).hexdigest()


@dataclass
class LockedLinearPolicy:
    receipt: LinearPolicyReceipt
    policy_id: str = "locked_linear_agent"

    def reset(self, public_seed: int) -> None:
        self._stage = 0

    def act(self, public_state: dict) -> AgentAction:
        if not public_state["acquired"]:
            # The initial action is stored as the first view row. The learner
            # is incapable of reading episode IDs or owner metadata.
            self._stage = 1
            return AgentAction("ACQUIRE", self.receipt.view_ids[0])
        if self._stage == 1:
            record = public_state["acquired"][-1]
            features = np.asarray([
                record["residual"], record["score_gap"],
                np.log1p(record["condition"]),
                record["minimum_coefficient"], record["ambiguity_size"],
            ], dtype=float)
            weights = np.asarray(self.receipt.weights, dtype=float)
            scores = weights @ features + np.asarray(self.receipt.bias)
            acquired_ids = {item["view_id"] for item in public_state["acquired"]}
            order = np.argsort(-scores, kind="stable")
            view_id = next(self.receipt.view_ids[index] for index in order
                           if self.receipt.view_ids[index] not in acquired_ids)
            self._stage = 2
            return AgentAction("CHALLENGE", view_id)
        return AgentAction("COMMIT")


def parse_llm_action(payload: str, allowed_view_ids: set[str]) -> AgentAction:
    """Provider-neutral strict JSON seam; no provider call is made here."""
    data = json.loads(payload)
    if set(data) != {"action", "view_id"}:
        raise ValueError("LLM action must contain exactly action and view_id")
    action = str(data["action"])
    view_id = data["view_id"]
    if action not in {"ACQUIRE", "CHALLENGE", "COMMIT", "ABSTAIN"}:
        raise ValueError("unknown LLM action")
    if action in {"ACQUIRE", "CHALLENGE"} and view_id not in allowed_view_ids:
        raise ValueError("LLM selected view outside frozen manifest")
    if action in {"COMMIT", "ABSTAIN"} and view_id is not None:
        raise ValueError("terminal LLM action must set view_id to null")
    return AgentAction(action, view_id)


def run_policy(policy: Policy, public_episode, *, budget: float, public_seed: int):
    from .environment import EvidenceEnvironment

    environment = EvidenceEnvironment(public_episode=public_episode, budget=budget)
    policy.reset(public_seed)
    for _ in range(4):
        terminal = environment.step(policy.act(environment.public_state()))
        if terminal is not None:
            return terminal, environment
    raise RuntimeError("policy exceeded four-step horizon without terminating")

