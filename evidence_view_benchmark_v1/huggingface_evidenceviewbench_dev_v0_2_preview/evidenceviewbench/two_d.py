"""Installable 2-D/coupled evidence-acquisition environment.

This module is a dependency-free packaging adapter for the frozen 2-D design
bank.  A policy receives only a view manifest and records that it explicitly
acquires.  Generating labels are loaded through a separate owner-side API and
are never consulted by the environment or trace replay.

The bundled rows are development data, not a hidden test set.  The interface
separation prevents accidental label use during a run; it does not create
secrecy after package installation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any, Iterable, Mapping


TERMINAL_STATUSES = ("IDENTIFIED", "OUTSIDE_LIBRARY", "INCONCLUSIVE")
ACTION_KINDS = ("ACQUIRE", "CHALLENGE", "COMMIT", "ABSTAIN")
SPATIAL_NODES = (12, 18, 24, 36)
TEMPORAL_SAMPLES = (7, 13, 25)
TEST_MODE_COUNTS = (2, 4)
WINDOWS = ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_gzip_jsonl(resource_name: str) -> list[dict[str, Any]]:
    resource = resources.files("evidenceviewbench.resources").joinpath(resource_name)
    with resource.open("rb") as raw:
        with gzip.open(raw, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def _validate_bound_row(row: Mapping[str, Any], hash_field: str) -> None:
    body = dict(row)
    claimed = body.pop(hash_field, None)
    if claimed != _digest(body):
        raise ValueError(f"{hash_field} does not bind the row contents")


def two_d_view_manifest() -> tuple[dict[str, Any], ...]:
    """Return the immutable 72-view acquisition manifest."""

    rows: list[dict[str, Any]] = []
    ordinal = 0
    for spatial_nodes in SPATIAL_NODES:
        for temporal_samples in TEMPORAL_SAMPLES:
            for test_mode_count in TEST_MODE_COUNTS:
                for window_start, window_end in WINDOWS:
                    rows.append(
                        {
                            "view_id": f"V{ordinal:03d}",
                            "spatial_nodes": spatial_nodes,
                            "temporal_samples": temporal_samples,
                            "test_mode_count": test_mode_count,
                            "window": [window_start, window_end],
                        }
                    )
                    ordinal += 1
    return tuple(rows)


_VIEW_MANIFEST = two_d_view_manifest()
_VIEW_IDS = {row["view_id"] for row in _VIEW_MANIFEST}


def load_two_d_public() -> list[dict[str, Any]]:
    """Load and validate 180 selector-visible development rows."""

    rows = _read_gzip_jsonl("two_d_design_public.jsonl.gz")
    if len(rows) != 180:
        raise ValueError("2-D public bank must contain 180 rows")
    keys: set[tuple[str, str]] = set()
    for row in rows:
        _validate_bound_row(row, "public_row_sha256")
        key = (str(row["origin_component"]), str(row["episode_id"]))
        if key in keys:
            raise ValueError(f"duplicate 2-D public row: {key}")
        keys.add(key)
        views = list(row.get("views", []))
        if len(views) != 72 or {str(view["view_id"]) for view in views} != _VIEW_IDS:
            raise ValueError(f"incomplete 2-D view bank: {key}")
        forbidden = {"family", "truth_kind", "truth_support", "omitted_mechanism"}
        if forbidden & set(row):
            raise ValueError(f"owner fields present in 2-D public row: {key}")
        for view in views:
            if view.get("episode_id") != row["episode_id"]:
                raise ValueError(f"cross-episode 2-D view: {key}")
            if view.get("status") not in TERMINAL_STATUSES:
                raise ValueError(f"unsupported 2-D terminal status: {key}")
            cost = view.get("registered_cost")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
                raise ValueError(f"invalid 2-D registered cost: {key}")
    return rows


def load_two_d_owner() -> list[dict[str, Any]]:
    """Load and validate 180 evaluator-side development labels."""

    rows = _read_gzip_jsonl("two_d_design_owner.jsonl.gz")
    if len(rows) != 180:
        raise ValueError("2-D owner bank must contain 180 rows")
    keys: set[tuple[str, str]] = set()
    for row in rows:
        _validate_bound_row(row, "owner_row_sha256")
        key = (str(row["origin_component"]), str(row["episode_id"]))
        if key in keys:
            raise ValueError(f"duplicate 2-D owner row: {key}")
        keys.add(key)
    return rows


def two_d_public_row(origin_component: str, episode_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in load_two_d_public()
        if row["origin_component"] == origin_component and row["episode_id"] == episode_id
    ]
    if len(matches) != 1:
        raise KeyError((origin_component, episode_id))
    return json.loads(_canonical_bytes(matches[0]))


def two_d_owner_row(origin_component: str, episode_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in load_two_d_owner()
        if row["origin_component"] == origin_component and row["episode_id"] == episode_id
    ]
    if len(matches) != 1:
        raise KeyError((origin_component, episode_id))
    return json.loads(_canonical_bytes(matches[0]))


@dataclass(frozen=True)
class TwoDAction:
    kind: str
    view_id: str | None = None

    @classmethod
    def acquire(cls, view_id: str) -> "TwoDAction":
        return cls("ACQUIRE", str(view_id))

    @classmethod
    def challenge(cls, view_id: str) -> "TwoDAction":
        return cls("CHALLENGE", str(view_id))

    @classmethod
    def commit(cls) -> "TwoDAction":
        return cls("COMMIT")

    @classmethod
    def abstain(cls) -> "TwoDAction":
        return cls("ABSTAIN")

    def validate(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unsupported action kind: {self.kind}")
        if self.kind in {"ACQUIRE", "CHALLENGE"}:
            if self.view_id not in _VIEW_IDS:
                raise ValueError("evidence action requires a frozen view_id")
        elif self.view_id is not None:
            raise ValueError("terminal actions cannot name a view")


@dataclass(frozen=True)
class TwoDTerminal:
    status: str
    support: tuple[str, ...]
    reason: str
    evidence_view_ids: tuple[str, ...]
    realized_cost: float


@dataclass(frozen=True)
class TwoDScore:
    correct: bool
    false_promotion: bool
    inconclusive: bool


def _consensus(records: Iterable[Mapping[str, Any]]) -> tuple[str, tuple[str, ...], str]:
    records = list(records)
    if not records:
        return "INCONCLUSIVE", (), "no_evidence"
    signatures = {
        (str(record["status"]), tuple(str(atom) for atom in record["support"]))
        for record in records
    }
    if len(signatures) == 1:
        status, support = next(iter(signatures))
        return status, support, "exact_cross_view_agreement"
    return "INCONCLUSIVE", (), "cross_view_disagreement"


class TwoDEvidenceEnv:
    """Budgeted public environment for one 2-D development row."""

    def __init__(self, public_row: Mapping[str, Any], budget: float) -> None:
        row = json.loads(_canonical_bytes(dict(public_row)))
        _validate_bound_row(row, "public_row_sha256")
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or budget < 0
        ):
            raise ValueError("budget must be a nonnegative finite number")
        self._row = row
        self.budget = float(budget)
        self._views = {str(view["view_id"]): view for view in row["views"]}
        if len(self._views) != 72 or set(self._views) != _VIEW_IDS:
            raise ValueError("public row does not contain the frozen 72-view bank")
        self.acquired: list[dict[str, Any]] = []
        self.actions: list[TwoDAction] = []
        self.terminal: TwoDTerminal | None = None

    @property
    def spent(self) -> float:
        return float(sum(float(record["registered_cost"]) for record in self.acquired))

    def public_state(self) -> dict[str, Any]:
        state = {
            "episode_id": self._row["episode_id"],
            "origin_component": self._row["origin_component"],
            "noise_model": self._row["noise_model"],
            "noise_level": self._row["noise_level"],
            "budget": self.budget,
            "spent": self.spent,
            "available_view_manifest": [
                {**manifest, "registered_cost": float(self._views[manifest["view_id"]]["registered_cost"])}
                for manifest in _VIEW_MANIFEST
            ],
            "acquired": json.loads(_canonical_bytes(self.acquired)),
        }
        forbidden = {"family", "truth_kind", "truth_support", "omitted_mechanism"}
        if forbidden & set(state):
            raise AssertionError("owner material leaked into public state")
        return state

    def step(self, action: TwoDAction) -> TwoDTerminal | None:
        action.validate()
        if self.terminal is not None:
            raise RuntimeError("episode already terminated")
        if action.kind in {"ACQUIRE", "CHALLENGE"}:
            assert action.view_id is not None
            if any(record["view_id"] == action.view_id for record in self.acquired):
                raise ValueError("repeat acquisitions are not admitted")
            record = self._views[action.view_id]
            if self.spent + float(record["registered_cost"]) > self.budget + 1e-9:
                raise ValueError("action exceeds registered budget")
            self.actions.append(action)
            self.acquired.append(json.loads(_canonical_bytes(record)))
            return None
        self.actions.append(action)
        if action.kind == "ABSTAIN":
            status, support, reason = "INCONCLUSIVE", (), "agent_abstained"
        else:
            status, support, reason = _consensus(self.acquired)
        self.terminal = TwoDTerminal(
            status=status,
            support=support,
            reason=reason,
            evidence_view_ids=tuple(str(record["view_id"]) for record in self.acquired),
            realized_cost=self.spent,
        )
        return self.terminal

    def export_trace(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "evidenceviewbench.two-d-trace.v1",
            "episode_id": self._row["episode_id"],
            "origin_component": self._row["origin_component"],
            "budget": self.budget,
            "actions": [asdict(action) for action in self.actions],
            "terminal": None if self.terminal is None else asdict(self.terminal),
        }
        # Normalize tuples to JSON arrays before hashing/returning so a trace
        # remains byte- and equality-stable after an actual JSON round trip.
        payload = json.loads(_canonical_bytes(payload))
        payload["trace_sha256"] = _digest(payload)
        return payload

    @classmethod
    def replay_trace(cls, public_row: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
        body = dict(trace)
        claimed = body.pop("trace_sha256", None)
        if claimed != _digest(body):
            raise ValueError("trace content hash mismatch")
        env = cls(public_row, budget=float(trace["budget"]))
        for action in trace["actions"]:
            env.step(TwoDAction(kind=str(action["kind"]), view_id=action.get("view_id")))
        replay = env.export_trace()
        if replay != dict(trace):
            raise ValueError("trace replay mismatch")
        return replay


def score_two_d_terminal(terminal: TwoDTerminal, owner_row: Mapping[str, Any]) -> TwoDScore:
    """Score a terminal against a separately supplied development label."""

    _validate_bound_row(owner_row, "owner_row_sha256")
    if terminal.status not in TERMINAL_STATUSES:
        raise ValueError("unknown terminal status")
    if terminal.status == "INCONCLUSIVE":
        return TwoDScore(False, False, True)
    correct = (
        owner_row["truth_kind"] == "in_library"
        and terminal.status == "IDENTIFIED"
        and terminal.support == tuple(str(atom) for atom in owner_row["truth_support"])
    ) or (
        owner_row["truth_kind"] == "outside_library"
        and terminal.status == "OUTSIDE_LIBRARY"
    )
    return TwoDScore(correct=bool(correct), false_promotion=not bool(correct), inconclusive=False)
