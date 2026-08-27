"""Deterministic, metadata-blind acquisition environment for the public dev set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_BANK_PATH = PACKAGE_ROOT / "resources/view_bank.jsonl"
DEFAULT_CASE_REGISTRY_PATH = PACKAGE_ROOT / "resources/dev_cases.jsonl"
ALLOWED_NODES = (7, 13, 25, 49)
ALLOWED_MODES = (2, 3, 4, 5)
ALLOWED_PRODUCERS = ("spectral_etdrk4", "fd_bdf")
ALLOWED_VISIBILITIES = ("acquire_only",)
ALLOWED_AUTHORITY_STATUSES = ("IDENTIFIED", "OUTSIDE_LIBRARY", "INCONCLUSIVE")
MINIMUM_VIEW_COST = min(ALLOWED_NODES)
MAX_TOOL_CALLS = 12
FORBIDDEN_FRAGMENTS = (
    "truth",
    "held_out",
    "correct",
    "false",
    "owner",
    "oracle",
    "reference",
    "error_to_49",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden_paths(value: Any, location: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS):
                findings.append(f"{location}.{key}")
            findings.extend(forbidden_paths(nested, f"{location}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            findings.extend(forbidden_paths(nested, f"{location}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS):
            findings.append(location)
    return findings


def assert_public_payload(value: Mapping[str, Any]) -> None:
    findings = forbidden_paths(value)
    if findings:
        raise ValueError(f"forbidden material in public payload: {findings[:5]}")


@dataclass(frozen=True)
class Action:
    kind: str
    nodes: int | None = None
    modes: int | None = None
    reason: str | None = None

    @classmethod
    def acquire_view(cls, nodes: int, modes: int) -> "Action":
        return cls("acquire_view", nodes=int(nodes), modes=int(modes))

    @classmethod
    def commit(cls, nodes: int, modes: int) -> "Action":
        return cls("commit", nodes=int(nodes), modes=int(modes))

    @classmethod
    def abstain(cls, reason: str) -> "Action":
        return cls("abstain", reason=str(reason))


class EvidenceViewEnv:
    """Lookup-only evidence-view environment with deterministic accounting."""

    _bank_cache: ClassVar[dict[Path, tuple[str, dict[str, dict[str, Any]]]]] = {}
    _case_registry_cache: ClassVar[dict[Path, tuple[str, dict[str, dict[str, Any]]]]] = {}

    def __init__(
        self,
        case: Mapping[str, Any],
        budget: int,
        visibility: str,
        *,
        producer: str = "spectral_etdrk4",
        acquisition_bank_path: str | Path | None = None,
        case_registry_path: str | Path | None = None,
    ) -> None:
        self.case = json.loads(canonical_bytes(dict(case)))
        self._validate_case()
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if visibility not in ALLOWED_VISIBILITIES:
            raise ValueError(f"unsupported visibility: {visibility}")
        if producer not in ALLOWED_PRODUCERS:
            raise ValueError(f"unsupported producer: {producer}")
        self.budget = budget
        self.spent = 0
        self.remaining = budget
        self.visibility = visibility
        self.producer = producer
        self.done = False
        self.tool_calls = 0
        self.termination: str | None = None
        self._events: list[dict[str, Any]] = []
        self._acquired: dict[tuple[int, int], dict[str, Any]] = {}
        self._invalid_commit_attempts = 0
        self._floor_warning_active = budget < MINIMUM_VIEW_COST
        selected_path = Path(acquisition_bank_path or DEFAULT_BANK_PATH).resolve()
        self._bank_path = selected_path
        self._bank_file_sha256, bank = self._load_bank(selected_path)
        selected_registry = Path(case_registry_path or DEFAULT_CASE_REGISTRY_PATH).resolve()
        self._case_registry_path = selected_registry
        self._case_registry_file_sha256, registry = self._load_case_registry(selected_registry)
        try:
            bank_case = bank[str(self.case["case_id"])]
        except KeyError as exc:
            raise ValueError("case is absent from acquisition bank") from exc
        if bank_case["public_case_sha256"] != self._case_binding_sha256:
            raise ValueError("case and acquisition bank bindings differ")
        try:
            registered_case = registry[str(self.case["case_id"])]
        except KeyError as exc:
            raise ValueError("case is absent from the case registry") from exc
        if registered_case["case_sha256"] != self._case_binding_sha256:
            raise ValueError("case and registry bindings differ")
        if canonical_bytes(registered_case["public_rows"]) != canonical_bytes(self.case["public_rows"]):
            raise ValueError("selector rows differ from the bound case registry")
        self._views = self._build_view_table(bank_case)

    def _validate_case(self) -> None:
        schema = self.case.get("schema_version")
        body = dict(self.case)
        if schema == "evidence_view_benchmark.public_case.v1":
            claimed = body.pop("case_sha256", None)
            if claimed != digest(body):
                raise ValueError("public case content hash mismatch")
            self._case_binding_sha256 = str(claimed)
        elif schema == "evidence_view_benchmark.selection_case.v1":
            claimed = body.pop("selection_case_sha256", None)
            if claimed != digest(body):
                raise ValueError("selection case content hash mismatch")
            parent = body.get("parent_public_case_sha256")
            if not isinstance(parent, str) or len(parent) != 64:
                raise ValueError("selection case has invalid parent binding")
            try:
                int(parent, 16)
            except ValueError as exc:
                raise ValueError("selection case has invalid parent binding") from exc
            self._case_binding_sha256 = parent
        else:
            raise ValueError("unsupported public case schema")
        assert_public_payload(body)
        rows = body.get("public_rows")
        if not isinstance(rows, list) or len(rows) != 12:
            raise ValueError("public case must contain twelve Gate-1 views")

    @classmethod
    def _load_bank(cls, bank_path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
        if not bank_path.is_file():
            raise FileNotFoundError(bank_path)
        bank_hash = file_sha256(bank_path)
        cached = cls._bank_cache.get(bank_path)
        if cached is not None and cached[0] == bank_hash:
            return cached
        records: dict[str, dict[str, Any]] = {}
        with bank_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                body = dict(row)
                claimed = body.pop("acquisition_case_sha256", None)
                if claimed != digest(body):
                    raise ValueError(f"acquisition bank hash mismatch at line {line_number}")
                assert_public_payload(body)
                case_id = str(row["case_id"])
                if case_id in records:
                    raise ValueError(f"duplicate acquisition bank case: {case_id}")
                records[case_id] = row
        if len(records) != 216:
            raise ValueError("acquisition bank must contain 216 cases")
        result = (bank_hash, records)
        cls._bank_cache[bank_path] = result
        return result

    @classmethod
    def _load_case_registry(cls, registry_path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
        if not registry_path.is_file():
            raise FileNotFoundError(registry_path)
        registry_hash = file_sha256(registry_path)
        cached = cls._case_registry_cache.get(registry_path)
        if cached is not None and cached[0] == registry_hash:
            return cached
        records: dict[str, dict[str, Any]] = {}
        with registry_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                body = dict(row)
                claimed = body.pop("case_sha256", None)
                if claimed != digest(body):
                    raise ValueError(f"case registry hash mismatch at line {line_number}")
                case_id = str(row["case_id"])
                if case_id in records:
                    raise ValueError(f"duplicate case registry row: {case_id}")
                records[case_id] = row
        if len(records) != 216:
            raise ValueError("case registry must contain 216 cases")
        result = (registry_hash, records)
        cls._case_registry_cache[registry_path] = result
        return result

    def _build_view_table(self, bank_case: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
        rows = list(self.case["public_rows"]) + list(bank_case["views"])
        table: dict[tuple[int, int], dict[str, Any]] = {}
        allowed = {(n, m) for n in ALLOWED_NODES for m in ALLOWED_MODES}
        for row in rows:
            coordinate = (int(row["nodes"]), int(row["temporal_modes"]))
            if coordinate in table:
                raise ValueError(f"duplicate view coordinate: {coordinate}")
            if coordinate not in allowed:
                raise ValueError(f"unsupported view coordinate: {coordinate}")
            public = row[self.producer]
            status = str(public["terminal_status"])
            if status not in ALLOWED_AUTHORITY_STATUSES:
                raise ValueError(f"unsupported authority status: {status}")
            view = {
                "nodes": coordinate[0],
                "modes": coordinate[1],
                "terminal": status,
                "support": public["terminal_support"],
                "residual": float(public["best_residual"]),
                "score_gap": float(public["score_gap"]),
                "gram_condition": float(public["gram_condition"]),
            }
            assert_public_payload(view)
            table[coordinate] = view
        if set(table) != allowed:
            raise ValueError("public and acquisition rows do not cover the 16-view grid")
        return table

    def _base_state(self) -> dict[str, Any]:
        # The raw development files deliberately expose family/noise metadata for
        # analysis.  The selection interface does not.  A future hidden
        # service must additionally replace case_id with a random opaque nonce.
        return {
            "case_id": self.case["case_id"],
            "budget_total": self.budget,
            "budget_spent": self.spent,
            "budget_remaining": self.remaining,
            "tool_calls_used": self.tool_calls,
            "tool_calls_max": MAX_TOOL_CALLS,
        }

    def reset(self) -> dict[str, Any]:
        if self._events:
            raise RuntimeError("reset may be called only once")
        observation = {
            "schema_version": "evidence_view_benchmark.observation.v1",
            "event": "episode_reset",
            "visibility": self.visibility,
            "producer": self.producer,
            "available_nodes": list(ALLOWED_NODES),
            "available_modes": list(ALLOWED_MODES),
            **self._base_state(),
        }
        if self._floor_warning_active:
            observation.update(self._floor_instruction())
        self._emit(observation)
        self._events = [{"event": "reset", "observation": observation}]
        return observation

    def _floor_instruction(self) -> dict[str, Any]:
        return {
            "budget_exhausted": True,
            "required_next_action": "commit_or_abstain",
            "instruction": "Budget exhausted. Commit to an acquired view or abstain.",
        }

    def _emit(self, payload: dict[str, Any]) -> None:
        assert_public_payload(payload)
        canonical_bytes(payload)

    def _validate_coordinate(self, action: Action) -> tuple[int, int] | None:
        if action.nodes not in ALLOWED_NODES or action.modes not in ALLOWED_MODES:
            return None
        return int(action.nodes), int(action.modes)

    def _forced_abstention(self, trigger: str, last_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.done = True
        self.termination = "FORCED_ABSTENTION"
        observation: dict[str, Any] = {
            "schema_version": "evidence_view_benchmark.observation.v1",
            "event": "episode_terminal",
            "termination": self.termination,
            "trigger": trigger,
            "cost_charged": 0,
            **self._base_state(),
        }
        if last_result is not None:
            observation["last_action_result"] = dict(last_result)
        return observation

    def _acquire(self, action: Action) -> dict[str, Any]:
        coordinate = self._validate_coordinate(action)
        if coordinate is None:
            return {
                "schema_version": "evidence_view_benchmark.observation.v1",
                "event": "action_error",
                "error": "unsupported_view",
                "cost_charged": 0,
                **self._base_state(),
            }
        if coordinate in self._acquired:
            return {
                "schema_version": "evidence_view_benchmark.observation.v1",
                "event": "view_reused",
                **self._acquired[coordinate],
                "cost_charged": 0,
                **self._base_state(),
            }
        cost = coordinate[0]
        if cost > self.remaining:
            return {
                "schema_version": "evidence_view_benchmark.observation.v1",
                "event": "action_error",
                "error": "insufficient_budget",
                "requested_cost": cost,
                "cost_charged": 0,
                **self._base_state(),
            }
        self.spent += cost
        self.remaining -= cost
        view = dict(self._views[coordinate])
        self._acquired[coordinate] = view
        observation = {
            "schema_version": "evidence_view_benchmark.observation.v1",
            "event": "view_acquired",
            **view,
            "cost_charged": cost,
            **self._base_state(),
        }
        if self.remaining < MINIMUM_VIEW_COST:
            self._floor_warning_active = True
            observation.update(self._floor_instruction())
        return observation

    def _commit(self, action: Action) -> dict[str, Any]:
        coordinate = self._validate_coordinate(action)
        if coordinate is None or coordinate not in self._acquired:
            self._invalid_commit_attempts += 1
            if self._invalid_commit_attempts >= 2:
                return self._forced_abstention("unacquired_commit_retry_exhausted")
            return {
                "schema_version": "evidence_view_benchmark.observation.v1",
                "event": "action_error",
                "error": "view_not_acquired",
                "retry_remaining": 1,
                "cost_charged": 0,
                **self._base_state(),
            }
        self.done = True
        self.termination = "COMMIT"
        return {
            "schema_version": "evidence_view_benchmark.observation.v1",
            "event": "episode_terminal",
            "termination": self.termination,
            "committed_view": dict(self._acquired[coordinate]),
            "cost_charged": 0,
            **self._base_state(),
        }

    def _voluntary_abstention(self, action: Action) -> dict[str, Any]:
        self.done = True
        self.termination = "VOLUNTARY_ABSTENTION"
        return {
            "schema_version": "evidence_view_benchmark.observation.v1",
            "event": "episode_terminal",
            "termination": self.termination,
            "reason_recorded_in_action_trace": bool(action.reason),
            "cost_charged": 0,
            **self._base_state(),
        }

    def step(self, action: Action) -> tuple[dict[str, Any], bool]:
        if not self._events:
            raise RuntimeError("reset must be called before step")
        if self.done:
            observation = {
                "schema_version": "evidence_view_benchmark.observation.v1",
                "event": "action_error",
                "error": "episode_done",
                "cost_charged": 0,
                **self._base_state(),
            }
            self._emit(observation)
            return observation, True
        if not isinstance(action, Action):
            raise TypeError("step expects an Action")

        self.tool_calls += 1
        if self._floor_warning_active and action.kind not in {"commit", "abstain"}:
            if action.kind == "acquire_view":
                attempted = self._acquire(action)
            else:
                attempted = {
                    "schema_version": "evidence_view_benchmark.observation.v1",
                    "event": "action_error",
                    "error": "unknown_action",
                    "cost_charged": 0,
                    **self._base_state(),
                }
            observation = self._forced_abstention("budget_floor", attempted)
        elif action.kind == "acquire_view":
            observation = self._acquire(action)
        elif action.kind == "commit":
            observation = self._commit(action)
        elif action.kind == "abstain":
            observation = self._voluntary_abstention(action)
        else:
            observation = {
                "schema_version": "evidence_view_benchmark.observation.v1",
                "event": "action_error",
                "error": "unknown_action",
                "cost_charged": 0,
                **self._base_state(),
            }

        if self.tool_calls >= MAX_TOOL_CALLS and not self.done:
            observation = self._forced_abstention("tool_call_cap", observation)
        self._emit(observation)
        self._events.append(
            {
                "event": "step",
                "action": asdict(action),
                "observation": observation,
                "done": self.done,
            }
        )
        return observation, self.done

    def export_trace(self) -> dict[str, Any]:
        trace = {
            "schema_version": "evidence_view_benchmark.trace.v1",
            "case_id": self.case["case_id"],
            "public_case_sha256": self._case_binding_sha256,
            "acquisition_bank_file_sha256": self._bank_file_sha256,
            "case_registry_file_sha256": self._case_registry_file_sha256,
            "budget": self.budget,
            "visibility": self.visibility,
            "producer": self.producer,
            "events": json.loads(canonical_bytes(self._events)),
            "spent": self.spent,
            "remaining": self.remaining,
            "tool_calls": self.tool_calls,
            "done": self.done,
            "termination": self.termination,
        }
        # Selector-supplied action text is retained verbatim in the trace but is never
        # re-emitted as an observation. Leakage checks apply to environment-produced
        # observations, not to a method echoing ordinary words such as "reference".
        for event in trace["events"]:
            assert_public_payload(event["observation"])
        trace["trace_sha256"] = digest(trace)
        return trace

    @classmethod
    def replay_trace(
        cls,
        case: Mapping[str, Any],
        trace: Mapping[str, Any],
        *,
        acquisition_bank_path: str | Path | None = None,
        case_registry_path: str | Path | None = None,
    ) -> dict[str, Any]:
        supplied = json.loads(canonical_bytes(dict(trace)))
        body = dict(supplied)
        claimed = body.pop("trace_sha256", None)
        if claimed != digest(body):
            raise ValueError("trace content hash mismatch")
        if supplied.get("schema_version") != "evidence_view_benchmark.trace.v1":
            raise ValueError("unsupported trace schema")
        if supplied.get("case_id") != case.get("case_id"):
            raise ValueError("trace case ID mismatch")
        case_schema = case.get("schema_version")
        if case_schema == "evidence_view_benchmark.public_case.v1":
            case_binding = case.get("case_sha256")
        elif case_schema == "evidence_view_benchmark.selection_case.v1":
            case_binding = case.get("parent_public_case_sha256")
        else:
            raise ValueError("unsupported public case schema")
        if supplied.get("public_case_sha256") != case_binding:
            raise ValueError("trace case hash mismatch")
        replay = cls(
            case,
            budget=int(supplied["budget"]),
            visibility=str(supplied["visibility"]),
            producer=str(supplied["producer"]),
            acquisition_bank_path=acquisition_bank_path,
            case_registry_path=case_registry_path,
        )
        if replay._bank_file_sha256 != supplied["acquisition_bank_file_sha256"]:
            raise ValueError("trace acquisition bank hash mismatch")
        if replay._case_registry_file_sha256 != supplied["case_registry_file_sha256"]:
            raise ValueError("trace case registry hash mismatch")
        replay.reset()
        for event in supplied["events"]:
            if event["event"] == "reset":
                continue
            replay.step(Action(**event["action"]))
        regenerated = replay.export_trace()
        if canonical_bytes(regenerated) != canonical_bytes(supplied):
            raise ValueError("trace does not replay byte-identically")
        return regenerated
