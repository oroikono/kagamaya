"""Small, dependency-free loaders for the public development files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "resources"


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


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            records.append(row)
    return records


def _index(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate case_id: {case_id}")
        result[case_id] = record
    return result


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(path or DATA_ROOT / "dev_cases.jsonl")


def load_observations(path: str | Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(path or DATA_ROOT / "dev_observations.jsonl")


def load_labels(path: str | Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(path or DATA_ROOT / "dev_labels.jsonl")


def load_view_bank(path: str | Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(path or DATA_ROOT / "view_bank.jsonl")


def case_bundle(case_id: str) -> dict[str, dict[str, Any]]:
    cases = _index(load_cases())
    observations = _index(load_observations())
    labels = _index(load_labels())
    views = _index(load_view_bank())
    if case_id not in cases or case_id not in observations or case_id not in labels or case_id not in views:
        raise KeyError(case_id)
    return {
        "case": cases[case_id],
        "observation": observations[case_id],
        "label": labels[case_id],
        "views": views[case_id],
    }
