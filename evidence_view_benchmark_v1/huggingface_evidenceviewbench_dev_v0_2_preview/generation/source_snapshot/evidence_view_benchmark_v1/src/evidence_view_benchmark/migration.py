"""Typed, invertible boundary adapter for legacy benchmark artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TokenMigration:
    authority_forward: Mapping[str, str]
    execution_forward: Mapping[str, str]
    fields_forward: Mapping[str, str]
    aliases_forward: Mapping[str, str]

    @staticmethod
    def _inverse(mapping: Mapping[str, str], label: str) -> dict[str, str]:
        inverse = {value: key for key, value in mapping.items()}
        if len(inverse) != len(mapping):
            raise ValueError(f"{label} is not invertible")
        return inverse

    @property
    def authority_reverse(self) -> Mapping[str, str]:
        return self._inverse(self.authority_forward, "authority mapping")

    @property
    def execution_reverse(self) -> Mapping[str, str]:
        return self._inverse(self.execution_forward, "execution mapping")

    @property
    def fields_reverse(self) -> Mapping[str, str]:
        return self._inverse(self.fields_forward, "field mapping")

    @property
    def aliases_reverse(self) -> Mapping[str, str]:
        return self._inverse(self.aliases_forward, "alias mapping")

    @property
    def current_authority_statuses(self) -> frozenset[str]:
        return frozenset(self.authority_forward.values())

    @property
    def current_terminal_statuses(self) -> frozenset[str]:
        return frozenset({"IDENTIFIED", "OUTSIDE_LIBRARY"})

    def migrate_status(self, status: str, *, namespace: str = "authority") -> str:
        mapping = {
            "authority": self.authority_forward,
            "execution": self.execution_forward,
        }.get(namespace)
        if mapping is None:
            raise ValueError(f"unknown status namespace: {namespace}")
        if status not in mapping:
            raise ValueError(f"unknown legacy {namespace} status: {status}")
        return mapping[status]

    def _walk(self, value: Any, *, reverse: bool) -> Any:
        fields = self.fields_reverse if reverse else self.fields_forward
        strings: dict[str, str] = {}
        for mapping in (
            self.authority_reverse if reverse else self.authority_forward,
            self.execution_reverse if reverse else self.execution_forward,
            self.aliases_reverse if reverse else self.aliases_forward,
        ):
            for source, target in mapping.items():
                if source in strings and strings[source] != target:
                    raise ValueError(f"ambiguous string mapping: {source}")
                strings[source] = target
        if isinstance(value, Mapping):
            output: dict[Any, Any] = {}
            for key, nested in value.items():
                mapped_key = fields.get(key, key) if isinstance(key, str) else key
                if mapped_key in output:
                    raise ValueError(f"field migration collision: {mapped_key}")
                output[mapped_key] = self._walk(nested, reverse=reverse)
            return output
        if isinstance(value, list):
            return [self._walk(item, reverse=reverse) for item in value]
        if isinstance(value, tuple):
            return tuple(self._walk(item, reverse=reverse) for item in value)
        if isinstance(value, str):
            return strings.get(value, value)
        return value

    def migrate(self, value: Any) -> Any:
        return self._walk(value, reverse=False)

    def restore_legacy(self, value: Any) -> Any:
        return self._walk(value, reverse=True)


def load_migration(schema_path: Path) -> TokenMigration:
    with schema_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "evidence_view_benchmark.token_migration.v1":
        raise ValueError("unsupported migration schema")
    migration = TokenMigration(
        authority_forward=dict(payload["authority_status"]["legacy_to_current"]),
        execution_forward=dict(payload["execution_status"]["legacy_to_current"]),
        fields_forward=dict(payload["field_names"]["legacy_to_current"]),
        aliases_forward=dict(payload["value_aliases"]["legacy_to_current"]),
    )
    if set(migration.authority_forward) != {
        "IDENTIFIED",
        "OUT_OF_GRAMMAR",
        "UNIDENTIFIED",
    }:
        raise ValueError("authority migration domain is incomplete")
    if migration.current_authority_statuses != {
        "IDENTIFIED",
        "OUTSIDE_LIBRARY",
        "INCONCLUSIVE",
    }:
        raise ValueError("authority migration codomain is incorrect")
    migration.authority_reverse
    migration.execution_reverse
    migration.fields_reverse
    migration.aliases_reverse
    return migration

