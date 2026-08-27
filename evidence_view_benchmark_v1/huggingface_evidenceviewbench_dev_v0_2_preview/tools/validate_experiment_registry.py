#!/usr/bin/env python3
"""Dependency-free validation for EXPERIMENT_REGISTRY.json."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "EXPERIMENT_REGISTRY.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    if ROOT not in candidate.parents:
        raise ValueError(f"registry path escapes release root: {relative}")
    return candidate


def resolve_pointer(document: object, pointer: str) -> object:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer}")
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise KeyError(f"cannot descend through {type(value).__name__}: {pointer}")
    return value


def validate() -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    errors: list[str] = []
    if registry.get("schema_version") != "evidenceviewbench.experiment_registry.v1":
        errors.append("unexpected registry schema_version")
    if registry.get("release_version") != "0.1.2":
        errors.append("unexpected registry release_version")

    claims = registry.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("claims must be a non-empty list")
        claims = []
    ids = [claim.get("claim_id") for claim in claims]
    if len(ids) != len(set(ids)):
        errors.append("claim IDs are not unique")

    json_cache: dict[Path, object] = {}
    evidence_files: set[Path] = set()
    check_count = 0
    for claim in claims:
        claim_id = claim.get("claim_id", "<missing>")
        if not re.fullmatch(r"EVB-[0-9]{3}", str(claim_id)):
            errors.append(f"{claim_id}: malformed claim ID")
        for field in ("headline", "status", "split", "denominators", "source_hashes", "claim_ceiling"):
            if field not in claim:
                errors.append(f"{claim_id}: missing {field}")
        for name, value in claim.get("source_hashes", {}).items():
            if not isinstance(value, str) or not SHA256.fullmatch(value):
                errors.append(f"{claim_id}: invalid source hash {name}")
        for evidence in claim.get("evidence", []):
            try:
                path = safe_path(evidence["path"])
                expected = evidence["file_sha256"]
                if not SHA256.fullmatch(expected):
                    errors.append(f"{claim_id}: malformed evidence hash for {evidence['path']}")
                elif not path.is_file():
                    errors.append(f"{claim_id}: missing evidence {evidence['path']}")
                else:
                    evidence_files.add(path)
                    actual = digest(path)
                    if actual != expected:
                        errors.append(
                            f"{claim_id}: evidence hash mismatch for {evidence['path']}: {actual}"
                        )
            except (KeyError, ValueError) as exc:
                errors.append(f"{claim_id}: invalid evidence entry: {exc}")
        for check in claim.get("checks", []):
            check_count += 1
            try:
                path = safe_path(check["path"])
                if path not in json_cache:
                    json_cache[path] = json.loads(path.read_text(encoding="utf-8"))
                observed = resolve_pointer(json_cache[path], check["pointer"])
                if observed != check["equals"]:
                    errors.append(
                        f"{claim_id}: {check['path']}{check['pointer']} expected "
                        f"{check['equals']!r}, got {observed!r}"
                    )
            except (KeyError, ValueError, IndexError, json.JSONDecodeError) as exc:
                errors.append(f"{claim_id}: invalid semantic check: {exc}")

    return {
        "valid": not errors,
        "claim_count": len(claims),
        "evidence_file_count": len(evidence_files),
        "semantic_check_count": check_count,
        "errors": errors,
    }


def main() -> None:
    report = validate()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
