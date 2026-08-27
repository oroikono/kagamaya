"""Content-addressed authorization for each sealed scientific stage.

The receipt is an internal execution interlock.  It binds one stage to the
one-time source freeze and, where applicable, to the exact upstream aggregate.
It is not an external signature or a claim that the result will be positive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parent
STAGES = (
    "internal_mechanism",
    "internal_confirmation",
    "external_validation",
    "external_confirmation",
)


def digest(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class StageAuthorization:
    stage: str
    freeze_digest: str
    input_bindings: tuple[tuple[str, str], ...]
    authorization_digest: str


def build_payload(stage: str, input_bindings: Mapping[str, str],
                  root_review_receipt: str) -> dict[str, object]:
    from .freeze import verify

    if stage not in STAGES:
        raise ValueError("unknown stage")
    if not root_review_receipt.strip():
        raise ValueError("root review receipt is required")
    freeze = verify()
    if not freeze["valid"]:
        raise RuntimeError("source freeze does not validate")
    bindings = {str(key): str(value) for key, value in input_bindings.items()}
    if any(not _is_sha256(value) for value in bindings.values()):
        raise ValueError("every input binding must be a SHA-256 digest")
    body: dict[str, object] = {
        "schema_version": "maintrack.spotlight-stage-authorization.v1",
        "authority": "root",
        "decision": "AUTHORIZE_EXACT_STAGE",
        "stage": stage,
        "freeze_digest": freeze["freeze_digest"],
        "input_bindings": bindings,
        "root_review_receipt": root_review_receipt,
        "single_stage_only": True,
        "post_outcome_tuning_permitted": False,
        "external_signature": False,
        "claim_ceiling": "internal execution interlock; not external witnessing",
    }
    return {**body, "authorization_digest": digest(body)}


def filename(payload: Mapping[str, object]) -> str:
    stage = str(payload.get("stage"))
    claimed = str(payload.get("authorization_digest"))
    if stage not in STAGES or not _is_sha256(claimed):
        raise ValueError("invalid authorization naming fields")
    return f"root_authorization_{stage}_{claimed[:16]}.json"


def validate(path: Path, *, stage: str,
             input_bindings: Mapping[str, str]) -> StageAuthorization:
    from .freeze import verify

    freeze = verify()
    if not freeze["valid"]:
        raise RuntimeError("source freeze does not validate")
    if stage not in STAGES or not path.is_file():
        raise RuntimeError(f"authorization is absent for {stage}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("authorization root must be an object")
    claimed = value.pop("authorization_digest", None)
    if not isinstance(claimed, str) or digest(value) != claimed:
        raise RuntimeError("authorization content digest mismatch")
    value["authorization_digest"] = claimed
    if path.name != filename(value):
        raise RuntimeError("authorization path is not content addressed")
    bindings = {str(key): str(item) for key, item in input_bindings.items()}
    expected = {
        "schema_version": "maintrack.spotlight-stage-authorization.v1",
        "authority": "root",
        "decision": "AUTHORIZE_EXACT_STAGE",
        "stage": stage,
        "freeze_digest": freeze["freeze_digest"],
        "input_bindings": bindings,
        "single_stage_only": True,
        "post_outcome_tuning_permitted": False,
        "external_signature": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"authorization mismatch: {key}")
    if any(not _is_sha256(item) for item in bindings.values()):
        raise RuntimeError("stage input binding is not a SHA-256 digest")
    if not str(value.get("root_review_receipt", "")).strip():
        raise RuntimeError("authorization lacks root review receipt")
    return StageAuthorization(
        stage=stage,
        freeze_digest=str(freeze["freeze_digest"]),
        input_bindings=tuple(sorted(bindings.items())),
        authorization_digest=claimed,
    )


def publish(stage: str, input_bindings: Mapping[str, str],
            root_review_receipt: str) -> Path:
    payload = build_payload(stage, input_bindings, root_review_receipt)
    path = ROOT / filename(payload)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path

