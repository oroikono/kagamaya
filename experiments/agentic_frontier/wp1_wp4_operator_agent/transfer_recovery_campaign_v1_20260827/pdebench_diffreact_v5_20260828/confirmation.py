"""One-shard prospective PDEBench confirmation runner."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import contract, freeze
from .evidence import build_from_projection, project
from .pipeline import Diagnostics, correct, fit, read_source, runtime_info, terminal, write_json
from .policies import evaluate_all, public_objective


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream_v4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_upstream() -> contract.Thresholds:
    expected = {
        "PREOUTCOME_FREEZE.json": contract.UPSTREAM_V4_FREEZE_SHA256,
        "development_thresholds.json": contract.UPSTREAM_DEVELOPMENT_SHA256,
        "authority_audit.json": contract.UPSTREAM_AUDIT_SHA256,
    }
    for name, digest in expected.items():
        path = UPSTREAM / name
        if not path.is_file() or sha256(path) != digest:
            raise SystemExit(f"upstream v4 binding mismatch: {name}")
    development = json.loads((UPSTREAM / "development_thresholds.json").read_text())
    audit = json.loads((UPSTREAM / "authority_audit.json").read_text())
    upstream_freeze = json.loads((UPSTREAM / "PREOUTCOME_FREEZE.json").read_text())
    if upstream_freeze.get("contract_sha256") != contract.UPSTREAM_V4_CONTRACT_SHA256:
        raise SystemExit("upstream v4 freeze contract mismatch")
    if development.get("contract_sha256") != contract.UPSTREAM_V4_CONTRACT_SHA256:
        raise SystemExit("upstream development contract mismatch")
    if development.get("status") != "DEVELOPMENT_AUTHORITY_FEASIBLE":
        raise SystemExit("upstream development did not authorize an authority")
    if audit.get("contract_sha256") != contract.UPSTREAM_V4_CONTRACT_SHA256:
        raise SystemExit("upstream audit contract mismatch")
    if audit.get("status") != "AUTHORITY_AUDIT_PASSED" or not audit.get("all_authority_gates_passed"):
        raise SystemExit("upstream audit did not pass")
    if tuple(audit.get("source_ids", ())) != contract.AUDIT_IDS:
        raise SystemExit("upstream audit source IDs mismatch")
    if development.get("selected_thresholds") != contract.SELECTED_THRESHOLDS:
        raise SystemExit("selected threshold binding mismatch")
    if audit.get("selected_thresholds") != contract.SELECTED_THRESHOLDS:
        raise SystemExit("audited threshold binding mismatch")
    return contract.Thresholds(**contract.SELECTED_THRESHOLDS)


def validate_source_receipt(path: Path, data: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    active = freeze.verify()
    expected = {
        "schema": contract.SCHEMA,
        "stage": "source_preflight",
        "contract_sha256": contract.CONTRACT_SHA256,
        "source_closure_sha256": active["source_closure_sha256"],
        "source_filename": contract.FILENAME,
        "source_bytes": contract.FILE_BYTES,
        "source_md5": contract.FILE_MD5,
        "upstream_audit_sha256": contract.UPSTREAM_AUDIT_SHA256,
        "confirmation_remained_sealed": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise SystemExit(f"source preflight receipt mismatch: {key}")
    if data.name != contract.FILENAME or data.stat().st_size != contract.FILE_BYTES:
        raise SystemExit("source identity changed after preflight")
    return receipt


def encode_float(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    return value


def encode_diagnostics(value: Diagnostics) -> dict[str, Any]:
    raw = asdict(value)
    raw["support"] = list(value.support)
    for key in ("residual", "gap", "condition", "minimum"):
        raw[key] = encode_float(raw[key])
    return raw


def classify(status: str, support: tuple[str, ...], contract_id: str) -> str:
    if status == "INCONCLUSIVE":
        return "inconclusive"
    if correct(status, support, contract_id):
        return "correct"
    if status == "IDENTIFIED":
        return "false_identification"
    if status == "OUTSIDE_LIBRARY":
        return "false_omission"
    raise ValueError(f"unknown status {status}")


def build_shard(data_path: Path, task_index: int, source_receipt: Path) -> dict[str, Any]:
    contract.validate()
    active = freeze.verify()
    thresholds = validate_upstream()
    source_receipt_value = validate_source_receipt(source_receipt, data_path)
    if task_index < 0 or task_index >= len(contract.SEALED_CONFIRMATION_IDS):
        raise SystemExit("task index outside frozen 0..59 range")
    source_id = contract.SEALED_CONFIRMATION_IDS[task_index]
    import h5py

    episodes = []
    with h5py.File(data_path, "r") as handle:
        data, x, y, times = read_source(handle, source_id)
        for noise in contract.NOISE_LEVELS:
            bundle = project(
                source_id=source_id,
                data=data,
                x=x,
                y=y,
                times=times,
                noise_level=noise,
            )
            for contract_id, (target_name, atoms, _) in contract.CONTRACTS.items():
                rows = []
                for temporal_samples in contract.TEMPORAL_SAMPLES:
                    for test_modes in contract.TEST_MODE_COUNTS:
                        evidence = build_from_projection(
                            source_id=source_id,
                            bundle=bundle,
                            target_name=target_name,
                            allowed_atoms=atoms,
                            temporal_samples=temporal_samples,
                            test_modes=test_modes,
                            noise_level=noise,
                        )
                        diagnostics = fit(evidence)
                        status, support = terminal(diagnostics, thresholds)
                        rows.append(
                            {
                                "view_id": evidence.view_id,
                                "temporal_samples": temporal_samples,
                                "test_modes": test_modes,
                                "cost": evidence.cost,
                                "evidence_sha256": evidence.digest,
                                "diagnostics": encode_diagnostics(diagnostics),
                                "status": status,
                                "support": list(support),
                                "public_objective": public_objective(
                                    diagnostics, status, evidence.cost, thresholds
                                ),
                            }
                        )
                reference_evidence = build_from_projection(
                    source_id=source_id,
                    bundle=bundle,
                    target_name=target_name,
                    allowed_atoms=atoms,
                    temporal_samples=contract.REFERENCE_TEMPORAL_SAMPLES,
                    test_modes=contract.REFERENCE_TEST_MODE_COUNT,
                    noise_level=noise,
                )
                reference_diagnostics = fit(reference_evidence)
                reference_status, reference_support = terminal(reference_diagnostics, thresholds)
                proposals = evaluate_all(rows)
                for proposal in proposals:
                    proposal["outcome"] = classify(
                        str(proposal["status"]),
                        tuple(proposal.get("support") or ()),
                        contract_id,
                    )
                episodes.append(
                    {
                        "episode_id": f"PDB-C-{source_id:04d}-{contract_id}-{noise:.3g}",
                        "source_id": source_id,
                        "noise": noise,
                        "contract_id": contract_id,
                        "target_name": target_name,
                        "truth_kind": contract.CONTRACTS[contract_id][2],
                        "views": rows,
                        "reference": {
                            "view_id": reference_evidence.view_id,
                            "status": reference_status,
                            "support": list(reference_support),
                            "outcome": classify(reference_status, reference_support, contract_id),
                            "evidence_sha256": reference_evidence.digest,
                            "diagnostics": encode_diagnostics(reference_diagnostics),
                        },
                        "proposals": proposals,
                    }
                )
    if len(episodes) != len(contract.NOISE_LEVELS) * len(contract.CONTRACTS):
        raise AssertionError("shard episode denominator drift")
    return {
        "schema": contract.SCHEMA,
        "stage": "confirmation_shard",
        "contract_sha256": contract.CONTRACT_SHA256,
        "source_closure_sha256": active["source_closure_sha256"],
        "upstream_development_sha256": contract.UPSTREAM_DEVELOPMENT_SHA256,
        "upstream_audit_sha256": contract.UPSTREAM_AUDIT_SHA256,
        "source_preflight_content": source_receipt_value,
        "runtime": runtime_info(),
        "task_index": task_index,
        "source_id": source_id,
        "episodes": episodes,
        "confirmation_ids_opened": [source_id],
        "development_or_audit_ids_opened": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_shard(args.data, args.task_index, args.source_receipt)
    write_json(args.output, value)
    print(
        json.dumps(
            {
                "status": "CONFIRMATION_SHARD_COMPLETE",
                "task_index": value["task_index"],
                "source_id": value["source_id"],
                "episodes": len(value["episodes"]),
                "output_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
