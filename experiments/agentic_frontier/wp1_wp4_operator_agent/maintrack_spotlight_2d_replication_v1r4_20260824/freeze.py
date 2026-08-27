"""Create once and verify the prospective source closure."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE_PATH = ROOT / "PREOUTCOME_FREEZE.json"
REPO = ROOT.parents[3]
DEPENDENCY_FILES = {
    "wp1_wp4_operator_agent.__init__": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/__init__.py",
    "wp1_wp4_operator_agent.config": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/config.py",
    "wp1_wp4_operator_agent.engine": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/engine.py",
    "wp1_wp4_operator_agent.manufactured": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/manufactured.py",
    "wp1_wp4_operator_agent.proposals": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/proposals.py",
    "wp1_wp4_operator_agent.run_demo": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/run_demo.py",
    "wp1_wp4_operator_agent.serialization": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/serialization.py",
    "wp1_wp4_operator_agent.susano_candidates": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/susano_candidates.py",
    "wp1_wp4_operator_agent.validate_artifacts": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/validate_artifacts.py",
    "wp1_wp4_operator_agent.weak_form": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/weak_form.py",
    "maintrack_2d_physical_v1.__init__": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/maintrack_2d_physical_v1/__init__.py",
    "adversarial_verifier_gaming_v1.__init__": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/__init__.py",
    "adversarial_verifier_gaming_v1.contract": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/contract.py",
    "exact_2d_oracle": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/maintrack_2d_physical_v1/exact_2d_oracle.py",
    "exact_2d_coupled_oracle": REPO / "experiments/agentic_frontier/"
    "wp1_wp4_operator_agent/maintrack_2d_physical_v1/exact_2d_coupled_oracle.py",
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files() -> tuple[str, ...]:
    files: list[str] = []
    for pattern in ("*.py", "*.md", "docs/*.md", "schemas/*.json", "tests/*.py",
                    "scripts/*.sh", "scripts/*.sbatch", "scripts/*.py",
                    "external_sources/*"):
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            if path.name == "PREOUTCOME_FREEZE.json":
                continue
            if path.name == "STATUS.md":
                continue
            files.append(path.relative_to(ROOT).as_posix())
    files.extend(("EXTERNAL_WELL_MANIFEST.json", "README.md", "DECISIONS.md"))
    return tuple(sorted(set(files)))


def manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in source_files():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = file_sha(path)
    return result


def dependency_manifest() -> dict[str, str]:
    return {name: file_sha(path) for name, path in DEPENDENCY_FILES.items()}


def content_digest(value: dict[str, str]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def create() -> dict[str, object]:
    if FREEZE_PATH.exists():
        raise FileExistsError("one-time pre-outcome freeze already exists")
    files = manifest()
    dependencies = dependency_manifest()
    body: dict[str, object] = {
        "schema_version": "maintrack.spotlight-2d-replication.freeze.v1r4",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_FROZEN_NO_SCIENTIFIC_OUTCOMES",
        "source_manifest": files,
        "source_closure_sha256": content_digest(files),
        "dependency_manifest": dependencies,
        "dependency_closure_sha256": content_digest(dependencies),
        "external_manifest_sha256": file_sha(ROOT / "EXTERNAL_WELL_MANIFEST.json"),
        "opened_scientific_splits": [],
        "sealed_scientific_splits": [
            "internal_mechanism_161000_161019",
            "internal_confirmation_162000_162099",
            "external_validation_publisher_valid",
            "external_confirmation_publisher_test",
        ],
        "same_attack_and_guard": True,
        "post_outcome_tuning_permitted": False,
        "claim_ceiling": (
            "prospective finite 2-D/coupled replication; external arm is an "
            "independent numerical origin, not physical validation"
        ),
    }
    freeze_digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    payload = {**body, "freeze_digest": freeze_digest}
    FREEZE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify() -> dict[str, object]:
    if not FREEZE_PATH.is_file():
        return {"valid": False, "errors": ["freeze_absent"]}
    payload = json.loads(FREEZE_PATH.read_text())
    value = dict(payload)
    claimed = value.pop("freeze_digest", None)
    expected = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    errors: list[str] = []
    if claimed != expected:
        errors.append("freeze_digest")
    recorded_manifest = payload.get("source_manifest")
    if not isinstance(recorded_manifest, dict):
        return {"valid": False, "errors": ["source_manifest_type"]}
    try:
        actual = {
            str(name): file_sha(ROOT / str(name))
            for name in recorded_manifest
        }
    except FileNotFoundError:
        return {"valid": False, "errors": ["source_file_missing"]}
    actual_dependencies = dependency_manifest()
    if payload.get("source_manifest") != actual:
        errors.append("source_manifest")
    if payload.get("source_closure_sha256") != content_digest(actual):
        errors.append("source_closure")
    if payload.get("dependency_manifest") != actual_dependencies:
        errors.append("dependency_manifest")
    if payload.get("dependency_closure_sha256") != content_digest(
        actual_dependencies
    ):
        errors.append("dependency_closure")
    if payload.get("external_manifest_sha256") != file_sha(
        ROOT / "EXTERNAL_WELL_MANIFEST.json"
    ):
        errors.append("external_manifest")
    return {"valid": not errors, "errors": errors,
            "freeze_digest": claimed,
            "source_closure_sha256": content_digest(actual)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "verify"))
    args = parser.parse_args()
    result = create() if args.command == "create" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "verify" and not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
