#!/usr/bin/env python3
"""Fail-closed credential, PII, path, cache, and license-status scan."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".json", ".jsonl", ".yml", ".yaml"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "dist", "build", ".git"}

PATTERNS = {
    "anthropic_secret": re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),
    "openai_secret": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "huggingface_secret": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "github_secret": re.compile(r"(?:github_pat_|ghp_)[A-Za-z0-9_]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "workstation_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "email_address": re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"),
}


def main() -> None:
    errors: list[str] = []
    scanned = 0
    cache_artifacts: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            if path.is_file():
                cache_artifacts.append(str(relative))
            continue
        if not path.is_file() or path.name == "MANIFEST.sha256" or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="strict")
        for name, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                errors.append(f"{name}: {relative}: {match.group(0)[:80]}")

    if cache_artifacts:
        errors.append(f"generated/cache artifacts present: {sorted(cache_artifacts)[:20]}")

    license_map = json.loads((ROOT / "FILE_LICENSES.json").read_text(encoding="utf-8"))
    if license_map.get("public_release_authorized") is not False:
        errors.append("FILE_LICENSES.json must keep public_release_authorized=false")
    if license_map.get("default_status") != "internal_review_no_rights_granted":
        errors.append("unexpected default file license status")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "No license" not in license_text or "internal review only" not in license_text:
        errors.append("root LICENSE does not fail closed")
    if (ROOT / "CITATION.cff").exists():
        errors.append("CITATION.cff exists before citation approval")

    report = {
        "valid": not errors,
        "scanned_text_files": scanned,
        "secret_and_pii_pattern_count": len(PATTERNS),
        "cache_artifact_count": len(cache_artifacts),
        "public_release_authorized": False,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
