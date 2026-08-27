"""Build the frozen 24-case paired-producer public LLM campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .contract import LLM_RELATIVE_NOISES, MAX_LLM_CALLS, SCHEMA_PREFIX, digest
from .io_utils import read_json, write_json
from .llm_contract import build_request, prompt_sha256


def _validated(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    claimed = body.pop("campaign_sha256", None)
    if claimed != digest(body):
        raise ValueError("public campaign hash mismatch")


def build(spectral: Mapping[str, Any], fd: Mapping[str, Any]) -> dict[str, Any]:
    _validated(spectral)
    _validated(fd)
    if spectral["config"]["split"] != "confirmation" or fd["config"]["split"] != "confirmation":
        raise ValueError("LLM campaign requires the confirmation split")
    # The selector uses the frozen seed-position digits directly:
    # family*10000 + seed_position*100 + noise_index; seed position zero has
    # the two digits before noise_index equal to 00.
    spectral_rows = {
        row["episode_id"]: row
        for row in spectral["episodes"]
        if float(row["episode_metadata"]["relative_noise"]) in LLM_RELATIVE_NOISES
        and (int(row["episode_id"].rsplit("_", 1)[1]) % 10000) < 100
    }
    fd_rows = {
        row["episode_id"]: row
        for row in fd["episodes"]
        if float(row["episode_metadata"]["relative_noise"]) in LLM_RELATIVE_NOISES
        and (int(row["episode_id"].rsplit("_", 1)[1]) % 10000) < 100
    }
    if set(spectral_rows) != set(fd_rows) or len(spectral_rows) != MAX_LLM_CALLS:
        raise ValueError("paired producer LLM subset must contain exactly 24 cases")
    cases = []
    for index, episode_id in enumerate(sorted(spectral_rows)):
        spectral_episode, fd_episode = spectral_rows[episode_id], fd_rows[episode_id]
        request = build_request(spectral_episode["actions"], fd_episode["actions"])
        cases.append(
            {
                "case_id": f"case_{index:02d}",
                "request": request,
                "prompt_sha256": prompt_sha256(request),
                "bindings": {
                    "spectral_etdrk4": {
                        "episode_id": episode_id,
                        "public_episode_sha256": spectral_episode["episode_sha256"],
                    },
                    "fd_bdf": {
                        "episode_id": episode_id,
                        "public_episode_sha256": fd_episode["episode_sha256"],
                    },
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.llm_campaign",
        "source_public_campaigns": {
            "spectral_etdrk4": spectral["campaign_sha256"],
            "fd_bdf": fd["campaign_sha256"],
        },
        "call_count": len(cases),
        "cases": cases,
    }
    payload["campaign_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral-public", required=True, type=Path)
    parser.add_argument("--fd-public", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(read_json(args.spectral_public), read_json(args.fd_public))
    write_json(args.output, result)
    print(result["campaign_sha256"])


if __name__ == "__main__":
    main()
