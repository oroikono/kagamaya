"""Compile the frozen operator fragment through the real SUSANO public API."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .serialization import content_digest, file_digest


@dataclass(frozen=True)
class OperatorCandidate:
    candidate_id: str
    display_name: str
    terms: tuple[str, ...]
    source_expression: str
    canonical_string: str
    structural_tokens: tuple[str, ...]
    susano_attributes: dict[str, Any]
    representation_digest: str

    @property
    def complexity(self) -> int:
        return len(self.terms)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terms"] = list(self.terms)
        payload["structural_tokens"] = list(self.structural_tokens)
        payload["complexity"] = self.complexity
        return payload


_DECLARATIONS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("reaction", "Reaction", ("u",), "dt(u) - 1.0*u"),
    ("diffusion", "Diffusion", ("uxx",), "dt(u) - 1.0*dxx(u)"),
    (
        "reaction_diffusion",
        "Reaction–diffusion",
        ("u", "uxx"),
        "dt(u) - 1.0*u - 1.0*dxx(u)",
    ),
    (
        "advection_diffusion",
        "Advection–diffusion",
        ("ux", "uxx"),
        "dt(u) - 1.0*dx(u) - 1.0*dxx(u)",
    ),
    (
        "burgers_diffusion",
        "Burgers–diffusion",
        ("uux", "uxx"),
        "dt(u) - 1.0*u*dx(u) - 1.0*dxx(u)",
    ),
    ("dispersion", "Dispersion", ("uxxx",), "dt(u) - 1.0*dxxx(u)"),
)


def _load_susano() -> tuple[Any, str, Path]:
    try:
        import susano
        from susano import compile_pde
        from susano import representation as representation_module
    except ImportError as exc:
        raise RuntimeError(
            "The WP1+WP4 lane requires the real SUSANO package. Install "
            "`/Users/oroikon/Downloads/susano-clean-private` in editable mode "
            "or prepend its `src` directory to PYTHONPATH."
        ) from exc
    return compile_pde, str(susano.__version__), Path(representation_module.__file__).resolve()


def _git_commit(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(completed.stdout.strip())
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_candidate_manifest(
    candidate_ids: Iterable[str] | None = None,
) -> tuple[tuple[OperatorCandidate, ...], dict[str, Any]]:
    compile_pde, version, module_path = _load_susano()
    requested = set(candidate_ids) if candidate_ids is not None else None
    known = {row[0] for row in _DECLARATIONS}
    if requested is not None and not requested <= known:
        raise ValueError(f"unknown candidate IDs: {sorted(requested - known)}")

    candidates: list[OperatorCandidate] = []
    for candidate_id, display_name, terms, expression in _DECLARATIONS:
        if requested is not None and candidate_id not in requested:
            continue
        representation = compile_pde(expression)
        attributes = asdict(representation.attributes)
        identity_payload = {
            "canonical_string": representation.canonical_string,
            "structural_tokens": list(representation.structural_tokens),
            "attributes": attributes,
            "terms": list(terms),
        }
        candidates.append(
            OperatorCandidate(
                candidate_id=candidate_id,
                display_name=display_name,
                terms=terms,
                source_expression=expression,
                canonical_string=representation.canonical_string,
                structural_tokens=tuple(representation.structural_tokens),
                susano_attributes=attributes,
                representation_digest=content_digest(identity_payload),
            )
        )

    candidates.sort(key=lambda item: item.candidate_id)
    runtime = {
        "susano_version": version,
        "susano_git_commit": _git_commit(module_path)
        or os.environ.get("SUSANO_GIT_COMMIT"),
        "representation_module_sha256": file_digest(module_path),
    }
    body = {
        "schema_version": "verisigs.wp1_wp4.susano_manifest.v1",
        "candidates": [candidate.to_dict() for candidate in candidates],
        "runtime": runtime,
    }
    body["manifest_sha256"] = content_digest(body)
    return tuple(candidates), body


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
