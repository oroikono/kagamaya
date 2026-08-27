"""Verification-first WP1+WP4 operator-discovery prototype.

The package is intentionally isolated from the frozen ``hybrid`` experiment.
SUSANO and proposal policies describe/order a bounded operator fragment; only
the weak-form engine in this package can assign a scientific terminal status.
"""

from .engine import DiscoveryResult, discover_operators
from .manufactured import manufactured_reaction_diffusion
from .susano_candidates import build_candidate_manifest

__all__ = [
    "DiscoveryResult",
    "build_candidate_manifest",
    "discover_operators",
    "manufactured_reaction_diffusion",
]
