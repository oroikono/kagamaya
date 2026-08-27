"""EvidenceViewBench-Dev: reproducible numerical-view selection utilities."""

from .data import case_bundle, load_cases, load_labels, load_observations, load_view_bank
from .environment import Action, EvidenceViewEnv
from .scoring import score_trace
from .two_d import (
    TwoDAction,
    TwoDEvidenceEnv,
    TwoDScore,
    TwoDTerminal,
    load_two_d_owner,
    load_two_d_public,
    score_two_d_terminal,
    two_d_owner_row,
    two_d_public_row,
    two_d_view_manifest,
)

__all__ = [
    "Action",
    "EvidenceViewEnv",
    "case_bundle",
    "load_cases",
    "load_labels",
    "load_observations",
    "load_view_bank",
    "score_trace",
    "TwoDAction",
    "TwoDEvidenceEnv",
    "TwoDScore",
    "TwoDTerminal",
    "load_two_d_owner",
    "load_two_d_public",
    "score_two_d_terminal",
    "two_d_owner_row",
    "two_d_public_row",
    "two_d_view_manifest",
]

__version__ = "0.1.2"
