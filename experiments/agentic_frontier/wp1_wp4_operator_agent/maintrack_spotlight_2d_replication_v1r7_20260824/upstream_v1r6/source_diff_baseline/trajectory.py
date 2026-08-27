"""Shared serialization schema for separately coded numerical origins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    episode_id: str
    origin: str
    case_id: str
    seed: int
    target: str
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray | None
    forcing: np.ndarray
    convergence_error: float

    def validate(self) -> None:
        nt, nx, ny = self.u.shape
        if nx != len(self.x) or ny != len(self.y) or nt != len(self.times):
            raise ValueError("trajectory coordinate/data shape mismatch")
        if self.v is not None and self.v.shape != self.u.shape:
            raise ValueError("u/v shape mismatch")
        if self.forcing.shape != self.u.shape:
            raise ValueError("forcing shape mismatch")
        if not (np.isfinite(self.u).all() and np.isfinite(self.forcing).all()):
            raise ValueError("non-finite trajectory")
        if self.v is not None and not np.isfinite(self.v).all():
            raise ValueError("non-finite coupled trajectory")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly increasing")

    @property
    def public_digest(self) -> str:
        h = hashlib.sha256()
        for array in (self.times, self.x, self.y, self.u,
                      self.forcing if self.v is None else self.v,
                      self.forcing):
            contiguous = np.ascontiguousarray(array, dtype=np.float64)
            h.update(str(contiguous.shape).encode())
            h.update(contiguous.tobytes())
        h.update(self.origin.encode())
        h.update(self.episode_id.encode())
        return h.hexdigest()

