"""A06 — smooth-SAT penalty quench and box squeeze.

The energy is a squared hinge on penetration; ``tau`` smooths ``|.|`` and
``cos/sin`` so L-BFGS sees a C1 surface.  Continuation
``tau: 1e-3 -> 1e-5 -> 1e-8 -> 0`` converges to an exact (solid) layout.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from . import geom

TAUS = (1e-3, 1e-5, 1e-8, 0.0)


def _flat(sq: np.ndarray) -> np.ndarray:
    return np.concatenate([sq[:, 0], sq[:, 1], sq[:, 2]])


def _unflat(v: np.ndarray) -> np.ndarray:
    n = len(v) // 3
    return np.column_stack([v[:n], v[n : 2 * n], v[2 * n :]])


def quench(
    sq: np.ndarray,
    L: float,
    lam: float = 25.0,
    taus=TAUS,
    maxiter: int = 120,
    gtol: float = 1e-11,
) -> np.ndarray:
    """Minimise the penetration energy inside a box of side L."""
    sq = np.array(sq, dtype=np.float64, copy=True)
    n = len(sq)

    def fg(v):
        s = _unflat(v)
        x = np.ascontiguousarray(s[:, 0])
        y = np.ascontiguousarray(s[:, 1])
        a = np.ascontiguousarray(s[:, 2])
        E, gx, gy, ga = geom.energy_grad(x, y, a, L, lam, fg.tau)
        return E, np.concatenate([gx, gy, ga])

    fg.tau = 0.0
    for tau in taus:
        fg.tau = tau
        v0 = _flat(sq)
        res = minimize(
            fg,
            v0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxcor": 8, "gtol": gtol, "ftol": 1e-15},
        )
        sq = _unflat(res.x)
    return sq


def squeeze(
    sq: np.ndarray,
    rounds: int = 12,
    shrink: float = 2e-3,
    lam: float = 25.0,
    taus=TAUS,
    tol: float = 1e-9,
) -> np.ndarray:
    """Alternate quench and box shrink until no further progress."""
    sq = geom.normalise(sq)
    L = geom.side_of(
        np.ascontiguousarray(sq[:, 0]),
        np.ascontiguousarray(sq[:, 1]),
        np.ascontiguousarray(sq[:, 2]),
    )
    step = shrink
    for _ in range(rounds):
        target = L * (1.0 - step)
        cand = quench(sq, target, lam=lam, taus=taus, maxiter=80)
        if geom.pen_max(
            np.ascontiguousarray(cand[:, 0]),
            np.ascontiguousarray(cand[:, 1]),
            np.ascontiguousarray(cand[:, 2]),
        ) <= tol:
            cand = geom.normalise(cand)
            Lc = geom.side_of(
                np.ascontiguousarray(cand[:, 0]),
                np.ascontiguousarray(cand[:, 1]),
                np.ascontiguousarray(cand[:, 2]),
            )
            if Lc < L - 1e-12:
                sq, L = cand, Lc
                step = min(step * 1.5, 1e-1)
                continue
        step *= 0.5
        if step < 1e-7:
            break
    return sq


def grid_layout(n: int) -> np.ndarray:
    """Trivial integer grid at side ceil(sqrt(n))."""
    k = math.ceil(math.sqrt(n))
    sq = np.zeros((n, 3))
    for i in range(n):
        sq[i, 0] = (i % k) + 0.5
        sq[i, 1] = (i // k) + 0.5
    return sq


def side(sq: np.ndarray) -> float:
    return geom.side_of(
        np.ascontiguousarray(sq[:, 0]),
        np.ascontiguousarray(sq[:, 1]),
        np.ascontiguousarray(sq[:, 2]),
    )
