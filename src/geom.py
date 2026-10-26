"""Geometry, penetration, energy and gradient kernels (CPU / numba).

Conventions
-----------
* A layout is an ``(n, 3)`` float64 array: columns ``x, y, theta``.
* ``theta`` is in RADIANS.
* The container side ``L`` is derived from the layout (``side_of``).

Penetration
-----------
For unit squares ``i, j`` with centre difference ``d`` and angle difference
``delta``, the projections on a unit axis ``a`` are intervals of half width
``0.5*|a.u_i| + 0.5*|a.u_j|``, whose sum is
``R(delta) = 0.5*(1 + |cos delta| + |sin delta|)``.  The pair is separated iff
``max over the four edge normals of |d.a| >= R``; therefore

    pen = R(delta) - max_{a in axes} |d.a|        (> 0 means overlap)

The gradient includes the rotation of the winning axis, which is essential:
dropping it gives a ~50% wrong angle derivative (verified against finite
differences in ``tests/test_geom.py``).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

__all__ = [
    "side_of",
    "half_extent",
    "pen_max",
    "pen_matrix",
    "energy",
    "energy_grad",
    "repair_scale",
    "normalise",
    "solid",
]


@njit(cache=True, fastmath=True, inline="always")
def _sabs(z: float, tau: float) -> float:
    if tau == 0.0:
        return abs(z)
    return math.sqrt(z * z + tau * tau)


@njit(cache=True, fastmath=True)
def half_extent(theta: np.ndarray) -> np.ndarray:
    out = np.empty(theta.shape[0], dtype=np.float64)
    for i in range(theta.shape[0]):
        out[i] = 0.5 * (abs(math.cos(theta[i])) + abs(math.sin(theta[i])))
    return out


@njit(cache=True, fastmath=True)
def side_of(x: np.ndarray, y: np.ndarray, a: np.ndarray) -> float:
    """Axis-aligned bounding-box side of the corner cloud (exact, O(n))."""
    n = x.shape[0]
    if n == 0:
        return 0.0
    xmin = x[0]
    xmax = x[0]
    ymin = y[0]
    ymax = y[0]
    for i in range(n):
        c = math.cos(a[i])
        s = math.sin(a[i])
        h = 0.5 * (abs(c) + abs(s))
        lo = x[i] - h
        hi = x[i] + h
        if lo < xmin:
            xmin = lo
        if hi > xmax:
            xmax = hi
        lo = y[i] - h
        hi = y[i] + h
        if lo < ymin:
            ymin = lo
        if hi > ymax:
            ymax = hi
    dx = xmax - xmin
    dy = ymax - ymin
    return dx if dx > dy else dy


@njit(cache=True, fastmath=True)
def _prep(a: np.ndarray):
    n = a.shape[0]
    c = np.empty(n, dtype=np.float64)
    s = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)
    dh = np.empty(n, dtype=np.float64)
    for i in range(n):
        ci = math.cos(a[i])
        si = math.sin(a[i])
        c[i] = ci
        s[i] = si
        h[i] = 0.5 * (abs(ci) + abs(si))
        dh[i] = 0.5 * (
            -si * (1.0 if ci >= 0.0 else -1.0) + ci * (1.0 if si >= 0.0 else -1.0)
        )
    return c, s, h, dh


@njit(cache=True, fastmath=True)
def pen_max(x: np.ndarray, y: np.ndarray, a: np.ndarray) -> float:
    """Largest SAT penetration depth; > 0 means the layout overlaps."""
    c, s, _, _ = _prep(a)
    n = x.shape[0]
    worst = 0.0
    for i in range(n - 1):
        ci = c[i]
        si = s[i]
        for j in range(i + 1, n):
            cj = c[j]
            sj = s[j]
            d = a[i] - a[j]
            R = 0.5 * (1.0 + abs(math.cos(d)) + abs(math.sin(d)))
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            p = abs(dx * ci + dy * si)
            q = abs(-dx * si + dy * ci)
            if q > p:
                p = q
            q = abs(dx * cj + dy * sj)
            if q > p:
                p = q
            q = abs(-dx * sj + dy * cj)
            if q > p:
                p = q
            pen = R - p
            if pen > worst:
                worst = pen
    return worst


@njit(cache=True, fastmath=True)
def pen_matrix(x: np.ndarray, y: np.ndarray, a: np.ndarray) -> np.ndarray:
    c, s, _, _ = _prep(a)
    n = x.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        ci = c[i]
        si = s[i]
        for j in range(i + 1, n):
            cj = c[j]
            sj = s[j]
            d = a[i] - a[j]
            R = 0.5 * (1.0 + abs(math.cos(d)) + abs(math.sin(d)))
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            p = abs(dx * ci + dy * si)
            q = abs(-dx * si + dy * ci)
            if q > p:
                p = q
            q = abs(dx * cj + dy * sj)
            if q > p:
                p = q
            q = abs(-dx * sj + dy * cj)
            if q > p:
                p = q
            v = R - p
            P[i, j] = v
            P[j, i] = v
    return P


@njit(cache=True, fastmath=True)
def energy(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    L: float,
    lam: float = 25.0,
    tau: float = 0.0,
) -> float:
    """Penalty energy; zero iff feasible inside a box of side L."""
    c, s, h, _ = _prep(a)
    n = x.shape[0]
    E = 0.0
    for i in range(n):
        hi = h[i]
        p = x[i] + hi - L
        if p > 0.0:
            E += lam * p * p
        p = hi - x[i]
        if p > 0.0:
            E += lam * p * p
        p = y[i] + hi - L
        if p > 0.0:
            E += lam * p * p
        p = hi - y[i]
        if p > 0.0:
            E += lam * p * p
    for i in range(n - 1):
        ci = c[i]
        si = s[i]
        for j in range(i + 1, n):
            cj = c[j]
            sj = s[j]
            d = a[i] - a[j]
            R = 0.5 * (1.0 + _sabs(math.cos(d), tau) + _sabs(math.sin(d), tau))
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            p = _sabs(dx * ci + dy * si, tau)
            q = _sabs(-dx * si + dy * ci, tau)
            if q > p:
                p = q
            q = _sabs(dx * cj + dy * sj, tau)
            if q > p:
                p = q
            q = _sabs(-dx * sj + dy * cj, tau)
            if q > p:
                p = q
            pen = R - p
            if pen > 0.0:
                E += pen * pen
    return E


@njit(cache=True, fastmath=True)
def energy_grad(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    L: float,
    lam: float = 25.0,
    tau: float = 0.0,
):
    """Energy and gradient (includes the rotation of the winning axis)."""
    c, s, h, dh = _prep(a)
    n = x.shape[0]
    E = 0.0
    gx = np.zeros(n, dtype=np.float64)
    gy = np.zeros(n, dtype=np.float64)
    ga = np.zeros(n, dtype=np.float64)

    for i in range(n):
        hi = h[i]
        dhi = dh[i]
        p = x[i] + hi - L
        if p > 0.0:
            E += lam * p * p
            gx[i] += 2.0 * lam * p
            ga[i] += 2.0 * lam * p * dhi
        p = hi - x[i]
        if p > 0.0:
            E += lam * p * p
            gx[i] -= 2.0 * lam * p
            ga[i] += 2.0 * lam * p * dhi
        p = y[i] + hi - L
        if p > 0.0:
            E += lam * p * p
            gy[i] += 2.0 * lam * p
            ga[i] += 2.0 * lam * p * dhi
        p = hi - y[i]
        if p > 0.0:
            E += lam * p * p
            gy[i] -= 2.0 * lam * p
            ga[i] += 2.0 * lam * p * dhi

    for i in range(n - 1):
        ci = c[i]
        si = s[i]
        for j in range(i + 1, n):
            cj = c[j]
            sj = s[j]
            d = a[i] - a[j]
            cd = math.cos(d)
            sd = math.sin(d)
            acd = _sabs(cd, tau)
            asd = _sabs(sd, tau)
            R = 0.5 * (1.0 + acd + asd)
            if tau > 0.0:
                dR = 0.5 * (-cd * sd / acd + sd * cd / asd)
            else:
                dR = 0.5 * (
                    -sd * (1.0 if cd >= 0.0 else -1.0)
                    + cd * (1.0 if sd >= 0.0 else -1.0)
                )
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            ax = ci
            ay = si
            axis_id = 0
            p = _sabs(dx * ax + dy * ay, tau)
            q = _sabs(-dx * si + dy * ci, tau)
            if q > p:
                p = q
                ax = -si
                ay = ci
                axis_id = 1
            q = _sabs(dx * cj + dy * sj, tau)
            if q > p:
                p = q
                ax = cj
                ay = sj
                axis_id = 2
            q = _sabs(-dx * sj + dy * cj, tau)
            if q > p:
                p = q
                ax = -sj
                ay = cj
                axis_id = 3

            pen = R - p
            if pen > 0.0:
                E += pen * pen
                dp = 2.0 * pen
                sd_ = (dx * ax + dy * ay) / p
                # centre derivatives
                gx[i] -= dp * sd_ * ax
                gy[i] -= dp * sd_ * ay
                gx[j] += dp * sd_ * ax
                gy[j] += dp * sd_ * ay
                # rotation of the winning axis
                dpi = 0.0
                dpj = 0.0
                if axis_id == 0:
                    dpi = sd_ * (-dx * si + dy * ci)
                elif axis_id == 1:
                    dpi = -sd_ * (dx * ci + dy * si)
                elif axis_id == 2:
                    dpj = sd_ * (-dx * sj + dy * cj)
                else:
                    dpj = -sd_ * (dx * cj + dy * sj)
                ga[i] += dp * (dR - dpi)
                ga[j] += dp * (-dR - dpj)
    return E, gx, gy, ga


@njit(cache=True, fastmath=True)
def repair_scale(x: np.ndarray, y: np.ndarray, a: np.ndarray, clr: float = 0.0) -> float:
    """Smallest uniform dilation of the centre cloud that removes all overlaps."""
    c, s, _, _ = _prep(a)
    n = x.shape[0]
    f = 1.0
    for i in range(n - 1):
        ci = c[i]
        si = s[i]
        for j in range(i + 1, n):
            cj = c[j]
            sj = s[j]
            d = a[i] - a[j]
            R = 0.5 * (1.0 + abs(math.cos(d)) + abs(math.sin(d))) + clr
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            p = abs(dx * ci + dy * si)
            q = abs(-dx * si + dy * ci)
            if q > p:
                p = q
            q = abs(dx * cj + dy * sj)
            if q > p:
                p = q
            q = abs(-dx * sj + dy * cj)
            if q > p:
                p = q
            if p < 1e-15:
                continue
            need = R / p
            if need > f:
                f = need
    return f


def normalise(sq: np.ndarray) -> np.ndarray:
    """Translate so the bounding box starts at the origin (metric-invariant)."""
    sq = np.array(sq, dtype=np.float64, copy=True)
    h = half_extent(sq[:, 2])
    sq[:, 0] -= (sq[:, 0] - h).min()
    sq[:, 1] -= (sq[:, 1] - h).min()
    return sq


def solid(sq: np.ndarray, tol: float = 1e-9) -> bool:
    return (
        pen_max(
            np.ascontiguousarray(sq[:, 0]),
            np.ascontiguousarray(sq[:, 1]),
            np.ascontiguousarray(sq[:, 2]),
        )
        <= tol
    )
