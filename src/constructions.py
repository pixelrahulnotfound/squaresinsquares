"""A03 — generative construction families.

Ported from the (public, competition-licensed) `squarecraft-beyond-the-grid`
approach and re-validated with our own SAT checker.  Each family produces a
layout for a *capacity*, and every layout can be trimmed to any smaller ``n``
and extended by outer shells, which is what covers the whole range 1..1000.

Families
--------
* ``goebel_square(a, b)``  four triangular staircases + tilted ``b x b`` core
* ``goebel_strip(a, b)``   two staircase triangles + tilted chain
* ``rectangle_core``       four staircases with a ``b x c`` tilted core
* ``asymmetric_core``      unequal corner staircases, free core centre

Capacity and side
-----------------
``goebel_square``: ``n = 2a(a+1) + b²``,
``L = max(a + 1 + b/√2, 2a, b√2)``
``goebel_strip`` : ``n = a(a+1) + b + 2``,
``L = max(a + 1 + 1/√2, 2 + b/√2)``
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "extent",
    "outer_shell",
    "offer",
    "goebel_square",
    "goebel_strip",
    "rectangle_core",
    "asymmetric_core",
    "enumerate_best",
    "grid",
]


def grid(n: int) -> np.ndarray:
    k = math.ceil(math.sqrt(n))
    return np.array([[i % k + 0.5, i // k + 0.5, 0.0] for i in range(n)])


def extent(p: np.ndarray) -> tuple[np.ndarray, float]:
    p = np.asarray(p, dtype=np.float64).copy()
    c = np.cos(p[:, 2])
    s = np.sin(p[:, 2])
    h = 0.5 * (np.abs(c) + np.abs(s))
    p[:, 0] -= (p[:, 0] - h).min()
    p[:, 1] -= (p[:, 1] - h).min()
    return p, float(max((p[:, 0] + h).max(), (p[:, 1] + h).max()))


def outer_shell(L: float) -> np.ndarray:
    """Top row + right column filling the ``L -> L+1`` frame."""
    m = int(math.floor(L))
    top = [[i + 0.5, L + 0.5, 0.0] for i in range(m + 1)]
    right = [[L + 0.5, i + 0.5, 0.0] for i in range(m)]
    return np.array(top + right, dtype=np.float64)


def _tri(steps: int) -> np.ndarray:
    return np.array([[i + 0.5, j + 0.5] for i in range(steps) for j in range(steps - i)])


def _rot_block(steps: int, k: int, L: float) -> np.ndarray:
    q = _tri(steps)
    for _ in range(k):
        q = np.column_stack([L - q[:, 1], q[:, 0]])
    return np.column_stack([q, np.zeros(len(q))])


def goebel_square(a: int, b: int) -> np.ndarray:
    L = max(a + 1 + b / math.sqrt(2), 2 * a, b * math.sqrt(2))
    blocks = [_rot_block(a, k, L) for k in range(4)]
    core = []
    for i in range(b):
        for j in range(b):
            u = i - (b - 1) / 2
            v = j - (b - 1) / 2
            core.append([L / 2 + (u - v) / math.sqrt(2), L / 2 + (u + v) / math.sqrt(2), math.pi / 4])
    return np.vstack(blocks + [np.array(core, dtype=np.float64)])


def goebel_strip(a: int, b: int) -> np.ndarray:
    L = max(a + 1 + 1 / math.sqrt(2), 2 + b / math.sqrt(2))
    tri = np.column_stack([_tri(a), np.zeros(a * (a + 1) // 2)])
    other = tri.copy()
    other[:, :2] = L - other[:, :2]
    chain = np.array(
        [
            [L / 2 + (i - (b - 1) / 2) / math.sqrt(2), L / 2 - (i - (b - 1) / 2) / math.sqrt(2), math.pi / 4]
            for i in range(b)
        ]
    )
    corners = np.array([[0.5, L - 0.5, 0.0], [L - 0.5, 0.5, 0.0]])
    return np.vstack([tri, other, chain, corners])


def rectangle_core(a: int, d: int, b: int, c: int) -> np.ndarray:
    L = max(
        a + 1 + b / math.sqrt(2),
        d + 1 + c / math.sqrt(2),
        a + d,
        (b + c) / math.sqrt(2),
    )
    chunks = [_rot_block(steps, k, L) for k, steps in enumerate([a, d, a, d])]
    core = []
    for i in range(b):
        for j in range(c):
            u = i - (b - 1) / 2
            v = j - (c - 1) / 2
            core.append([L / 2 + (u - v) / math.sqrt(2), L / 2 + (u + v) / math.sqrt(2), math.pi / 4])
    return np.vstack(chunks + [np.array(core, dtype=np.float64)])


def asymmetric_core(a: int, d: int, u: int, v: int, b: int, c: int) -> np.ndarray:
    p1 = a + u + 1 + b / math.sqrt(2)
    p2 = d + v + 1 + c / math.sqrt(2)
    W = (b + c) / math.sqrt(2)
    L = max(
        a + u / 2 + 1 + b / math.sqrt(2),
        d + v / 2 + 1 + c / math.sqrt(2),
        a + d + u + v,
        W,
        (p1 + W) / 2,
        (p2 + W) / 2,
        (p1 + p2 + W) / 3,
    )
    U = max(L, p1)
    U = min(U, 2 * L - a - 1 - b / math.sqrt(2))
    V = max(0.0, p2 - L)
    V = min(V, L - d - 1 - c / math.sqrt(2))
    cx, cy = (U + V) / 2, (U - V) / 2
    chunks = [_rot_block(steps, k, L) for k, steps in enumerate([a + u, d, a, d + v])]
    core = []
    for i in range(b):
        for j in range(c):
            xx = i - (b - 1) / 2
            yy = j - (c - 1) / 2
            core.append(
                [cx + (xx - yy) / math.sqrt(2), cy + (xx + yy) / math.sqrt(2), math.pi / 4]
            )
    return np.vstack(chunks + [np.array(core, dtype=np.float64)])


def offer(
    best: dict[int, np.ndarray],
    side: dict[int, float],
    origin: dict[int, str],
    p: np.ndarray,
    label: str,
    n_max: int = 1000,
) -> None:
    """Trim ``p`` to every capacity it can improve (and extend by shells)."""
    p, L = extent(p)
    cap = min(len(p), n_max)
    for n in range(cap, max(0, int(math.floor(L)) ** 2), -1):
        if L < side[n] - 1e-9:
            best[n] = p[:n].copy()
            side[n] = L
            origin[n] = label


def enumerate_best(n_max: int = 1000, verbose: bool = True) -> tuple[dict, dict, dict, list]:
    best = {n: grid(n) for n in range(1, n_max + 1)}
    side = {n: float(math.ceil(math.sqrt(n))) for n in range(1, n_max + 1)}
    origin = {n: "grid" for n in range(1, n_max + 1)}

    stages = []

    def snap(tag: str) -> None:
        total = sum(side[n] / math.sqrt(n) - 1.0 for n in best)
        stages.append((tag, total))
        if verbose:
            print(f"  {tag:<28s} score={total:.6f}", flush=True)

    snap("grid")

    def with_shells(p: np.ndarray, label: str) -> None:
        p, L = extent(p)
        offer(best, side, origin, p, label, n_max)
        while L < n_max**0.5 + 2:
            p = np.vstack([p, outer_shell(L)])
            L += 1.0
            offer(best, side, origin, p, label + "+shell", n_max)

    # ---- phase 1: arithmetic-only candidate screen -------------------------
    # ``cand[n] = (L, tag, (family, args))`` -- no arrays are built here.
    cand: dict[int, tuple[float, str, tuple]] = {}

    def consider(cap: int, L: float, tag: str, spec: tuple) -> None:
        if L <= 0.0:
            return
        lo = max(1, int(math.floor(L)) ** 2 + 1)
        hi = min(cap, n_max)
        for n in range(hi, lo - 1, -1):
            if L < side[n] - 1e-9:
                cur = cand.get(n)
                if cur is None or L < cur[0] - 1e-12:
                    cand[n] = (L, tag, spec)

    for a in range(1, 33):
        for b in range(1, 47):
            n_sq = 2 * a * (a + 1) + b * b
            L_sq = max(a + 1 + b / math.sqrt(2), 2 * a, b * math.sqrt(2))
            if n_sq <= n_max and L_sq < math.ceil(math.sqrt(n_sq)) - 1e-9:
                consider(n_sq, L_sq, f"goebel_square a={a} b={b}", ("gs", a, b))
            n_st = a * (a + 1) + b + 2
            L_st = max(a + 1 + 1 / math.sqrt(2), 2 + b / math.sqrt(2))
            if n_st <= n_max and L_st < math.ceil(math.sqrt(n_st)) - 1e-9:
                consider(n_st, L_st, f"goebel_strip a={a} b={b}", ("gt", a, b))
    snap(f"goebel screen ({len(cand)})")

    for a in range(34):
        for d in range(a + 1):
            base = a * (a + 1) + d * (d + 1)
            if base >= n_max:
                continue
            for b in range(1, 48):
                for c in range(1, 48):
                    n = base + b * c
                    if n > n_max:
                        break
                    L = max(
                        a + 1 + b / math.sqrt(2),
                        d + 1 + c / math.sqrt(2),
                        a + d,
                        (b + c) / math.sqrt(2),
                    )
                    if L < math.ceil(math.sqrt(n)) - 1e-9:
                        consider(n, L, f"rect {a}/{d}/{b}/{c}", ("rc", a, d, b, c))
    snap(f"rect screen ({len(cand)})")

    for a in range(15):
        for d in range(a + 1):
            for u in range(0, 11):
                for v in range(0, 11):
                    base = (
                        a * (a + 1)
                        + d * (d + 1)
                        + (a + u) * (a + u + 1)
                        + (d + v) * (d + v + 1)
                    )
                    if base >= n_max:
                        continue
                    p1 = a + u + 1
                    p2 = d + v + 1
                    for b in range(1, 32):
                        for c in range(1, 32):
                            n = base + b * c
                            if n > n_max:
                                break
                            W = (b + c) / math.sqrt(2)
                            p1b = p1 + b / math.sqrt(2)
                            p2b = p2 + c / math.sqrt(2)
                            L = max(
                                a + u / 2 + 1 + b / math.sqrt(2),
                                d + v / 2 + 1 + c / math.sqrt(2),
                                a + d + u + v,
                                W,
                                (p1b + W) / 2,
                                (p2b + W) / 2,
                                (p1b + p2b + W) / 3,
                            )
                            if L < math.ceil(math.sqrt(n)) - 1e-9:
                                consider(n, L, f"asym {a}/{d}/{u}/{v}/{b}/{c}", ("ac", a, d, u, v, b, c))
    snap(f"asym screen ({len(cand)})")

    # ---- phase 2: build only the winners, then shell them -------------------
    builders = {
        "gs": lambda t: goebel_square(t[1], t[2]),
        "gt": lambda t: goebel_strip(t[1], t[2]),
        "rc": lambda t: rectangle_core(t[1], t[2], t[3], t[4]),
        "ac": lambda t: asymmetric_core(t[1], t[2], t[3], t[4], t[5], t[6]),
    }
    for n, (_, tag, spec) in sorted(cand.items()):
        try:
            p = builders[spec[0]](spec)
        except Exception:
            continue
        with_shells(p, tag)
    snap("built + shells")

    return best, side, origin, stages


if __name__ == "__main__":
    import sys

    from . import geom

    best, side, origin, stages = enumerate_best()
    bad = 0
    for n, p in best.items():
        pen = geom.pen_max(
            np.ascontiguousarray(p[:, 0]),
            np.ascontiguousarray(p[:, 1]),
            np.ascontiguousarray(p[:, 2]),
        )
        if pen > 1e-9:
            bad += 1
            if bad < 5:
                print(f"  INVALID n={n} pen={pen:.2e} ({origin[n]})")
    total = sum(side[n] / math.sqrt(n) - 1.0 for n in best)
    print(f"constructions: total={total:.6f} invalid={bad}")
