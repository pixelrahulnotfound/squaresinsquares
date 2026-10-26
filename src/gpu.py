"""Batched L-BFGS packing search (GPU-first, CPU-testable).

The Adam quench in :mod:`src.torchpack` plateaus at 1e-8..1e-6 residual
penetration, which is larger than the box shrink we can afford, so every
acceptance test loses.  This module implements a fully batched L-BFGS with
per-basin two-loop recursion, per-basin backtracking line search and a penalty
continuation, so basins can reach ~1e-13 and tiny shrinks become viable.

Layout tensor ``X``: ``(B, n, 3)`` = ``(x, y, angle)`` per square.
``L``: ``(B,)`` target box side per basin.
"""
from __future__ import annotations

import torch

from .torchpack import energy, pen_max, side_of, normalise

ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 7.8125e-3, 1e-3, 1e-4)


def _E(X: torch.Tensor, L: torch.Tensor, lam: float, tau: float, chunk: int) -> torch.Tensor:
    return energy(X, L, lam=lam, tau=tau, chunk=chunk)


def lbfgs(
    X0: torch.Tensor,
    L: torch.Tensor,
    iters: int = 80,
    m: int = 10,
    lam: float = 1.0,
    tau: float = 0.0,
    chunk: int | None = None,
    alphas=ALPHAS,
) -> torch.Tensor:
    """Minimise the penalty energy of every basin independently."""
    B, n, _ = X0.shape
    D = n * 3
    if chunk is None:
        chunk = max(8, min(64, 2_000_000 // max(1, B * n)))
    P = X0.detach().reshape(B, D).clone()

    def ev(Pc: torch.Tensor):
        Pc = Pc.requires_grad_(True)
        eb = _E(Pc.view(B, n, 3), L, lam, tau, chunk)
        g, = torch.autograd.grad(eb.sum(), Pc)
        return eb.detach(), g.detach()

    E, G = ev(P)
    S: list[torch.Tensor] = []
    Y: list[torch.Tensor] = []
    RHO: list[torch.Tensor] = []
    A = torch.tensor(alphas, dtype=P.dtype, device=P.device)
    fails = torch.zeros(B, 1, dtype=P.dtype, device=P.device)

    for _ in range(iters):
        gn = G.norm(dim=1, keepdim=True).clamp_min(1e-30)
        if S:
            q = G.clone()
            coef: list[torch.Tensor] = []
            for s, y, rho in zip(reversed(S), reversed(Y), reversed(RHO)):
                a = rho * (s * q).sum(1, keepdim=True)
                q = q - a * y
                coef.append(a)
            s, y = S[-1], Y[-1]
            gamma = ((s * y).sum(1, keepdim=True) / (y * y).sum(1, keepdim=True).clamp_min(1e-30))
            gamma = gamma.clamp(1e-8, 1e8)
            r = q * gamma
            for (s, y, rho), a in zip(zip(S, Y, RHO), reversed(coef)):
                b = rho * (y * r).sum(1, keepdim=True)
                r = r + (a - b) * s
            d = -r
            dd = (G * d).sum(1, keepdim=True)
            bad = (dd > -1e-10 * gn * gn) | (~torch.isfinite(dd))
            d = torch.where(bad, -G / gn, d)
            dd = torch.where(bad, -gn, dd)
        else:
            d = -G / gn
            dd = -gn
        with torch.no_grad():
            tries = torch.stack(
                [_E((P + a * d).view(B, n, 3), L, lam, tau, chunk) for a in A], 0
            )
        arm = tries <= E.reshape(1, -1) + 1e-4 * A.reshape(-1, 1) * dd.reshape(1, B)
        anyok = arm.any(0)
        pick = torch.where(anyok, arm.to(torch.int8).argmax(0), tries.argmin(0))
        step = A[pick].reshape(B, 1)
        P_new = P + step * d
        E2, G2 = ev(P_new)
        use = (E2 < E).reshape(B, 1)
        P_old = P
        P = torch.where(use, P_new, P)
        E = torch.where(use.reshape(B), E2, E)
        G = torch.where(use, G2, G)
        svec = torch.where(use, P - P_old, torch.zeros_like(P))
        yvec = torch.where(use, G - G2, torch.zeros_like(P))
        sy = (svec * yvec).sum(1, keepdim=True)
        good = sy > 1e-18
        S.append(torch.where(good, svec, torch.zeros_like(svec)))
        Y.append(torch.where(good, yvec, torch.zeros_like(yvec)))
        RHO.append(torch.where(good, 1.0 / sy.clamp_min(1e-30), torch.zeros_like(sy)))
        if len(S) > m:
            S.pop(0)
            Y.pop(0)
            RHO.pop(0)
        fails = torch.where(use, torch.zeros_like(fails), (fails + 1).clamp_max(3))
        if float(fails.max()) >= 3:
            reset = fails >= 3
            for i in range(len(S)):
                S[i] = torch.where(reset, torch.zeros_like(S[i]), S[i])
                Y[i] = torch.where(reset, torch.zeros_like(Y[i]), Y[i])
                RHO[i] = torch.where(reset, torch.zeros_like(RHO[i]), RHO[i])
            fails = torch.where(reset, torch.zeros_like(fails), fails)
        if float(gn.max()) < 1e-15:
            break
    return P.view(B, n, 3)


def solve(
    X0: torch.Tensor,
    L: torch.Tensor,
    lams=(1.0, 10.0, 100.0, 1_000.0, 10_000.0),
    taus=(1e-3, 1e-5, 0.0),
    iters: int = 80,
    chunk: int | None = None,
) -> torch.Tensor:
    """Continuation on penalty weight then smoothing: soft moves, then rigid."""
    X = X0
    for tau in taus:
        for lam in lams:
            X = lbfgs(X, L, iters=iters, lam=lam, tau=tau, chunk=chunk)
    return X


@torch.no_grad()
def repair(X: torch.Tensor, clr: float = 1e-12) -> torch.Tensor:
    """Uniform dilation about the centroid until the worst penetration clears."""
    B, n, _ = X.shape
    x, y, a = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    c, s = torch.cos(a), torch.sin(a)
    h = 0.5 * (c.abs() + s.abs())
    dx = x[:, :, None] - x[:, None, :]
    dy = y[:, :, None] - y[:, None, :]
    ci, si = c[:, :, None], s[:, :, None]
    cj, sj = c[:, None, :], s[:, None, :]
    dd = a[:, :, None] - a[:, None, :]
    R = 0.5 * (1.0 + torch.cos(dd).abs() + torch.sin(dd).abs())
    p = torch.maximum(
        torch.maximum((dx * ci + dy * si).abs(), (-dx * si + dy * ci).abs()),
        torch.maximum((dx * cj + dy * sj).abs(), (-dx * sj + dy * cj).abs()),
    )
    eye = torch.eye(n, device=X.device, dtype=torch.bool)[None]
    worst = torch.where(eye, torch.zeros_like(R), R - p).amax(dim=(1, 2))
    scale = (1.0 + worst.clamp_min(0.0)).clamp_min(1.0 + clr)
    out = X.clone()
    cx = x.mean(1, keepdim=True)
    cy = y.mean(1, keepdim=True)
    out[:, :, 0] = cx + (x - cx) * scale[:, None]
    out[:, :, 1] = cy + (y - cy) * scale[:, None]
    return normalise(out)
