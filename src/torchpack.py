"""GPU search engine (PyTorch): batched penalty quench + basin hopping.

Designed for Kaggle notebooks (T4/P100).  Runs on CPU too (slow) so the maths
can be tested locally.

Layout tensor: ``X`` of shape ``(B, n, 3)`` = ``B`` independent basins of the
same puzzle size ``n``; ``L`` is a ``(B,)`` tensor of box sides.
"""
from __future__ import annotations

import math

import torch

TAUS = (1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 0.0)


def _sabs(z: torch.Tensor, tau: float) -> torch.Tensor:
    if tau == 0.0:
        return z.abs()
    return torch.sqrt(z * z + tau * tau)


def energy(
    X: torch.Tensor,
    L: torch.Tensor,
    lam: float = 25.0,
    tau: float = 0.0,
    chunk: int = 64,
) -> torch.Tensor:
    """Penalty energy per basin, shape ``(B,)``. Differentiable in ``X``."""
    x = X[:, :, 0]
    y = X[:, :, 1]
    a = X[:, :, 2]
    n = X.shape[1]
    dev = X.device
    c = torch.cos(a)
    s = torch.sin(a)
    h = 0.5 * (c.abs() + s.abs())
    Lc = L[:, None]
    wall = (
        torch.relu(x + h - Lc) ** 2
        + torch.relu(h - x) ** 2
        + torch.relu(y + h - Lc) ** 2
        + torch.relu(h - y) ** 2
    )
    E = lam * wall.sum(1)
    idx = torch.arange(n, device=dev)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        ii = idx[i0:i1]
        dx = x[:, i0:i1, None] - x[:, None, :]
        dy = y[:, i0:i1, None] - y[:, None, :]
        ci = c[:, i0:i1, None]
        si = s[:, i0:i1, None]
        cj = c[:, None, :]
        sj = s[:, None, :]
        d = a[:, i0:i1, None] - a[:, None, :]
        R = 0.5 * (1.0 + _sabs(torch.cos(d), tau) + _sabs(torch.sin(d), tau))
        p1 = _sabs(dx * ci + dy * si, tau)
        p2 = _sabs(-dx * si + dy * ci, tau)
        p3 = _sabs(dx * cj + dy * sj, tau)
        p4 = _sabs(-dx * sj + dy * cj, tau)
        p = torch.maximum(torch.maximum(p1, p2), torch.maximum(p3, p4))
        pen = R - p
        mask = (ii[:, None] < idx[None, :]).to(pen.dtype)
        E = E + (torch.relu(pen * mask) ** 2).sum((1, 2))
    return E


@torch.no_grad()
def pen_max(X: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    x, y, a = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    n = X.shape[1]
    dev = X.device
    c, s = torch.cos(a), torch.sin(a)
    best = torch.zeros(X.shape[0], device=dev, dtype=X.dtype)
    idx = torch.arange(n, device=dev)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        ii = idx[i0:i1]
        dx = x[:, i0:i1, None] - x[:, None, :]
        dy = y[:, i0:i1, None] - y[:, None, :]
        ci, si = c[:, i0:i1, None], s[:, i0:i1, None]
        cj, sj = c[:, None, :], s[:, None, :]
        d = a[:, i0:i1, None] - a[:, None, :]
        R = 0.5 * (1.0 + torch.cos(d).abs() + torch.sin(d).abs())
        p1 = (dx * ci + dy * si).abs()
        p2 = (-dx * si + dy * ci).abs()
        p3 = (dx * cj + dy * sj).abs()
        p4 = (-dx * sj + dy * cj).abs()
        p = torch.maximum(torch.maximum(p1, p2), torch.maximum(p3, p4))
        pen = R - p
        mask = (ii[:, None] < idx[None, :]).to(pen.dtype)
        cur = (pen * mask).amax(dim=(1, 2))
        best = torch.maximum(best, cur)
    return best


@torch.no_grad()
def side_of(X: torch.Tensor) -> torch.Tensor:
    a = X[:, :, 2]
    h = 0.5 * (torch.cos(a).abs() + torch.sin(a).abs())
    x, y = X[:, :, 0], X[:, :, 1]
    return torch.maximum(
        (x + h).amax(1) - (x - h).amin(1),
        (y + h).amax(1) - (y - h).amin(1),
    )


def quench(
    X: torch.Tensor,
    L: torch.Tensor,
    steps: int = 900,
    lr: float = 0.02,
    lam: float = 25.0,
    taus=TAUS,
    noise: float = 0.0,
    chunk: int = 64,
) -> torch.Tensor:
    """Adam-based continuation quench (fast, robust, GPU friendly)."""
    X = X.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=lr)
    per = max(1, steps // len(taus))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=per, eta_min=lr * 0.02)
    for tau in taus:
        sched.last_epoch = 0
        for _ in range(per):
            opt.zero_grad(set_to_none=True)
            E = energy(X, L, lam=lam, tau=tau, chunk=chunk).sum()
            E.backward()
            if noise > 0.0:
                with torch.no_grad():
                    X.grad.add_(torch.randn_like(X.grad) * noise)
            opt.step()
            sched.step()
    return X.detach()


def kick(
    X: torch.Tensor,
    sigma: float = 0.06,
    frac: float = 0.15,
    ang: float = 0.06,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Perturb a random subset of squares in every basin."""
    B, n, _ = X.shape
    dev = X.device
    k = max(2, int(frac * n))
    out = X.clone()
    score = torch.rand(B, n, device=dev, generator=gen)
    thr = score.topk(k, dim=1).values[:, -1:]
    sel = (score >= thr).to(X.dtype)
    out[:, :, 0] += torch.randn(B, n, device=dev, generator=gen) * sigma * sel
    out[:, :, 1] += torch.randn(B, n, device=dev, generator=gen) * sigma * sel
    out[:, :, 2] += torch.randn(B, n, device=dev, generator=gen) * ang * sel
    return out


def normalise(X: torch.Tensor) -> torch.Tensor:
    a = X[:, :, 2]
    h = 0.5 * (torch.cos(a).abs() + torch.sin(a).abs())
    dx = (X[:, :, 0] - h).amin(1, keepdim=True)
    dy = (X[:, :, 1] - h).amin(1, keepdim=True)
    out = X.clone()
    out[:, :, 0] -= dx
    out[:, :, 1] -= dy
    return out


def hop(
    seed: torch.Tensor,
    L_target: float,
    basins: int = 64,
    rounds: int = 1,
    steps: int = 900,
    lr: float = 0.02,
    sigma: float = 0.06,
    frac: float = 0.15,
    ang: float = 0.06,
    chunk: int = 64,
    gen: torch.Generator | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Basin hopping from one seed layout at a fixed target side.

    Returns ``(X, pen, L)`` for the best basin per round (feasible preferred).
    """
    dev = seed.device
    n = seed.shape[0]
    B = basins
    X = seed[None, :, :].repeat(B, 1, 1)
    L = torch.full((B,), float(L_target), device=dev, dtype=X.dtype)
    bestX, bestL, bestPen = None, None, None
    for r in range(rounds):
        Xk = kick(X, sigma=sigma, frac=frac, ang=ang, gen=gen)
        Xk = quench(Xk, L, steps=steps, lr=lr, chunk=chunk)
        pen = pen_max(Xk, chunk=chunk)
        feas = pen <= 1e-7
        candL = side_of(Xk)
        if feas.any():
            score = torch.where(feas, candL, torch.full_like(candL, 1e9))
            j = int(torch.argmin(score))
            if bestL is None or score[j] < bestL:
                bestX, bestL, bestPen = Xk[j].clone(), float(score[j]), float(pen[j])
        if verbose:
            print(
                f"    round {r}: feasible={int(feas.sum())}/{B} "
                f"pen_min={float(pen.min()):.2e} L_min={float(candL.min()):.6f}",
                flush=True,
            )
        # next round starts from the best feasible basin if any, else the least-penetrating
        if feas.any():
            X = Xk[torch.where(feas, candL, torch.full_like(candL, 1e9)).argmin()][None].repeat(B, 1, 1)
        else:
            X = Xk[pen.argmin()][None].repeat(B, 1, 1)
        if bestL is not None:
            L = torch.full((B,), float(bestL), device=dev, dtype=X.dtype)
    if bestX is None:
        return X, pen_max(X, chunk=chunk), side_of(X)
    return bestX, torch.tensor([bestPen], device=dev), torch.tensor([bestL], device=dev)


def squeeze_hop(
    seed: torch.Tensor,
    basins: int = 64,
    rounds: int = 4,
    steps: int = 900,
    rel: float = 1e-3,
    chunk: int = 64,
    gen: torch.Generator | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Repeatedly try to shrink the box by ``rel`` and keep any feasible result."""
    dev = seed.device
    X = seed.clone()
    L = side_of(X)[0]
    for r in range(rounds):
        target = float(L) * (1.0 - rel)
        Xb, pen, Lb = hop(
            X[0] if X.dim() == 3 else X,
            target,
            basins=basins,
            steps=steps,
            chunk=chunk,
            gen=gen,
            verbose=verbose,
        )
        if float(Lb[0]) < float(L):
            X = Xb[None] if Xb.dim() == 2 else Xb
            L = Lb[0]
            if verbose:
                print(f"  squeeze round {r}: L={float(L):.9f}", flush=True)
        else:
            break
    return X if X.dim() == 3 else X[None]
