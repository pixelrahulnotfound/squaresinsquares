"""Submission IO: load layouts per puzzle, write full-precision CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data" / "sample_submission.csv"


def load(path: str | Path) -> dict[int, np.ndarray]:
    """Read a submission CSV into ``{n: (n,3) float64 array of x,y,theta}``."""
    df = pd.read_csv(path, dtype=str)
    for col in ("n", "square_id"):
        df[col] = df[col].astype(int)
    out: dict[int, np.ndarray] = {}
    for n, g in df.groupby("n", sort=True):
        g = g.sort_values("square_id")
        if len(g) != int(n):
            raise SystemExit(f"{path}: n={n} has {len(g)} rows")
        if g["square_id"].tolist() != list(range(int(n))):
            raise SystemExit(f"{path}: n={n} square_id not 0..{int(n)-1}")
        out[int(n)] = np.column_stack(
            [
                g["x"].astype(np.float64).to_numpy(),
                g["y"].astype(np.float64).to_numpy(),
                g["angle"].astype(np.float64).to_numpy(),
            ]
        )
    return out


def _template_ids() -> pd.DataFrame:
    tmpl = pd.read_csv(TEMPLATE, dtype=str)[["id", "n", "square_id"]]
    tmpl["n"] = tmpl["n"].astype(int)
    tmpl["square_id"] = tmpl["square_id"].astype(int)
    return tmpl


def write(packs: dict[int, np.ndarray], path: str | Path) -> int:
    """Write a complete submission; requires all n = 1..1000."""
    missing = [n for n in range(1, 1001) if n not in packs]
    if missing:
        raise SystemExit(f"missing puzzles: {missing[:10]} ... ({len(missing)} total)")
    tmpl = _template_ids()
    blocks = []
    for n in range(1, 1001):
        sq = np.asarray(packs[n], dtype=np.float64)
        g = tmpl[tmpl.n == n].sort_values("square_id")
        block = pd.DataFrame(
            {
                "id": g["id"].to_numpy(),
                "n": n,
                "square_id": g["square_id"].to_numpy(),
                "x": [repr(float(v)) for v in sq[:, 0]],
                "y": [repr(float(v)) for v in sq[:, 1]],
                "angle": [repr(float(v)) for v in sq[:, 2]],
            }
        )
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True).sort_values("id")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(out)


def merge(*sources: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-puzzle argmin of the bounding side."""
    from .geom import side_of

    best: dict[int, np.ndarray] = {}
    for src in sources:
        for n, sq in src.items():
            L = side_of(sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy())
            if n not in best:
                best[n] = sq
                continue
            cur = best[n]
            Lc = side_of(cur[:, 0].copy(), cur[:, 1].copy(), cur[:, 2].copy())
            if L < Lc:
                best[n] = sq
    return best
