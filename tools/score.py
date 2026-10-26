"""Fast local scorer / validator for the Squares in Squares competition.

The Kaggle metric (metric/squares_in_squares_metric.py) is authoritative but
slow (interval arithmetic at 50-500 dps).  This module reproduces its geometry
in float64 for day-to-day work and can optionally re-certify with the official
metric.

    python tools/score.py submissions/couzo.csv
    python tools/score.py submissions/a.csv submissions/b.csv   # compare / merge
    python tools/score.py submission.csv --official              # mpmath metric

Geometry: side = max(max_x-min_x, max_y-min_y) over the four rotated corners,
per-puzzle score = side/sqrt(n) - 1, total = sum over n.
Angles are in RADIANS.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HALF_DIAG = np.array(
    [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], dtype=np.float64
)


def corners(x: np.ndarray, y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(n,4,2) corner array."""
    c = np.cos(a)[:, None]
    s = np.sin(a)[:, None]
    rot = np.empty((len(x), 4, 2))
    rot[:, :, 0] = HALF_DIAG[None, :, 0] * c - HALF_DIAG[None, :, 1] * s
    rot[:, :, 1] = HALF_DIAG[None, :, 0] * s + HALF_DIAG[None, :, 1] * c
    rot[:, :, 0] += x[:, None]
    rot[:, :, 1] += y[:, None]
    return rot


def side_of(x: np.ndarray, y: np.ndarray, a: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    c = corners(x, y, a)
    return float(
        max(c[:, :, 0].max() - c[:, :, 0].min(), c[:, :, 1].max() - c[:, :, 1].min())
    )


def max_overlap(x, y, a, tol=0.0, chunk=200_000) -> float:
    """Largest pairwise SAT penetration depth (positive => overlap).

    For unit squares the projection radius of square k on a unit axis at angle
    phi is 0.5*(|cos(a_k-phi)| + |sin(a_k-phi)|).  The four SAT axes reduce to
    the two square orientations (the +90deg axes have the same projections).
    """
    n = len(x)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    worst = 0.0
    m = len(iu[0])
    for start in range(0, m, chunk):
        ii = iu[0][start : start + chunk]
        jj = iu[1][start : start + chunk]
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        delta = a[ii] - a[jj]
        # radius of square j projected on square i's axes (and vice versa)
        rj = 0.5 * (np.abs(np.cos(delta)) + np.abs(np.sin(delta)))
        ui, vi = np.cos(a[ii]), np.sin(a[ii])
        uj, vj = np.cos(a[jj]), np.sin(a[jj])
        r = 0.5 + rj
        ov_i = np.minimum(
            r - np.abs(dx * ui + dy * vi),
            r - np.abs(-dx * vi + dy * ui),
        )
        ov_j = np.minimum(
            r - np.abs(dx * uj + dy * vj),
            r - np.abs(-dx * vj + dy * uj),
        )
        best = np.minimum(ov_i, ov_j)
        worst = max(worst, float(best.max(initial=0.0)))
    return worst - tol


def load(path: Path) -> dict[int, np.ndarray]:
    df = pd.read_csv(path, dtype=str)
    df["n"] = df["n"].astype(int)
    df["square_id"] = df["square_id"].astype(int)
    out = {}
    for n, g in df.groupby("n", sort=True):
        g = g.sort_values("square_id")
        if len(g) != n:
            raise SystemExit(f"{path}: n={n} has {len(g)} rows")
        if g["square_id"].tolist() != list(range(n)):
            raise SystemExit(f"{path}: n={n} square_id not 0..n-1")
        out[int(n)] = np.column_stack(
            [
                g["x"].astype(np.float64).to_numpy(),
                g["y"].astype(np.float64).to_numpy(),
                g["angle"].astype(np.float64).to_numpy(),
            ]
        )
    return out


def score_pack(sq: np.ndarray) -> tuple[float, float]:
    L = side_of(sq[:, 0], sq[:, 1], sq[:, 2])
    return L, L / math.sqrt(len(sq)) - 1.0


def score_all(packs: dict[int, np.ndarray], check=True):
    rows = []
    for n, sq in packs.items():
        L, s = score_pack(sq)
        ov = max_overlap(sq[:, 0], sq[:, 1], sq[:, 2]) if check else 0.0
        rows.append({"n": n, "L": L, "score": s, "overlap": ov})
    return pd.DataFrame(rows)


def official(path: Path, puzzles: set[int]) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sim", Path(__file__).resolve().parents[1] / "metric" / "squares_in_squares_metric.py"
    )
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    text = path.read_text()
    puzzles_map = sim._parse_submission_csv(text)
    total, results = sim._evaluate_puzzles_parallel(puzzles_map, puzzles)
    return pd.DataFrame(results, columns=["n", "score", "L", "reason"]), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+", type=Path)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--write", type=Path, default=None, help="merge best-of and write")
    args = ap.parse_args()

    report = {}
    for path in args.submissions:
        packs = load(path)
        df = score_all(packs, check=not args.no_check)
        bad = df[df.overlap > 1e-9]
        print(
            f"{path}: puzzles={len(df)} total={df.score.sum():.10f} "
            f"grid_gain={28.9376699166 - df.score.sum():.4f} "
            f"overlapping={len(bad)}"
        )
        if len(bad):
            print("  worst:", bad.nlargest(5, "overlap")[["n", "overlap"]].to_dict("records"))
        report[path] = packs

    if len(report) == 2 and args.write:
        (pa, A), (pb, B) = report.items()
        merged, src = {}, {}
        for n in sorted(set(A) | set(B)):
            cands = [(A.get(n), pa), (B.get(n), pb)]
            best = None
            for sq, p in cands:
                if sq is None:
                    continue
                L, s = score_pack(sq)
                if best is None or s < best[0]:
                    best = (s, sq, p)
            merged[n] = best[1]
            src[n] = best[2].name
        total = sum(score_pack(v)[1] for v in merged.values())
        print(f"merged total={total:.10f}")

        tmpl = pd.read_csv(next(iter([Path("data/sample_submission.csv"), Path("extracted/sample_submission.csv")])), dtype=str)
        tmpl = tmpl[["id", "n", "square_id"]]
        tmpl["n"] = tmpl["n"].astype(int)
        tmpl["square_id"] = tmpl["square_id"].astype(int)
        rows = []
        for n, sq in merged.items():
            g = tmpl[tmpl.n == n].sort_values("square_id")
            for (idx, sid), (x, y, a) in zip(g[["id", "square_id"]].itertuples(index=False), sq):
                rows.append({"id": idx, "n": n, "square_id": sid,
                             "x": repr(float(x)), "y": repr(float(y)), "angle": repr(float(a))})
        out = pd.DataFrame(rows).sort_values("id")
        args.write.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.write, index=False)
        print(f"wrote {args.write} rows={len(out)}")

    if args.official:
        for path in args.submissions:
            packs = load(path)
            df, total = official(path, set(packs))
            print(f"OFFICIAL {path}: total={total:.10f}")
            bad = df[~df.reason.astype(str).str.startswith("ok")]
            if len(bad):
                print("  rejected:", bad.head(10).to_dict("records"))
            df.to_csv(path.with_suffix(".official.csv"), index=False)


if __name__ == "__main__":
    main()
