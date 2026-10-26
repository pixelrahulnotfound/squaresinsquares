"""Submission gate: schema -> float SAT -> dilation -> official metric.

A puzzle that fails the official checker is scored as the integer grid, so the
gate must run on the *exact bytes* that will be uploaded.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import io
from .geom import normalise, pen_max, repair_scale, side_of

ROOT = Path(__file__).resolve().parents[1]


def audit(packs: dict[int, np.ndarray], tol: float = 1e-9) -> dict:
    """Fast float64 audit. Returns per-n rows plus the worst offender."""
    rows = []
    for n, sq in sorted(packs.items()):
        x = np.ascontiguousarray(sq[:, 0])
        y = np.ascontiguousarray(sq[:, 1])
        a = np.ascontiguousarray(sq[:, 2])
        L = side_of(x, y, a)
        pen = pen_max(x, y, a)
        rows.append(
            {
                "n": n,
                "L": L,
                "score": L / math.sqrt(n) - 1.0,
                "pen": pen,
                "grid": L >= math.ceil(math.sqrt(n)) - 1e-12,
            }
        )
    total = sum(r["score"] for r in rows)
    bad = [r for r in rows if r["pen"] > tol]
    return {
        "rows": rows,
        "total": total,
        "bad": sorted(bad, key=lambda r: -r["pen"])[:10],
        "n_bad": len(bad),
        "at_grid": sum(1 for r in rows if r["grid"]),
    }


def dilate(packs: dict[int, np.ndarray], clr: float = 1e-11) -> dict[int, np.ndarray]:
    """Uniform dilation about each layout's centroid, then re-normalise.

    ``clr`` is *relative*: a layout spanning ``L`` gains an absolute clearance of
    roughly ``clr*L``, which survives ``repr(float)`` rounding (~1e-16 relative)
    and is provably above the metric's 50-digit interval resolution.
    """
    out = {}
    for n, sq in packs.items():
        sq = np.array(sq, dtype=np.float64, copy=True)
        f = repair_scale(
            np.ascontiguousarray(sq[:, 0]),
            np.ascontiguousarray(sq[:, 1]),
            np.ascontiguousarray(sq[:, 2]),
            clr,
        )
        f = max(f, 1.0 + clr)
        c = sq[:, :2].mean(axis=0)
        sq[:, :2] = c + f * (sq[:, :2] - c)
        out[n] = normalise(sq)
    return out


def official(path: str | Path, puzzles: set[int] | None = None) -> tuple[float, list]:
    """Run the competition metric exactly as Kaggle does."""
    path = Path(path)
    mdir = str(ROOT / "metric")
    if mdir not in sys.path:
        sys.path.insert(0, mdir)
    import squares_in_squares_metric as sim  # noqa: E402  (needs sys.path first)

    text = path.read_text()
    pm = sim._parse_submission_csv(text)
    total, results = sim._evaluate_puzzles_parallel(
        pm, set(range(1, 1001)) if puzzles is None else puzzles
    )
    return total, results


def gate(packs: dict[int, np.ndarray], work: str | Path, clr: float = 1e-11) -> tuple[bool, float]:
    """Full gate: dilate, write, certify with the official metric, escalate if needed.

    Exact-touching packings are *undecidable* in interval arithmetic, so every
    candidate gets a relative clearance before it is written.  If the metric
    still rejects puzzles (e.g. a genuine overlap), the clearance is escalated.
    """
    work = Path(work)
    total = 0.0
    for attempt, c in enumerate((clr, clr * 10, clr * 100, clr * 1e3)):
        cand = dilate(packs, c)
        a = audit(cand)
        io.write(cand, work)
        total, results = official(work)
        bad = [r for r in results if not str(r[3]).startswith("ok")]
        if not bad:
            return True, total
        print(f"  attempt {attempt + 1} clr={c:.0e}: {len(bad)} rejected, e.g. {bad[:3]}")
        if attempt == 3:
            return False, total
    return False, total


def main(argv: list[str]) -> None:
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else src.with_name(src.stem + "_gated.csv")
    packs = io.load(src)
    ok, total = gate(packs, out)
    print(f"{'PASS' if ok else 'FAIL'} {out} official_total={total:.10f}")


if __name__ == "__main__":
    main(sys.argv)
