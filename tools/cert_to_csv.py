"""Convert exact rational certificates (Couzo / jlevy registry format) to submission rows.

Certificate format (see external/couzo/certificates):
    # comment line
    n side
    x y t          # n lines, x,y,t rationals; t = tan(theta/2)

Unit squares are centred at (x, y) and rotated by theta = 2*atan(t).

Touching packings cannot be certified from rounded decimals, so every centre is
dilated about the packing's bounding-box centre by (1 + DILATE) before writing.

Usage:
    python tools/cert_to_csv.py --certs external/couzo/certificates/best \
        --out submissions/couzo.csv [--ns 1-1000] [--dps 60]
"""
from __future__ import annotations

import argparse
import math
import re
from fractions import Fraction
from pathlib import Path

import pandas as pd

DILATE = Fraction(1, 10**30)


def parse_int_range(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out


def parse_cert(path: Path, dps: int = 60):
    import mpmath

    lines = [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    header = lines[0].split()
    n = int(header[0])
    squares = []
    with mpmath.workdps(dps):
        for ln in lines[1 : n + 1]:
            toks = ln.split()
            x = Fraction(toks[0])
            y = Fraction(toks[1])
            t = Fraction(toks[2])
            mx = mpmath.mpf(x.numerator) / x.denominator
            my = mpmath.mpf(y.numerator) / y.denominator
            mt = mpmath.mpf(t.numerator) / t.denominator
            theta = 2 * mpmath.atan(mt)
            squares.append((mx, my, theta))
        # bounding-box centre
        cx = (min(s[0] for s in squares) + max(s[0] for s in squares)) / 2
        cy = (min(s[1] for s in squares) + max(s[1] for s in squares)) / 2
        k = mpmath.mpf(1) + mpmath.mpf(DILATE.numerator) / DILATE.denominator
        return [
            (cx + (x - cx) * k, cy + (y - cy) * k, theta) for x, y, theta in squares
        ]


def fmt(value, digits: int) -> str:
    import mpmath

    return mpmath.nstr(value, digits, strip_zeros=False, min_fixed=-10**6, max_fixed=10**6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--certs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ns", default="1-1000")
    ap.add_argument("--dps", type=int, default=60)
    ap.add_argument("--digits", type=int, default=55)
    args = ap.parse_args()

    wanted = parse_int_range(args.ns)
    rows = []
    missing = []
    for n in sorted(wanted):
        path = args.certs / f"n{n}.cert"
        if not path.exists():
            missing.append(n)
            continue
        squares = parse_cert(path, args.dps)
        if len(squares) != n:
            raise SystemExit(f"cert {path} has {len(squares)} squares, expected {n}")
        for sid, (x, y, theta) in enumerate(squares):
            rows.append(
                {
                    "n": n,
                    "square_id": sid,
                    "x": fmt(x, args.digits),
                    "y": fmt(y, args.digits),
                    "angle": fmt(theta, args.digits),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(df)} puzzles={df.n.nunique()}")
    if missing:
        print(f"missing certs for {len(missing)} n: {missing[:20]}...")


if __name__ == "__main__":
    main()
