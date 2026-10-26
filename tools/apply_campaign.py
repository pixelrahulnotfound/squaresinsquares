"""Fold a campaign result file (improved.jsonl) into a gated submission."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from src import io, validate
from src.geom import side_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base", default="submissions/best.csv")
    ap.add_argument("--out", default="submissions/campaign.csv")
    a = ap.parse_args()

    packs = io.load(a.base)
    before = sum(side_of(packs[n][:, 0].copy(), packs[n][:, 1].copy(), packs[n][:, 2].copy())
                 / math.sqrt(n) - 1 for n in packs)
    wins = []
    for line in open(a.jsonl):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n = int(d["n"])
        sq = np.column_stack([d["x"], d["y"], d["t"]]).astype(np.float64)
        if n not in packs or len(sq) != n or not np.isfinite(sq).all():
            print(f"  skip n={n}: malformed ({len(sq)} squares)")
            continue
        cur = side_of(packs[n][:, 0].copy(), packs[n][:, 1].copy(), packs[n][:, 2].copy())
        new = side_of(sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy())
        if new < cur - 1e-9:
            wins.append((n, cur, new))
            packs[n] = sq
    wins.sort()
    gain = sum((c - s) / math.sqrt(n) for n, c, s in wins)
    for n, c, s in wins:
        print(f"  n={n:4d} {c:.9f} -> {s:.9f}  ({s - c:+.6f})")
    print(f"accepted {len(wins)} wins, score {before:.12f} -> {before - gain:.12f}")
    if not wins:
        raise SystemExit("nothing to apply")
    ok, total = validate.gate(packs, a.out)
    print(f"gate {'PASS' if ok else 'FAIL'} official={total:.12f} ({a.out})")


if __name__ == "__main__":
    main()
