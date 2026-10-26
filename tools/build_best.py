"""Merge all known sources into submissions/best.csv and certify it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import geom, io, validate

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "assets/kmmmvs_squares-in-squares-best-packings/submission.csv"
COU = ROOT / "submissions/couzo.csv"


def side(sq: np.ndarray) -> float:
    return geom.side_of(
        np.ascontiguousarray(sq[:, 0]),
        np.ascontiguousarray(sq[:, 1]),
        np.ascontiguousarray(sq[:, 2]),
    )


def main() -> None:
    sources = [io.load(PUB)]
    if COU.exists():
        sources.append(io.load(COU))
    merged: dict[int, np.ndarray] = {}
    prov: dict[int, str] = {}
    for n in range(1, 1001):
        best_L, best_sq, best_src = 1e18, None, "none"
        for name, src in zip(["public", "couzo"], sources):
            sq = src.get(n)
            if sq is None:
                continue
            L = side(sq)
            if L < best_L - 1e-12:
                best_L, best_sq, best_src = L, sq, name
        if best_sq is None:
            raise SystemExit(f"no source for n={n}")
        merged[n] = best_sq
        prov[n] = best_src

    a = validate.audit(merged)
    print(f"merged float total={a['total']:.10f} bad={a['n_bad']} at_grid={a['at_grid']}")
    ok, total = validate.gate(merged, ROOT / "submissions/best.csv")
    print(f"gate={'PASS' if ok else 'FAIL'} official_total={total:.10f}")
    import json

    (ROOT / "submissions/best_provenance.json").write_text(
        json.dumps({str(k): v for k, v in prov.items()}, indent=0)
    )


if __name__ == "__main__":
    main()
