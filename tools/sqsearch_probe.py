"""Probe `sqsearch` on selected n and record anything that beats our current L.

Usage: python3 tools/sqsearch_probe.py [n ...] --chains 8 --budget 8000000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io, geom  # noqa: E402

BIN = os.environ.get("SQSEARCH", "/tmp/opencode/jlevy/sqsearch/target/release/sqsearch")
OUT = "runs/candidates"


def our_side(packs, n):
    s = packs[n]
    return float(geom.side_of(*[np.ascontiguousarray(s[:, k]) for k in range(3)]))


def write_seed(packs, n, path):
    s = packs[n]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for k, name in ((0, "x"), (1, "y"), (2, "t")):
            vals = ",".join(repr(float(v)) for v in s[:, k])
            f.write(f'"{name}":[{vals}]\n')
    return path


def run(n, chains, budget, threads, extra=()):
    cmd = [BIN, "--n", str(n), "--chains", str(chains), "--budget-moves", str(budget),
           "--threads", str(threads), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    summary = None
    chains_out = []
    for line in p.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("kind") == "summary":
            summary = d
        elif d.get("kind") == "chain":
            chains_out.append(d)
    return summary, chains_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ns", nargs="*", type=int)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--budget", type=int, default=8_000_000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--seeded", action="store_true")
    ap.add_argument("--t-hot", type=float, default=1e-3)
    ap.add_argument("--jitter", type=float, default=1e-3)
    a = ap.parse_args()

    packs = io.load("submissions/best.csv")
    ns = a.ns or [int(x) for x in sorted(packs)]
    os.makedirs(OUT, exist_ok=True)
    improved = 0
    for n in ns:
        cur = our_side(packs, n)
        grid = float(np.ceil(np.sqrt(n)))
        if a.seeded:
            seed = write_seed(packs, n, f"runs/seeds/n{n}.json")
            extra = ["--seed-config", seed, "--t-hot", str(a.t_hot),
                     "--jitter", str(a.jitter), *a.extra]
        else:
            seed = None
            extra = list(a.extra)
        s, ch = run(n, a.chains, a.budget, a.threads, extra)
        best = float(s["best_side"])
        tag = "WIN " if best < cur - 1e-12 else "    "
        print(f"{tag}n={n:4d} ours={cur:.12f} sqsearch={best:.12f} d={best-cur:+.3e} "
              f"grid={grid:.0f} overlap={s['best_overlap']:.1e} t={s['seconds']:.1f}s", flush=True)
        if best < cur - 1e-12:
            improved += 1
            cfg = ch[int(s["best_chain"])] if ch else s
            np.savez(f"{OUT}/n{n}.npz",
                     x=np.array(cfg["x"]), y=np.array(cfg["y"]), t=np.array(cfg["t"]),
                     side=best)
            with open(f"{OUT}/n{n}.json", "w") as f:
                json.dump({"n": n, "side": best, "ours": cur, "params": s.get("params")}, f)
    print(f"\nimproved {improved}/{len(ns)}")


if __name__ == "__main__":
    main()
