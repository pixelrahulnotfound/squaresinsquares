"""Squares in Squares campaign: run the sqsearch engine over the unsolved n.

Reads the current champion submission from the attached dataset, then spends a
wall-clock budget trying to beat it on the puzzles where we are still exactly at
the grid (n in 325..1000).  Everything is checkpointed to /kaggle/working.

Output:
  improved.jsonl   one line per real improvement (n, side, x, y, t)
  merged.csv       full submission with every improvement folded in
  report.json      per-n summary
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

WORK = "/kaggle/working"


def find(name):
    for root, _, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


BEST = find("best.csv")
BIN = os.path.join(WORK, "sqsearch")
if not os.path.exists(BIN):
    shutil.copy(find("sqsearch"), BIN)
    os.chmod(BIN, 0o755)
THREADS = 4
DEADLINE = time.time() + 4.0 * 3600
MIN_SLICE = 25.0
GAIN_EPS = 1e-9


def side_of(x, y, a):
    h = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
    return max(float((x + h).max() - (x - h).min()), float((y + h).max() - (y - h).min()))


def load_best(path):
    df = pd.read_csv(path)
    packs = {}
    for n, g in df.groupby("n", sort=True):
        g = g.sort_values("square_id")
        packs[int(n)] = np.stack(
            [g.x.to_numpy(float), g.y.to_numpy(float), g.angle.to_numpy(float)], 1
        )
    return packs, list(df.columns)


def write_seed(packs, n, path, pts=None):
    p = packs[n] if pts is None else pts
    with open(path, "w") as f:
        for k, name in ((0, "x"), (1, "y"), (2, "t")):
            f.write('"%s":[%s]\n' % (name, ",".join(repr(float(v)) for v in p[:, k])))


def grid_holes(n, rng):
    k = int(math.ceil(math.sqrt(n)))
    cells = k * k
    pts = np.array([[((i % k) + 0.5), ((i // k) + 0.5), 0.0] for i in range(cells)], float)
    if cells > n:
        drop = rng.choice(cells, cells - n, replace=False)
        pts = np.delete(pts, drop, axis=0)
    return pts


def run(args, timeout):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    for line in p.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("kind") == "summary":
            return d
    return None


def fresh(n, budget, arms, t_hot, timeout):
    cmd = [BIN, "--n", str(n), "--chains", "4", "--budget-moves", str(budget),
           "--threads", str(THREADS), "--t-hot", str(t_hot), "--steps", "400000", *arms]
    return run(cmd, timeout)


def seeded(seed, budget, arms, t_hot, jitter, timeout):
    cmd = [BIN, "--seed-config", seed, "--chains", "4", "--budget-moves", str(budget),
           "--threads", str(THREADS), "--t-hot", str(t_hot), "--jitter", str(jitter),
           "--steps", "300000", *arms]
    return run(cmd, timeout)


def main():
    print("input tree:", sorted(os.listdir("/kaggle/input")), flush=True)
    print("engine:", BIN, "| data:", BEST, flush=True)
    packs, cols = load_best(BEST)
    st = subprocess.run([BIN, "--selftest"], capture_output=True, text=True)
    print("selftest:", st.stdout.strip().splitlines()[-1] if st.stdout else st.stderr[-200:], flush=True)

    cur = {n: side_of(*[packs[n][:, k] for k in range(3)]) for n in packs}
    # Perfect squares are exactly optimal at sqrt(n): nothing to find there.
    targets = [n for n in sorted(packs)
               if 325 <= n <= 1000
               and abs(cur[n] - math.ceil(math.sqrt(n))) < 1e-6
               and int(math.isqrt(n)) ** 2 != n]
    # Lowest density first: those grids sit furthest above the area bound, so
    # they have the most room and the most to gain.
    targets.sort(key=lambda n: (n / math.ceil(math.sqrt(n)) ** 2, -1.0 / math.sqrt(n)))
    print(f"targets: {len(targets)} non-square grid puzzles; "
          f"first n={targets[0]} last n={targets[-1]}", flush=True)

    improvd = {}
    log = open(os.path.join(WORK, "improved.jsonl"), "a")
    rng = np.random.default_rng(int(time.time()) % 2**31)
    seed_path = os.path.join(WORK, "seed.json")
    gh_path = os.path.join(WORK, "gh.json")

    def consider(n, d, tag):
        if not d:
            return False
        x = np.array(d["x"]); y = np.array(d["y"]); t = np.array(d["t"])
        ov = float(d.get("best_overlap", 1.0))
        if ov > 1e-9 or len(x) != n:
            return False
        s = side_of(x, y, t)
        if s < cur[n] - GAIN_EPS:
            print(f"  WIN n={n} {cur[n]:.9f} -> {s:.9f} ({tag})", flush=True)
            cur[n] = s
            improvd[n] = (x, y, t)
            log.write(json.dumps({"n": n, "side": s, "tag": tag,
                                  "x": x.tolist(), "y": y.tolist(), "t": t.tolist()}) + "\n")
            log.flush()
            return True
        return False

    # Pass 1: exploration arms that are OFF in the engine's defaults.  A fresh
    # anneal converges to the grid on these n, so the arms are the whole point.
    ratios = [(0.5, ["--p-perturb", "0.05", "--perturb-scale", "1"]),
              (0.5, ["--mu0", "1.0", "--mu1", "0.001"]),
              (0.8, ["--lambda0", "0.5", "--p-perturb", "0.1", "--perturb-scale", "1"]),
              (0.2, [])]
    for i, n in enumerate(targets):
        left = DEADLINE - time.time()
        if left < MIN_SLICE * 2:
            print("out of time", flush=True)
            break
        slice_s = max(MIN_SLICE, left / max(1, len(targets) - i))
        t0 = time.time()
        for t_hot, arms in ratios:
            if time.time() - t0 > slice_s:
                break
            d = fresh(n, 3_000_000, arms, t_hot, max(10, slice_s - (time.time() - t0)))
            if consider(n, d, f"fresh th={t_hot}"):
                break
        print(f"[p1 {i+1}/{len(targets)}] n={n} best={cur[n]:.9f} "
              f"wins={len(improvd)}", flush=True)

    # Pass 2: seeded refinement from the champion and from grid-minus-holes.
    for i, n in enumerate(targets):
        left = DEADLINE - time.time()
        if left < MIN_SLICE * 2:
            print("pass2 out of time", flush=True)
            break
        slice_s = max(MIN_SLICE, left / max(1, len(targets) - i))
        t0 = time.time()
        write_seed(packs, n, seed_path)
        d = seeded(seed_path, 2_000_000,
                   ["--p-perturb", "0.1", "--perturb-scale", "1", "--mu0", "1.0", "--mu1", "0.001"],
                   0.05, 0.01, max(10, slice_s * 0.5))
        if consider(n, d, "seeded"):
            continue
        pts = grid_holes(n, rng)
        write_seed(packs, n, gh_path, pts)
        d = seeded(gh_path, 2_000_000,
                   ["--p-perturb", "0.1", "--perturb-scale", "1", "--mu0", "2.0", "--mu1", "0.001"],
                   0.3, 0.05, max(10, slice_s - (time.time() - t0)))
        consider(n, d, "grid-holes")

    # Fold improvements into a full submission.
    rows = []
    for n in sorted(packs):
        p = packs[n]
        if n in improvd:
            x, y, t = improvd[n]
            p = np.stack([x, y, t], 1)
        for k in range(len(p)):
            rows.append((n, k, p[k, 0], p[k, 1], p[k, 2]))
    out = pd.DataFrame(rows, columns=["n", "square_id", "x", "y", "angle"])
    out.insert(0, "id", np.arange(len(out)))
    out.to_csv(os.path.join(WORK, "merged.csv"), index=False)

    score = sum(cur[n] / math.sqrt(n) - 1 for n in packs)
    with open(os.path.join(WORK, "report.json"), "w") as f:
        json.dump({"improved": sorted(improvd), "count": len(improvd),
                   "score": score, "targets": len(targets),
                   "sides": {str(n): cur[n] for n in improvd}}, f)
    print(f"DONE improved={len(improvd)} score={score:.12f}", flush=True)


if __name__ == "__main__":
    main()
