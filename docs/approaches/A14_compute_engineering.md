# A14 — Compute engineering & orchestration

**Priority: high — it is a multiplier on A06–A10, and the local machine is the
binding constraint.**

## The situation

* CPU only: **no CUDA** (`torch.cuda.is_available() == False`, no `nvidia-smi`).
* 20 cores, 15 GB RAM, no `ortools`/`highspy`/`cvxpy` (use `scipy.optimize.milp`
  / `linprog`, which ship HiGHS).
* Every public notebook that reaches 22.3078 uses a GPU for its search. We must
  compensate with **better seeds and better polishing per evaluation**, not with
  more evaluations.

## What to build

### 14.1 Fast inner loop
* **Spatial hash** for near‑pair queries (cell size `R_max ≈ 1.5`): `O(n)` per
  energy evaluation instead of `O(n²)`. ~30× at `n = 1000`.
* **numba** (`@njit(parallel=True)`) kernels for penetration, energy, gradient,
  and the repair.
* **Batched L‑BFGS**: many independent starts in one call (the CPU analogue of
  `malyshevdanil`'s CUDA kernel). Keep `M = 8` history, Armijo `c₁ = 1e-4`.
* **float32 propose / float64 verify**: cheap proposals, exact acceptance.

### 14.2 Parallelism
* Parallelise over **puzzles** (independent), not within a puzzle.
* `multiprocessing` with 16–18 workers (leave headroom); each worker owns a
  puzzle, checkpoints every `k` basins.
* Deterministic seeding: `seed = hash(n, campaign, worker)` so any result is
  reproducible.

### 14.3 Campaign manager
```
campaign = {
  "name": "band_400_700",
  "puzzles": range(400, 701),
  "seed_source": "submissions/best.csv",
  "pipeline": [construction, quench, hop(200 basins), contact_lp, gate],
  "budget_s": 120,        # per puzzle
  "output": "submissions/campaign_band_400_700.csv"
}
```
* Resume from checkpoints after a crash.
* Emit a per‑puzzle JSON: `L_start`, `L_end`, basins tried, basins that hit a new
  best, time spent. This drives the *allocation* decision (which `n` deserve
  more budget).

### 14.4 Portfolio & submission hygiene
* `submissions/best.csv` is the rolling champion; every campaign merges into it
  only through the A01 gate.
* Keep a hash + timestamp + provenance per puzzle.
* **Daily submission budget is 10.** Reserve one slot per day for the current
  champion, and only spend slots on validated merges.
* Log every submission's public score in `boards/` to detect overfitting to the
  public LB (the LB is per‑puzzle, so public ≈ private; risk is low).

### 14.5 Cheap wins before any search
1. Apply the **closure running‑minimum** (A10) — fills many holes for free.
2. Apply **global rotation** (A11) to every puzzle — minutes.
3. Run **A08 contact LP** on the 547 already‑improved puzzles — hours, and it
   strictly improves or leaves them unchanged.

## Expected gain

* 14.1–14.2: 10–30× effective throughput → converts directly into A07's yield.
* 14.5: `0.002–0.01` before any heavy search.

## Cost

Low to medium; mostly refactoring, and it pays for itself immediately.

## Risks

* Over‑engineering before the solvers exist. Build 14.1 and 14.3 first, defer
  the rest.
* Memory: `n = 1000` layouts are tiny; the risk is retaining many basins —
  store only `L` and the best layout.

## Plan

* `src/core/hash.py`, `src/core/numba_geom.py` — the fast primitives.
* `src/core/worker.py` — one‑puzzle runner with checkpointing.
* `src/core/campaign.py` — config‑driven driver + merge + gate.
* `tools/submit.py` — validates, checks the daily quota, uploads via the Kaggle
  CLI, records the score.

## Decisions needed from you

1. Worker count: 16 (safe) or 20 (max)? RAM is ample either way.
2. Do you want automatic submission (script uploads the champion daily), or
   manual approval each time? I recommend manual for the first week.
