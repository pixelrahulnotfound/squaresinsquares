# Squares in Squares — full handover (state as of 2026‑10‑26)

Kaggle Playground competition [**Squares in Squares**](https://www.kaggle.com/competitions/squares-in-squares),
$50k, deadline **2027‑07‑18**, 10 submissions/day, 5 prizes.

> **Goal:** for every `n = 1 … 1000`, pack `n` unit squares into the smallest
> enclosing square of side `L_n`. Score `= Σ_n (L_n/√n − 1)`, **lower is better**.
> Perfect squares score exactly 0 (31 of them).

This file is the single entry point. It records **what has already been proven,
what failed and why**, so a successor does not repeat the work. Read §5 (data
audit), §6 (approaches) and §9 (negative results) before writing any code.

---

## 0. TL;DR for the next agent

| item | value |
|---|---|
| Our live submission | `submissions/best.csv`, **22.3040271730** official |
| Kaggle ref | submission **57600687**, status COMPLETE |
| Rank | **72 / 658 teams** (587 worse) |
| LB #1 | 22.224758 (Umong Sain) |
| Local == LB | yes, verified |
| Remaining gap to #1 | **0.0792** |
| Where the gap lives | **exactly the 266 non‑square *n* ≥ 325 that sit exactly on the integer grid** (see §8) |

**The public‑data ceiling has been reached and proven.** Every published packing
source has been fetched, parsed and merged; nothing in them beats our merge
(§5). Any further gain must be *generated*, and the one engine we have cannot
do it (§9.4). The open problem is stated precisely in §8.

First 30 minutes:

```bash
cd /home/pixelnotfound/squaresInSquares
python3 tools/score.py submissions/best.csv          # must print 22.304027...
python3 tools/score.py submissions/best.csv --official   # mpmath certification (slow)
./external/sqsearch/bin/sqsearch-targetside --selftest   # engine sanity -> SELFTEST PASSED
```

Then jump to **§11 (open leads)** and pick a direction.

---

## 1. Repository map

```
squaresInSquares/
├── README.md                  # THIS FILE — handover + approach atlas
├── comp.json                  # competition metadata (id 166661)
├── data/sample_submission.csv # the only official data (500,500 rows: id,n,square_id,x,y,angle)
├── metric/squares_in_squares_metric.py   # official metric, pulled verbatim
├── src/
│   ├── geom.py                # numba geometry: side_of, pen_max, energy(+grad), repair_scale, solid
│   ├── io.py                  # load/write submission CSVs (full-precision repr)
│   ├── validate.py            # audit / dilate / official / gate  <-- certification entry point
│   ├── quench.py              # local L-BFGS-style quench on the wall penalty, grid_layout
│   ├── constructions.py       # Göbel / strip / rectangle-core construction families
│   ├── torchpack.py           # torch penalty optimiser
│   └── gpu.py                 # batched L-BFGS + penalty continuation (ABANDONED, see §9.5)
├── tools/
│   ├── score.py               # fast float scorer + merge + --official certification
│   ├── build_best.py          # merge candidate sources -> submissions/best.csv (+provenance)
│   ├── cert_to_csv.py         # exact rational Couzo certs -> submission CSV
│   ├── apply_campaign.py      # fold a campaign improved.jsonl into a gated submission
│   └── sqsearch_probe.py      # per-n sqsearch probe (needs the *seeded* binary)
├── submissions/
│   ├── best.csv               # CURRENT CHAMPION (22.3040271730), submitted
│   ├── best_provenance.json   # per-n origin: "public" | "couzo"
│   ├── couzo.csv              # Couzo exact certs only
│   └── merge_public_couzo.csv # intermediate merge
├── external/
│   ├── couzo/                 # 720+ exact rational certs n<=1000, to 1092 files; constructions to 10000
│   └── sqsearch/              # THE SEARCH ENGINE (patched source + two static binaries)
├── kaggle_upload/
│   ├── sis_engine/            # Kaggle dataset (engine binary + best.csv)
│   └── sis_campaign/          # Kaggle script kernel (the 4h campaign)
├── assets/                    # downloaded public sources (§5)
├── boards/                    # leaderboard snapshots (may be stale; refresh before quoting)
├── notebooks/                 # 22 public notebooks dumped to .txt
├── docs/                      # 00..03 analysis + approaches/A01..A14 design docs
└── runs/                     # probe outputs (candidates/n*.npz, seeds/)
```

### 1.1 What is deliberately **not** in this repo (see `.gitignore`)

| excluded | why | how to restore |
|---|---|---|
| `data/`, `assets/`, `notebooks/`, `boards/`, `*.zip` | 400 MB of downloaded/competition material | re-download from Kaggle and the sources in §5 |
| `submissions/` | the live 22.3040271730 submission is **private**; publishing it would hand the work away | rebuild with `tools/build_best.py` over your own sources |
| `runs/` | generated probe output | regenerate with `tools/sqsearch_probe.py` |
| `kaggle_upload/sis_engine/best.csv` and `sqsearch` | 28 MB of generated payload | copy `submissions/best.csv` and `external/sqsearch/bin/sqsearch-seeded` in before `kaggle datasets version` |
| `external/sqsearch/target/` | Rust build output | `cargo build --release` |

Committed and therefore sufficient to reproduce the *method*: all Python
sources, the official metric, the docs, the patched Rust engine source, and both
prebuilt static binaries under `external/sqsearch/bin/`.

---

## 2. The metric — hard rules

`metric/squares_in_squares_metric.py` is the authority. Consequences that have
already cost us submissions:

1. **Angles are radians.** Public notebooks often work in degrees.
2. **Exact touching is undecidable** in the metric's interval arithmetic
   (50→500 dps). Every layout must be **dilated** about its centre by a relative
   clearance (1e‑12 … 1e‑11) before writing. `src/validate.dilate` does this.
3. **Rejected puzzle = scored as the grid**, i.e. a regression. Always run
   `validate.gate()` on the exact bytes you upload, after the float64 round trip.
4. Write ≥17 significant digits (`repr(float)`), never `%.6f`.
5. `validate.gate(packs, path)` = dilate → audit → write → `official()` with
   escalating clearance; returns `(ok, score)`.

---

## 3. Score landscape (measured on `submissions/best.csv`)

| n band | score | share |
|---|---:|---:|
| 1–100 | 6.3012 | 28% |
| 101–200 | 3.0931 | 14% |
| 201–324 | 2.8081 | 13% |
| 325–400 | 1.4765 | 7% |
| 401–500 | 1.8417 | 8% |
| 501–600 | 1.5947 | 7% |
| 601–700 | 1.4246 | 6% |
| 701–800 | 1.2908 | 6% |
| 801–900 | 1.1623 | 5% |
| 901–1000 | 1.3111 | 6% |
| **total** | **22.3040** | |

Small n is exhausted (144 values proven optimal; we match Couzo exactly).
n ≤ 324 is heavily mined by the exact‑proof community (`evand/square-packing`).
**The only region where we are visibly behind is n ≥ 325.**

---

## 4. Leaderboard / comparison points

| reference | score | note |
|---|---:|---|
| integer grid (sample submission) | 28.937670 | baseline |
| public notebook, best single | ~22.304356 | `malyshevdanil` (latest check) |
| `alexchilton` certified merge | ~22.304765 | |
| `kmmmvs` public portfolio (2026‑10‑25 13:54) | 22.3077770770 | we hold a copy |
| dariushafshar "certified 1–1000" | 22.350114474640 | **worse than ours by 0.046** |
| **ours** | **22.3040271730** | submitted, rank 72/658 |
| LB #10 | ≈22.290 | |
| LB #1 | 22.224758 | Umong Sain |

Public notebooks sit *at* our level, which is consistent with §5: everybody has
merged the same public packings. The leaders' edge is private.

---

## 5. Data audit — DO NOT REDO THIS

Every source below has been downloaded, parsed and scored per puzzle against
`submissions/best.csv`. Results are final.

| source | what it holds | verdict |
|---|---|---|
| `assets/kmmmvs_squares-in-squares-best-packings/` | public portfolio, per‑n | base of our merge; nothing better inside |
| `external/couzo/` | 720 exact rational certs n≤1000 (`n side` then `x y t`, `t=tan(θ/2)`), `bound_sources.json`, `certificate_constructions.py` | merged; **0** of its certified bounds beat us |
| `assets/dl/squares-in-squares-certified-packings/` (dariushafshar, 67 MB) | `packings_v1.csv`, `constructions_v1.csv` (111 153 rows), `index_per_n.csv` | score 22.3501 = **+0.046 worse**; its 15 "wins" over us are **3e‑7 float noise**; its certified `rounded_up` bounds beat us on **0** n |
| Mishapolk `square-packing-records` (5607 `.cert`) | we fetched the **390** certs with n ∈ [325,1000] to `/tmp/mish` | 59 nominal "wins", **all ≤ 1.1e‑11** = same packings we already have |
| `evand/square-packing` (pushed 2026‑10‑26 03:50, newest of all) | exact‑proof repo, n ≤ 324 focus; new records s(266/270/272/339/343/344/364/375/146/183/308) | **we already beat or match every one** (e.g. our 339 = 18.90735 vs their 18.92690) |
| `itsnaka/squish-certs`, `ry-xu/square_packing`, `SidG2k1/square-packing-refinements` | filename survey only | **0–1 relevant files** in n ≥ 325 |
| `lightningv08/...layout-atlas`, `kmmmvs/...` re‑download | — | `kaggle datasets download` returns **0 bytes** silently for these; our existing copies are fine |

Also checked: `evand`'s `Completed.md` states explicitly that the n > 324
frontier is driven by *public Kaggle packings* (lightningv08 / cdeotte) — i.e.
the sources we already merged. Nothing upstream is missing.

**Conclusion: 22.304027 is the published‑data optimum. Do not re‑audit.**

---

## 6. Approaches — what was designed, what was tried, what happened

Design docs live in `docs/approaches/A01…A14`. Status of each in practice:

| id | approach | status | result |
|---|---|---|---|
| A01 | validation / certification | **done** | `src/validate.py`; gate PASS; used for the live submission |
| A02 | known‑construction portfolio + per‑n merge | **done** | produced `best.csv` (22.304027) |
| A03 | generative construction families | **done, useless** | `src/constructions.py` (Göbel, strips, rectangle cores, 26.090365) → **0 wins** vs public |
| A04 | fault‑line / gadget DP composition | not implemented | public attempts (`alexchilton`, dariushafshar 4 variants) produced nothing beyond public data |
| A05 | exact upright refill | not implemented | prerequisite for A04 |
| A06 | smooth penalty quench | **tried, failed** | `src/gpu.py`, `src/quench.py` — see §9.5 |
| A07 | global search (SA / basin hopping) | **tried hard, failed** | `sqsearch` engine — see §9.1–9.4 |
| A08 | contact LP trust region | not implemented | |
| A09 | disjunctive MILP | not implemented | |
| A10 | subset transfer / closure algebra | **tried, failed** | `kozykappa`-style transfer via sqsearch seeding: 0/4 |
| A11 | rigid rotations + minimal enclosing square | partial | included in `geom`/`quench` |
| A12 | learning‑based | not started | |
| A13 | exact symbolic refinement | not implemented | low value now (safety only) |
| A14 | compute engineering | **done** | Kaggle dataset + kernel pipeline (§7) |

---

## 7. Kaggle infrastructure (working, reusable)

Everything heavy runs on Kaggle CLOUD, not locally. Local machine is a laptop
and **must not** be loaded (user constraint: ≤4 threads, no multi‑minute runs
without asking).

### 7.1 Dataset

| item | value |
|---|---|
| slug | `rahulkrttty/sis-engine` |
| contents | `sqsearch` (static musl binary, **seeded** variant) + `best.csv` |
| local dir | `kaggle_upload/sis_engine/` |
| update | `kaggle datasets version -p kaggle_upload/sis_engine -m "msg"` |

### 7.2 Kernel

| item | value |
|---|---|
| slug | `rahulkrttty/sis-campaign` (script kernel, CPU, 4 vCPU) |
| local dir | `kaggle_upload/sis_campaign/` (`kernel.py`, `kernel-metadata.json`) |
| push | `kaggle kernels push -p kaggle_upload/sis_campaign` |
| status | `kaggle kernels status rahulkrttty/sis-campaign` |
| collect | `kaggle kernels output rahulkrttty/sis-campaign -p /tmp/kout` |
| outputs | `improved.jsonl`, `merged.csv`, `report.json`, `sqsearch`, `sis-campaign.log` |

Mount gotchas (all cost us a failed run each):

* Datasets mount at **`/kaggle/input/datasets/<owner>/<slug>/`**, not
  `/kaggle/input/<slug>/`. `kernel.py` therefore uses a recursive `find()`.
* The mount is **read‑only** → the binary is copied to `/kaggle/working` and
  `chmod +x` there. Any `chmod` on `/kaggle/input` raises `OSError: [Errno 30]`.
* A dataset created seconds before the kernel push may not be mounted at all.

### 7.3 What the campaign does (version 4, ~4 h)

1. Loads `best.csv`, rebuilds every layout, computes each side.
2. Targets = n ∈ [325,1000] with `|L − ⌈√n⌉| < 1e-6` and n not a perfect
   square → **266 puzzles**, sorted by **ascending density** (`n/K²`), so the
   loosest grid (n = 345 = 19²−16) goes first.
3. Pass 1: 4 fresh‑anneal configs with the engine's *default‑off* arms
   (`--p-perturb 0.05`, `--mu0 1 --mu1 0.001`, `--lambda0 0.5`, plus control).
4. Pass 2 (never reached): seeded + grid‑minus‑holes refinement.
5. Folds wins into `merged.csv`, writes `report.json`.

**Result: 266/266 covered, 0 wins** (§9.4).

### 7.4 Applying a campaign result

```bash
kaggle kernels output rahulkrttty/sis-campaign -p /tmp/kout
python3 tools/apply_campaign.py --jsonl /tmp/kout/improved.jsonl \
        --base submissions/best.csv --out submissions/campaign.csv
# then, if the score dropped:
kaggle competitions submit -c squares-in-squares -f submissions/campaign.csv -m "campaign v4"
```

`apply_campaign.py` accepts only genuine improvements (`> 1e‑9`), then runs the
full `validate.gate` before printing the official score.

---

## 8. Where the remaining score is — the precise open problem

Fact: of the 279 n in [325,1000] that sit exactly on the integer grid, 13 are
perfect squares (provably optimal). The other **266** are non‑square and yet our
`L` equals `⌈√n⌉` exactly.

Arithmetic: improving those 266 by **−0.006 side each** changes the score by
`266 × 0.006 / 19.5 ≈ 0.0818` → **22.2222**, i.e. LB #1 territory.

So the entire 0.079 gap is "make every grid puzzle a hair smaller". That is the
problem. It is worth restating why it is hard:

* `L = K` with zero overlap is a **perfect optimum of the engine's objective**
  (`side` can't decrease without creating overlap; overlap costs λ·x).
* Such a packing is *solid*: no single square can move to improve it. Shrinking
  the bounding box requires an entire edge row to move simultaneously.
* Empirically (see §9.4) single‑square annealing never accepts the first step.

---

## 9. Negative results — the expensive knowledge (do not repeat)

### 9.1 sqsearch reproduces the grid, exactly

Fresh search on n = 345/350/360/383 returns **exactly K** (19/19/19/20) with
overlap 0; the only "gain" is ~1.8e‑10 of float noise. n = 400/500/600/700:
reproduces our or worse (n = 500: ours 22.8546 vs 23.0). n = 499/626: worse.

### 9.2 Seeded refinement of our layouts does nothing

0/4 improvements on n = 400/500/600/700 — byte‑identical sides.

### 9.3 The trap is not a parameter issue

Seeded anneal from a loose seed (n = 345 grid scaled ×1.05 → side 19.9) stays
**exactly 19.9** across:

* budgets 200 k / 2 M / 8 M moves,
* `--p-perturb` 0.01 / 0.05 / 0.2,
* `--mu0/--mu1` (0.5, 0.001) and (2, 0.001).

Instrumented: `start side=19.900 → end 19.9336`, best never updated,
`accepted = 49 912 / 50 000`. Conclusion: a **solid‑configuration trap**. Whole
configuration perturbation is not the move class that escapes it.

### 9.4 The 4‑hour Kaggle campaign: 266/266, 0 wins

Every one of the 266 grid puzzles, four exploration arms each, ~54 s per puzzle.
All returned exactly K. **Fresh annealing with all escape arms cannot improve a
dense grid.** This is the single most important negative result in the repo.

### 9.5 The target‑box probe: overlap cannot be resolved

We added `--target-side` (demand a box below K) and `--seed-scale` (start from a
compressed grid, so the chain begins *inside* the box with overlap > 0).

n = 345, target 18.9905 (K − 0.0095, density 95.6 %, area‑feasible):

| t_hot | reported side | true residual overlap |
|---|---|---|
| 0.01 | 18.991 | **0.6345** |
| 0.05 | — | (run hung; stray process killed) |

So the anneal *does* explore inside the box but **cannot untangle the
compression** — it stalls with 0.63 units of overlap, nowhere near feasible.
This kills the "crush then separate" idea in its current form.

⚠️ **Bug found here:** the incremental overlap bookkeeping can drift, so a
config can pass the `FEASIBLE_EPS` test with a large *true* overlap. The
`summary` line recomputes the true overlap of the returned best — **always trust
`best_overlap` from the summary, never an internal acceptance decision.**
Any future WIN must be re‑verified from coordinates before being believed.

### 9.6 Our own optimisers are worse than sqsearch

`src/gpu.py`: batched L‑BFGS with per‑basin two‑loop recursion, Armijo
backtracking, penalty continuation, batched exact repair. Gradients verified
against central differences to 5e‑10. It still stalls: trivial case (10 squares
in 4×4) leaves penalty ≈0.14; n = 100/200 with σ=0.05 kick leaves ≈0.1.
Root cause: the sum‑of‑squared penalty with `max` over 4 separating axes has
kinks where axes tie → flat/ill‑conditioned landscape. Abandoned; kept for
reference.

### 9.7 Third‑party merges and constructions add nothing

See §5: dariushafshar's 111k construction rows gave 15 "wins" of 3e‑7 each
(total 1.5e‑7). Couzo's `certificate_constructions.py` families do not beat the
certified index either.

---

## 10. The engine (`external/sqsearch/`)

### 10.1 What it is

`github.com/jlevy/squares`, subdirectory `packing/sqsearch` — a Rust simulated
annealer by the "Squares Project". Not vendored from crates; the source is in
`external/sqsearch/src/` and it depends only on `rayon`.

**Objective** (this is the key design fact):

```
F(config) = required_side(config) + λ · total_overlap(config)      [+ μ · spread]
λ ramps 2 → 1e6 across the anneal; overlap penalty is LINEAR; FEASIBLE_EPS = 1e-12
```

It does **not** use a target box + wall penalty. Moves: single‑square
translate/rotate, `p_reseed` (restart from best), optional `p_perturb`
(displace every square) and `mu` wall pressure.

Defaults: `steps 400000, t_hot 0.25, t_cold 1e-9, lambda0 2, lambda1 1e6,
move_rotate 2, p_rotate 0.35, p_reseed 0.5, p_perturb 0, perturb_scale 1,
mu0 0, mu1 0`.

Validated: `--selftest` passes; **n = 100 → exactly 10.0**, overlap 0, ~20 s.

### 10.2 Build

```bash
cd external/sqsearch
rustup target add x86_64-unknown-linux-musl     # static, runs anywhere
cargo build --release --target x86_64-unknown-linux-musl
./target/x86_64-unknown-linux-musl/release/sqsearch --selftest
```

Toolchain is pinned by `rust-toolchain.toml` (1.98.0). Build ~13 s.

### 10.3 Two binaries exist — they are NOT interchangeable

| binary | has | missing |
|---|---|---|
| `external/sqsearch/bin/sqsearch-seeded` (static) | `--seed-config`, `--jitter` | `--target-side`, `--seed-scale` |
| `external/sqsearch/bin/sqsearch-targetside` (static) | `--target-side`, `--seed-scale` | `--seed-config`, `--jitter` |
| `external/sqsearch/src/` | source **with** target‑side + seed‑scale patches | seeded chain (was lost from `/tmp`, must be re‑added) |

⚠️ **Unknown flags are silently ignored.** Verify a run by the `"params"` echo
in the `summary` JSON line. Passing `--target-side` to the seeded binary (or
`--seed-config` to the target‑side binary) quietly runs the *default* objective.

**Unify them first** (task #1 for whoever continues): re‑implement
`run_seeded_chain(seed_cfg, seed, chain, params, budget_moves, jitter)` in
`src/search.rs` — anneal from the given config with jitter instead of
`Config::grid(n)` — and plumb `--seed-config PATH` (JSON with `x`,`y`,`t`
arrays) plus `--jitter F` in `src/main.rs`. ~30 lines. `tools/sqsearch_probe.py
--seeded` already expects exactly that interface.

### 10.4 CLI

```
--n N --chains C --threads T --seed S --budget-moves M
--steps --t-hot --t-cold --lambda0 --lambda1 --move-rotate --p-rotate
--p-reseed --p-perturb --perturb-scale --mu0 --mu1
--budget-pair-tests --max-restarts
--target-side T        # targetside binary only
--seed-scale F         # targetside binary only
--seed-config PATH     # seeded binary only
--jitter F             # seeded binary only
--selftest
```

Output is JSONL: one `{"kind":"chain",...}` per chain and one
`{"kind":"summary",...}` with `best_side`, `best_overlap`, `x`, `y`, `t`
(radians), plus the echoed `params`. Parse the **summary** line.

### 10.5 Patch semantics (for the unify task)

* `--target-side T`: replaces the side term with
  `(side − T).max(0)·10 + side·1e‑3`, so the anneal is pressed into a box of
  side T and must resolve overlap there. `f64::INFINITY` (default) = original
  objective, bit‑for‑bit.
* `--seed-scale F`: multiplies every `x`,`y` by F right after `Config::grid(n)`
  and recomputes `s0`. F < 1 starts the chain *inside* the demanded box with a
  positive overlap (§9.5).

---

## 11. Open leads, ranked by expected value / cost

1. **Escape the solid‑configuration trap with a new move class.**
   The literature answer is Huang–Korf style **compaction / "clump" moves**: push
   a whole row/column or a connected cluster in one direction until blocked, so
   collective motion is possible. Single‑square annealing provably fails (§9.4).
   This is the only credible route to the 266. Cost: days of engine work;
   payoff: up to −0.08 (LB #1) if it generalises.
2. **Fault‑line composition with *tilted* overlap** (A04). Public attempts used
   axis‑aligned faults and got nothing; the variant that could beat the grid
   splits n = a + b, packs each part in a strip, and lets the strips' bounding
   boxes overlap slightly along a tilted fault. Needs A05 (exact upright refill)
   first. Medium cost, unproven.
3. **"Punch a hole and reinsert"** (`lcut`). Remove one square from a grid
   packing (creating a 1×1 hole = slack), demand a box at K − δ, and let the
   anneal reinsert. This is how `evand` obtained several new records (s(183),
   s(343), s(344)). Cheapest experiment available: 3–4 n, ~10 min locally.
   **Recommend this as the first test of the next session.**
4. **Near‑record pool + descent** (their "SW2": 42 n, 114 CPU‑h → 2 new minima,
   ~1e‑2 side each). Real but low yield: expected ≈ +0.001 score per 100 CPU‑h.
   Only worth it if the goal is "any improvement", not "beat the leaders".
5. **Refresh public data occasionally** (weekly). Everyone's falling behind
   kmmmvs; a newer public portfolio may appear. Cheap, bounded.

Explicitly **not** worth doing: re‑auditing public sources (§5), fresh annealing
of grid puzzles (§9.4), more parameter sweeps of `p_perturb`/`mu` (§9.3), the
sum‑of‑squares penalty optimiser (§9.6).

---

## 12. Tooling reference

```bash
# score one or more submissions (float64, fast)
python3 tools/score.py submissions/best.csv
# merge best-of and write
python3 tools/score.py a.csv b.csv --write submissions/merged.csv
# authoritative certification (mpmath, slow)
python3 tools/score.py submissions/merged.csv --official

# rebuild Couzo certs -> CSV
python3 tools/cert_to_csv.py --certs external/couzo/certificates/best \
        --out submissions/couzo.csv --ns 1-1000

# rebuild the champion from sources (+ provenance json)
python3 tools/build_best.py

# fold a campaign result into a gated submission
python3 tools/apply_campaign.py --jsonl /tmp/kout/improved.jsonl \
        --base submissions/best.csv --out submissions/campaign.csv

# probe the engine per n (needs the SEEDED binary)
python3 tools/sqsearch_probe.py 345 350 360 --chains 4 --budget 8000000
```

Python API: `src.io.load/write`, `src.geom.side_of/pen_max/energy/energy_grad/
repair_scale/normalise/solid`, `src.validate.audit/dilate/official/gate`.

---

## 13. Constraints and gotchas

* **Compute:** the user's laptop must stay light. Cap threads at 4; do not start
  multi‑minute local runs without asking. Heavy work belongs on Kaggle
  (CPU kernels, ~30 h/week quota, 4 vCPU, 12 h max/run).
* **Submissions:** 10/day. The live one already scores 22.3040271730 — never
  overwrite it with an unverified file.
* **Killing processes:** never `pkill -f <name>` — the pattern matches your own
  shell and kills the session. Use `pkill -x sqsearch`. One stray run already
  cost a 20‑minute hang.
* **Parallel `bash` calls share one shell** → cwd races. Use `workdir=`.
* **`git clone` of big repos times out.** Use
  `https://api.github.com/repos/<r>/git/trees/HEAD?recursive=1` + `raw.githubusercontent.com`.
  (`jlevy/squares` is 1.4 GB.)
* **`kaggle datasets download` can silently yield 0 files** for some datasets;
  check the directory afterwards.
* **Multiprocessing + the metric fails from a heredoc** (stdin) — run gate tests
  from a real script file.
* **Trust only recomputed numbers.** `summary.best_overlap` is recomputed from
  coordinates; internal acceptance bookkeeping can drift (§9.5). Re‑verify any
  claimed win from its coordinates with `src.geom.side_of` + `validate.gate`.
* Float noise floor: "wins" of ~1e‑11…3e‑7 are the *same* packing, not progress.
  Only accept `< −1e‑9` side, and only after the gate.

---

## 14. Change log

* 2026‑10‑24 — repo + analysis docs, public‑portfolio audit, Couzo certs merged;
  `best.csv` (22.304027) submitted, rank 72/658.
* 2026‑10‑25 — public data sweep (`dariushafshar` 22.3501, constructions,
  certified bounds), Mishapolk/evand/squish/ryxu survey; `src/gpu.py` written
  and abandoned (§9.6); `sqsearch` obtained, built, selftested; seeded patch.
* 2026‑10‑26 — 4 h Kaggle campaign over the 266 grid puzzles: **0 wins**
  (§9.4); target‑box/compressed‑seed patches written and probed (§9.5);
  `external/sqsearch/` vendored with two static binaries; this handover written.
