# 00 — Competition analysis

## 1. The problem

Given `n`, find centres `(x_i, y_i)` and rotations `θ_i` for `n` unit squares such
that no two interiors overlap, and the axis-aligned bounding square

```
L_n = max( max_i x-extent , max_i y-extent )
```

is minimal. There is no requirement that the container be axis-aligned in any
other sense — the metric simply measures the bounding box of the corner cloud.

Formally, with `h_i = ½(|cos θ_i| + |sin θ_i|)` the half-extent of square `i`:

```
L = max( max_i(x_i + h_i) − min_i(x_i − h_i),
         max_i(y_i + h_i) − min_i(y_i − h_i) )
```

so `L` is translation invariant and the container can always be normalised to
`[0,L]²`.

## 2. Data

The competition ships **only** `sample_submission.csv` (12.9 MB, 500 500 rows):

```
id,n,square_id,x,y,angle
0,1,0,0.5,0.5,0
1,2,0,0.5,0.5,0
2,2,1,1.5,0.5,0
...
```

* `n` ∈ 1…1000, exactly `n` rows per puzzle, `square_id` ∈ `0…n−1`.
* `angle` is in **radians** (the sample is all zeros, which hides this).
* `x, y` are centres of unit squares; the square spans `±0.5` before rotation.
* There is no train/test split — the "test" is all 1000 puzzles. This is a pure
  optimisation competition, not a learning competition.

`Σ_{n=1}^{1000} n = 500 500` rows exactly.

## 3. Metric (official, `metric/squares_in_squares_metric.py`)

```
score = Σ_{n=1}^{1000} ( L_n / sqrt(n) − 1 )
```

* Lower is better; `L_n ≥ sqrt(n)` by area, so each term is ≥ 0.
* Perfect squares `n = k²` reach exactly 0 with the integer grid.
* A puzzle that is missing, malformed, unparseable, or **infeasible** is charged
  `⌈√n⌉/√n − 1`, i.e. the trivial grid. So a broken improvement *costs* you.
* Feasibility is decided by **interval arithmetic** (`mpmath.iv`) at 50, 100,
  200, 500 decimal digits: two squares are separated only if some edge normal
  gives a *provably* non-positive overlap. Exact touching is **undecidable**,
  so genuinely optimal packings must be written with a tiny dilation.
* `L` is measured at 60 dps **from the submitted decimal strings**, then
  rendered to 50 decimals. More digits in the CSV = more precision in `L`.

### Practical implications

| trap | consequence | fix |
|---|---|---|
| degrees written as radians | squares rotate 57× too far → overlaps | always convert |
| exact touching | "Overlap undecidable at max precision" → grid penalty | dilate centres by `1+1e-12` |
| `%.6f` formatting | `L` rounded up by up to 5e‑7, and `-1e-7` slivers become overlaps | `%.17g` / `repr` |
| 1e‑9 overlaps from float noise | rejected puzzle | repair with uniform scale about bbox centre |
| omitting puzzles | they are charged the grid, not zero | always submit all 1000 |

## 4. What the score actually means

Because `L_n` is measured from the coordinates, the only thing that matters is
the bounding box. You can:

* **translate** freely (the metric is translation invariant),
* **rotate the whole layout** (and re-measure) — the bounding box of a rotated
  configuration can be smaller,
* **relabel** squares freely (the metric ignores `square_id` order).

Degrees of freedom: `3n` per puzzle, minus 2 translations and (if you want) 1
global rotation. Symmetry group size `n! · 4^n` — enormous, which is why the
landscape is full of equivalent basins and why stochastic search works.

## 5. Difficulty structure

* `n = k²` → 0, trivial (31 puzzles).
* `n = k²−1, k²−2` → provably `k` (Nagamochi 2005; `k²−1` re‑proved by Karakuş
  2026, `k²−2` by chelokot in Lean). `n = k²−3` proved only for k ≤ 7.
* `n = k² + r` with small `r` → the hard, high‑value band: you must beat the
  `k+1` grid by fitting `r` extra squares into a strip of height `< 1`.
* `n = 5, 10` (Göbel) and `n = 11` (Trump) are the classic tilted solutions;
  `n = 11` is still not proven optimal (best known `3.87708359…`, AI‑assisted
  proof of the value announced 2026‑09‑29).
* For `n > 324` there is **no published record table** — Ellsworth's catalogue
  stops at 324 plus a handful of larger cases. Above that, everyone (including
  the public portfolio) is using *constructions + local search*, so the space is
  open and the competition is genuinely winnable by brute compute.

## 6. Competitive landscape (2026‑10‑24)

| | score | what it is |
|---|---:|---|
| grid | 28.9377 | sample submission |
| public portfolio | 22.3078 | merge of ~26 public notebooks, certified |
| LB #1 | 22.2248 | private search |
| LB #10 | 22.2901 | |
| LB #100 (approx) | ~22.32 | |

The public/private gap is ~0.08. Because the public portfolio is *already* the
merge of every public notebook, **copying public work gains nothing**. The only
ways forward are:

1. **New constructions** (gadgets / cores / families) that no public notebook has.
2. **Massive local search** on the 400–1000 band (the only place with enough
   puzzles × enough slack).
3. **Composition**: reuse one improved sub‑packing across dozens of `n`.

That ordering is also the cost/benefit ordering, and it drives the approach
catalogue.
