# 01 — Score anatomy: where the points actually are

All numbers computed from `assets/kmmmvs_squares-in-squares-best-packings/submission.csv`
(the certified public portfolio, 2026‑10‑24, score **22.3077770770**).

## 1. Distribution of the portfolio

| fact | value |
|---|---:|
| puzzles submitted | 1000 |
| perfect squares (score 0) | 31 |
| still exactly on the integer grid | **453** |
| improved over grid | 547 |
| score | 22.3078 |

The 453 grid puzzles are the single biggest structural weakness of the public
portfolio. 144 of them are **provably optimal** (`k²`, `k²−1`, `k²−2`, plus a
few small cases), so ~300 are genuinely improvable — but the public notebooks
that generate constructions only cover the ones a construction happens to hit.

## 2. Band decomposition

| band | puzzles | at grid | improved | score | mean excess | score share |
|---|---:|---:|---:|---:|---:|---:|
| 1–50 | 50 | — | — | 4.1663 | 0.0833 | 18.7 % |
| 51–100 | 50 | — | — | 2.1349 | 0.0427 | 9.6 % |
| 101–200 | 100 | — | — | 3.0931 | 0.0309 | 13.9 % |
| 201–400 | 200 | — | — | 4.2846 | 0.0214 | 19.2 % |
| 401–700 | 300 | — | — | 4.8634 | 0.0162 | 21.8 % |
| 701–1000 | 300 | — | — | 3.7654 | 0.0126 | 16.9 % |
| **all** | 1000 | 453 | 547 | **22.3078** | 0.0223 | 100 % |

Sub-band detail:

| band | at grid | improved | score |
|---|---:|---:|---:|
| 1–100 | 64 | 36 | 6.3012 |
| 101–324 | 110 | 114 | 5.9012 |
| 325–1000 | 279 | 397 | 10.1054 |

`n > 324` is where **no published record table exists** and where 279 puzzles are
still plain grid. That is the most under‑served region relative to its weight.

## 3. The Couzo comparison — a free reality check

Francisco Couzo's exact register (mirrored in `external/couzo`) has certified
rational witnesses for 720 of the 1000 puzzles:

* Couzo is **strictly better on 112** puzzles, total gain **0.00375**.
* The public portfolio is **better on 198** (because it has been locally
  squeezed since).
* 410 are identical.

Merging the two gives **22.304027** — i.e. the public portfolio already contains
~99.5 % of what the entire public record registry offers. **There is no free
lunch left in public data.**

## 4. Sensitivity: what one unit of score costs

The per‑puzzle score is `L/√n − 1`. To reduce it by `δ`:

```
ΔL = δ · √n
```

| n | ΔL for δ = 1e‑4 | ΔL for δ = 1e‑3 |
|---:|---:|---:|
| 25 | 0.0005 | 0.005 |
| 100 | 0.001 | 0.010 |
| 400 | 0.002 | 0.020 |
| 1000 | 0.0032 | 0.032 |

So reaching the current LB #1 (22.2248) from the public portfolio requires
**0.083 total**, i.e. on the order of:

* ~830 puzzles improved by 1e‑4 each, **or**
* ~83 puzzles improved by 1e‑3 each, **or**
* 8 puzzles improved by 1e‑2 each.

The public leaderboard shows exactly this: leaders differ by 0.01–0.03, which is
hundreds of micro‑improvements, not one breakthrough.

## 5. Structural bounds

* Area: `L ≥ √n` ⇒ score ≥ 0.
* Proven optimum `L = ⌈√n⌉` for `n = k², k²−1, k²−2` (and `k²−3` for k ≤ 7).
  Those puzzles cannot be improved — **144 of 1000** are dead weight.
* For large `n`, best constructions give `L ≈ ⌈√n⌉ − f` where `f ∈ (0,1)` is the
  "fault gap". Empirically the public portfolio has `L = ⌈√n⌉ − 0.1…0.6` on
  improved puzzles, and `W = L² − n ≈ k` (see skarin's EDA).
* There is no known closed‑form lower bound tight enough to matter here; the
  practical ceiling is "what the best searchers can squeeze".

## 6. Conclusion for strategy

1. **Do not touch** the 31 perfect squares and the ~113 other proven optima.
2. **Small n (< 100)**: already at/near record; only worth an occasional exact
   MILP run. Max remaining ≈ 0.1–0.2 but hard.
3. **Mid n (100–324)**: records exist (Ellsworth) and the portfolio matches most;
   residual ≈ 0.05–0.1.
4. **Large n (325–1000)**: 279 grid puzzles + 397 squeezed ones; residual vs any
   achievable construction is large. **This is the battleground**, and it rewards
   parallel local search + composition, not cleverness.
