# A09 — Disjunctive contact MILP / contact surgery

**Priority: medium. Expensive per puzzle, but it is the only way to *change* the
contact pattern systematically rather than by luck.**

## Why

A08's LP is local: it improves a fixed contact graph. The real gains come from
switching to a *different* graph (which pairs touch, on which side). That is a
combinatorial choice, and the natural model is a disjunctive program:

```
for each pair (i,j):  choose exactly one separating axis a ∈ axes(i,j)
                      a·(c_i − c_j) ≥ R(θ_i − θ_j)
```

Introduce binaries `z_{ij,a}` and big‑M from the trust radius:

```
a·(c_i − c_j) ≥ R(θ_i − θ_j) − M_{ij}(1 − z_{ij,a})
Σ_a z_{ij,a} = 1
|c_i − c_i⁰| ≤ r,   |θ_i − θ_i⁰| ≤ ρ
```

`M_{ij} = 2R + 2r + margin`. Solve with HiGHS (`scipy.optimize.milp`).

`avazaku/II` uses exactly this (contact‑choice MILP → soft‑pressure L‑BFGS →
free‑angle LP) and reports it as the step that moved the notebook from 22.74 to
22.41.

### Contact surgery (avazaku/III)

Cheaper and cleverer: solve the **fixed‑angle LP** first, read the **shadow
prices** (duals) of the contact constraints. A high dual value means that
contact is what is holding the box open. Then spend the MILP budget only on the
top‑`k` binding pairs — "cut the seam that costs the most".

## Algorithm

```
def surgery(layout, budget):
    lp = solve_lp(layout)                     # A08
    duals = lp.ineqlin.marginals
    hot = top_k(|duals|, k=8..16)             # pairs to re-decide
    milp = solve_disjunctive(layout, hot_pairs=hot, time_limit=budget)
    return quench(free_angle_lp(milp))
```

## Expected gain

* `~5e-5` per successful surgery; `avazaku` reports ~0.33 total from the
  II+III stack over the construction baseline.
* On top of the current portfolio: **0.003–0.01**.

## Cost

High: MILP with `O(k·n)` binaries, seconds to minutes per puzzle. Reserve for
puzzles where A07/A08 have stalled, and for `n` where a big gain is plausible
(small `n`, or puzzles just above the grid).

## Risks

* Big‑M models are numerically nasty at `1e-12` tolerances; always re‑validate
  and repair.
* The trust region must be small or the MILP just re‑discovers a worse basin.
* Time limits make the result non‑deterministic — log the seed and the budget.

## Plan

* `src/milp/disjunctive.py` — model builder, hot‑pair selection, HiGHS driver.
* `src/milp/duals.py` — shadow‑price ranking from the LP.
* `src/milp/presolve.py` — pair pruning by distance (`> R + 2r` ⇒ no binary).

## Decisions needed from you

1. Is a hard dependency on HiGHS via `scipy.optimize.milp` acceptable? (Yes by
   default; no extra install.)
2. Budget per puzzle: 10 s, 60 s, or 300 s? Different `n` bands may want
   different budgets.
