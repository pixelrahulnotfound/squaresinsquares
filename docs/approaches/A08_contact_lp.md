# A08 — Trust‑region contact LP (free centres **and** angles)

**Priority: high. Best gain‑per‑second of any polish method; it is the engine
behind `malyshevdanil`, `kozykappa`, `kirilltym` and `avazaku/III`.**

## Why

Once a layout is *solid*, the only way to improve is to move along the
constraint boundary. Non‑overlap between two squares, given a **chosen**
separating axis, is a linear constraint in the centres:

```
a·(c_i − c_j) ≥ R(θ_i − θ_j)
```

Fix the contact graph (which pairs touch, on which axis) and the whole problem
becomes: minimise the box side subject to half‑planes — an **LP**. Solving it
with exact arithmetic finds the true local optimum of that combinatorial
"contact pattern", far more precisely than any gradient method.

## Formulation

Variables: `c ∈ R^{2n}`, `t ∈ R` (side), optionally `θ ∈ R^n` fixed.

```
minimise   t + ε Σ_i (u_i⁺ + u_i⁻)            # ε = 2e-6, soft walls
s.t.       for each contact (i,j,a):   a·(c_i − c_j) ≥ R(θ_i − θ_j)
           c_i·e_x + h_i ≤ t + u_i⁺ ,  c_i·e_x − h_i ≥ −u_i⁻   (and y)
           |c_i − c_i⁰| ≤ r ,   |θ_i − θ_i⁰| ≤ ρ   (trust region)
```

`h_i = ½(|cosθ_i| + |sinθ_i|)`.

### The two details that matter

1. **Kinks.** At `θ_i − θ_j ≈ 0` the constraint is non‑differentiable in the
   angles. Use **one‑sided derivatives**: add both linearisations from the left
   and from the right (i.e. the two adjacent axes) rather than a single one.
2. **Trust region.** Without it the LP jumps to a different contact pattern and
   the linearisation is meaningless. `r ≈ 0.02·L`, `ρ ≈ 0.01` rad; on success
   `r ← 1.3r`, on failure `r ← 0.4r`.

### Free‑angle LP

Instead of fixing `θ`, linearise `R` around `θ⁰` and solve for `(c, θ, t)`; the
LP stays linear because `R`'s gradient is piecewise constant in `δ`. Iterate.

## Algorithm

```
def contact_polish(layout, rounds=64):
    best = layout
    for k in range(rounds):
        G = contact_graph(best)               # pairs with pen ≈ 0 and their argmax axis
        sol = linprog(minimise t, constraints=G + walls + trust)
        cand = repair(sol.c, sol.theta)       # A01 dilation
        if L(cand) < L(best) - 1e-12 and feasible(cand):
            best = cand;  r *= 1.3
        else:
            r *= 0.4
    return best
```

## Expected gain

* `~4e-5` per successful pass per puzzle; 3–5 passes with different trust
  schedules (as `kozykappa` does) multiplies that.
* Across the 400–1000 band: **0.005–0.02**.
* Crucially, it makes A07's basins *pay*: a hop that finds a new contact pattern
  is worth nothing until the LP resolves it exactly.

## Cost

Medium: one LP is `2n+1` variables, `O(n)` constraints → milliseconds with
HiGHS. Thousands of LPs per hour.

## Risks

* The contact graph from a float layout is noisy at `1e-12`; use a tolerance
  band (`|pen| < 1e-7`) and include both axes for ambiguous pairs.
* Without trust regions the LP oscillates; without the `ε` term the walls are
  infeasible by roundoff.
* Angles must be *frozen* consistently; a free‑angle LP that drifts past a kink
  produces an invalid layout.

## Plan

* `src/lp/contact.py` — graph extraction, constraint assembly, `linprog`.
* `src/lp/free_angle.py` — one‑sided derivative variant.
* `src/lp/trust.py` — the schedule driver, shared with A04's seam repair.

## Decisions needed from you

1. Contact graph from **penetration tolerance** or from an explicit
   **contact‑pattern enumeration** (try several patterns per pass, keep best)?
   The second is stronger and ~5× more expensive.
2. Free angles or fixed angles for the main pass? (Free is better but needs the
   kink handling; I recommend free with `ρ = 0.01`.)
