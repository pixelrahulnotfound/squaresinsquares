# A05 — Exact upright refill (staircase greedy + MIS‑MILP)

**Priority: high. It is the sub‑solver that makes A04 pay off.**

## Why

A fault‑line gadget (A04) splits the box into a **tilted core** and a leftover
region that is a *staircase* (a down‑set of integer cells) plus a thin sliver.
Filling that leftover optimally with **upright** squares is a small, discrete,
*exactly solvable* problem — and every extra square refilled is a square that
does not have to be packed by search.

`cdeotte`'s `fill_rect` is this idea; it is the step that took the FTMC notebook
from ~22.35 to 22.3116.

## Method

### 1. Staircase greedy is optimal

Let the free region be a Ferrers/down‑set `S ⊆ ℤ²` (if cell `(i,j) ∈ S` then
`(i',j') ∈ S` for `i'≤i, j'≤j`). Placing unit upright squares with the
**bottom‑left rule** (always the lowest, then leftmost, free cell) packs
`max{|S'| : S' ⊆ S, S' is a union of unit cells that can be covered by disjoint
squares}` — and for a down‑set the greedy is optimal by an exchange argument:
any optimal packing can be "pushed down‑left" without losing a square.

So for a pure down‑set, refill is `O(|S|)` and exact.

### 2. Slivers: MILP / MIS

The leftover is rarely a clean down‑set — the tilted core leaves a zig‑zag
boundary. Then:

```
maximise  Σ_c y_c
s.t.      y_c ∈ {0,1}                    # upright square anchored at cell c
          y_c + y_{c'} ≤ 1               # for every overlapping anchor pair
          y_c = 0                        # anchors that hit the tilted core
```

i.e. a **maximum independent set** on the anchor conflict graph, which is an
interval/unit‑disk‑like graph → small and fast with `scipy.optimize.milp`
(HiGHS) or a branch‑and‑bound over connected components.

### 3. Split parameter

The gadget has a free split `k` = number of squares assigned to the tilted core;
`w·h + Δm − k` go to the upright refill. Enumerate `k` over a small range and
keep the best feasible pair — the tilted core is solved by A06/A07 for each `k`.

## Algorithm

```
def refill(region, core):
    S = cells(region) \ cells(core)
    y = greedy_bottom_left(S)          # fast, optimal for down-sets
    if boundary_is_zigzag(S):
        y = mis_milp(S)                # exact for the general case
    return y

def gadget(w, h, f, k):
    core = optimise_tilted(w, h, f, k)         # A06/A07
    fill = refill(region(w,h,f), core)
    return core + fill, L
```

## Expected gain

* `cdeotte`: the refill step is worth ~0.06 in his notebook's trajectory.
* Applied systematically inside A04 with searched gadgets: **0.01–0.03**.
* Also useful standalone as a "repair" for any packing with an axis‑aligned
  empty corner.

## Cost

Low. MIS on ≤ a few hundred anchors solves in milliseconds.

## Risks

* Upright‑only refill cannot exploit slivers thinner than 1 — that is what the
  tilted core is for, so the split must be tuned.
* The down‑set optimality argument fails if the region is not a down‑set; always
  fall back to the MIS.

## Plan

* `src/refill/greedy.py` — bottom‑left greedy on a cell set (numba).
* `src/refill/mis.py` — conflict graph + `milp` solve, component decomposition.
* `src/refill/split.py` — the `k`‑sweep driver, calls A04's gadget optimiser.

## Decisions needed from you

1. Is `scipy.optimize.milp` (HiGHS) acceptable, or do you want a pure‑Python
   branch‑and‑bound to avoid the dependency? (HiGHS is already available via
   scipy 1.17.)
2. Should refill also allow **45°‑only** squares (a middle ground between
   upright and free tilt)? Cheaper than full A06, often nearly as good.
