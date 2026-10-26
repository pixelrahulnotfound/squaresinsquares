# A07 — Global search: basin hopping / SA / extremal optimisation

**Priority: high — this is where the LB leaders' margin comes from.**

## Why

The packing landscape has thousands of near‑equal local optima (Schadt's
statistics: for `s(51)` only **4 of 3004** basins refine to the record). The
record is a needle, and the only reliable way to find needles is to sample many
basins. Public notebooks either do a little of this (`mirzamilanfarabi`: SA,
`alexchilton`: evand hop) or a narrow GPU version (`lightningv08`, 261 s).
Nobody has run a long, well‑tuned, CPU‑parallel search across the whole 325–1000
band.

## Method

One fixed container side `L` (from A03/A04), energy `E` from A06, and a move
set. Four algorithms, in increasing order of my expectation:

### 7.1 Basin hopping (Wales–Doye)
```
x ← quench(x)                       # A06
loop:
    x' ← kick(x, k squares, σ)      # k ~ U{1..kmax}, σ ~ N(0, r²)
    x' ← quench(x')
    accept if E(x') < E(x) or exp(−ΔE/T) > U
    if E == 0:  squeeze L (A06) and reset T
```
`kmax ≈ 0.05n`, `r ≈ 0.05L`, `T` annealed on a fixed schedule. This is what
`evand/square-packing` and `alexchilton` use; it is strong and simple.

### 7.2 Simulated annealing with integer overlap count
Energy = `#overlapping pairs` (integer, cheap), plus a penalty on total
penetration for tie‑breaking. Moves: single‑square translation, rotation,
rigid‑body pair/subset translation. Temperature geometrically cooled. Cheaper
per step than basin hopping; worse asymptotically.

### 7.3 Threshold accepting (Dueck–Scheuer)
Accept if `ΔE < T` (no probability, no exp). ~3× cheaper per step; on rugged
discrete landscapes it matches Metropolis. **Untried in this competition** —
cheap experiment.

### 7.4 Extremal optimisation (Boettcher–Percus)
Rank squares by their penetration contribution `f_i = Σ_j pen_ij`; always mutate
the worst‑ranked (power‑law `P(k) ∝ k^{-τ}`, `τ ≈ 1.4`). Natural fit: the
"squares in the way" get kicked, not random ones. **Untried here.**

### 7.5 Perturbed billiard (Gensane–Ryckelynck)
Feasible‑only: never accept overlap; displace one square (or all squares at
once — the key layer) inside the container, accept if still feasible. Maintains
`solid` configurations and walks along the solid manifold. Excellent at
*squeezing* after a good layout is found, weaker at escaping.

## Combining them (recommended)

```
A03/A04 construction  →  A06 quench
        →  7.1 basin hopping (many seeds, parallel)
        →  7.5 billiard squeeze (feasible-only polish)
        →  A08 contact LP
        →  A01 gate
```

## Expected gain

* Per basin hit at large `n`: `1e-3…1e-2` in `L`.
* With 20 cores × hours: plausibly **0.02–0.08** total, concentrated in
  325–1000.
* At small `n` (< 100) the public portfolio is already at record; expect ~0.

## Cost

**Very high** — this is the compute sink. Rough CPU budget: one basin for
`n = 1000` ≈ 3–10 s (with spatial hashing); 1000 basins/puzzle × 700 puzzles =
~2–8 core‑days. Parallel over puzzles, not within a puzzle.

## Risks

* Diminishing returns: the same basins get revisited. Diversity matters more
  than raw iterations → vary seeds, `k`, `σ`, and the move mix.
* CPU‑only here (no CUDA). The published record engines are GPU; we must
  compensate with better seeds (A03/A04/A10) rather than more samples.
* A quench that returns `E > 0` wastes the whole basin. Use the repair (A01)
  and reject early.

## Plan

* `src/search/hop.py` — basin hopping driver, per‑puzzle checkpointing.
* `src/search/sa.py`, `threshold.py`, `eo.py`, `billiard.py` — move sets behind
  one `Proposer` interface so they can be mixed.
* `src/search/schedule.py` — temperature/budget allocation across `n`.
* Results written as `submissions/search_<tag>.csv` + a JSON of per‑puzzle
  basin statistics (how many basins hit the record), which tells us where to
  spend more.

## Decisions needed from you

1. Compute budget: how many core‑hours may I spend? (This single answer decides
   whether we target 22.25 or 22.20.)
2. Prioritise **325–1000** (volume) or **100–324** (bigger per‑puzzle gains but
   records are harder to beat)?
3. May I use all 20 cores for long runs, or should I keep the machine
   responsive (e.g. 16 workers)?
