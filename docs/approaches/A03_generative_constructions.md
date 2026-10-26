# A03 — Generative construction families

**Priority: must‑have. This is the only approach that produces a complete
1000‑puzzle submission without any search, and it sets the starting point for
everything else.**

## Why

For `n > 324` there is no record table. Every large‑`n` packing in the public
portfolio was *constructed* (lattice + tilted strip + refill) and then only
lightly polished. `avazaku/beyond-the-grid` demonstrates the ladder:

| stage | score |
|---|---:|
| integer grid | 28.9377 |
| public catalogue (n ≤ 324) | 25.77 |
| + outer rows | 23.42 |
| + Göbel & subdivisions | 23.13 |
| + rectangular cores | 22.84 |
| + asymmetric cores | 22.74 |
| + sparse LP polish | **22.7416** |

So constructions alone reach 22.74 — only 0.43 behind the full public portfolio,
and they cover all 1000 puzzles (no grid holes). Improving the *grammar* is the
highest‑leverage non‑search work available.

## Method: a grammar of patterns, each with continuous parameters

For each `n`, write `n = K² + r`, `L = K + f`, `f ∈ (0,1)`.

### F1 — Grid + upright strip (baseline)
`K×K` grid plus `r` squares in a row of height `1`, requires `r ≤ K`,
`f = 0`. Gives `L = K` for `n ≤ K²+K`. Score `K/√n − 1`.

### F2 — Tilted strip ("dented row")
`r` squares at angle `θ`, spaced by `d`, sitting on the `K×K` grid so that their
bottom corners **penetrate the gaps between the grid squares below**. With the
grid dilated to gap `g`, a `θ`‑rotated square drops by
`δ(θ) = (cosθ + sinθ − 1)/2 − g·tanθ` relative to an upright one, so the extra
height needed is

```
h(θ, g) = cosθ + sinθ − δ(θ)
```

Choose `(θ, g, d)` to satisfy `r·d ≤ K` and minimise `h`. This is a 3‑variable
NLP per `(K, r)` — trivial, and it is the family that produces most large‑`n`
records.

### F3 — Göbel strip (`n = K² + K + 1`)
One extra square in a corner, the rest of the strip tilted at 45°;
`L = K + 1/√2` when `r = K+1`? No — the exact Göbel values are
`s(5) = 2 + 1/√2`, `s(10) = 3 + 1/√2`, and the pattern generalises to
`n = K² + K + 1` with `L ≈ K + 1/√2` for the cases where the strip can be
"folded". Enumerate the fold pattern (which squares go 45°, which stay upright,
where the corner square sits) and solve the resulting small LP/NLP.

### F4 — Rectangular core
Replace the `K×K` grid by `a×b` with `a·b = K² − q` and use the freed `q` cells
plus the strip; the core can then be sheared by `(f, f)` (the fault‑line view).
Enumerate `a, b` near `K`.

### F5 — Asymmetric tilted core
Split the box by a **monotone staircase** into two lattices, one shifted by
`(f, f)`. Each side is an integer grid; the staircase trades width for height.
This is the same object as A04's fault line, but solved *analytically* for a
fixed staircase shape rather than by DP.

### F6 — L‑cut / corner removal
Remove an `a×b` corner from the grid, refill it with `a·b + Δ` tilted squares
(an "L‑cut" gadget), keep the rest integer. `Δ ≥ 1` is what buys `f`.

### F7 — Diagonal bands (yuki16)
Count `P(K,t) = #{(i,j) ∈ [0,K−1]² : i+j ≤ t}` in closed form; place `P` squares
at 45° along the anti‑diagonal plus upright squares elsewhere. Cheap, and it
covers some `n` that F2 misses.

### F8 — Subdivision / composition
If `n = a·b` and we have a `(a, L_a)` and a `(b, L_b)` packing, scale: an
`a×b` array of `L_b`-side blocks inside `a·L_b` gives `L = a·L_b`. Or split the
box into a `(K1, f1)` region and a `(K2, f2)` region along a guillotine cut
(slicing tree DP over `(width, height, count)`).

## Algorithm

```
for n in 1..1000:
    for K in {floor(sqrt(n)), floor(sqrt(n))-1}:
        r = n - K*K
        for pattern in {F2, F3, F4, F5, F6, F7}:
            for discrete choices (fold map, a, b, staircase shape):
                solve small NLP/LP over continuous params  -> f
                if f < best_f: record layout
    best[n] = min over all
```

Everything is independent per `n` → embarrassingly parallel over 20 cores.

## Expected gain

* Fills all 453 grid holes; alone reaches ≈ 22.74 (from `avazaku`'s ladder).
* Better grammar (F4–F6 with proper parameter optimisation) plausibly reaches
  **22.4–22.6** without any search.
* Every construction is a *seed* for A06/A07/A08, so its value is multiplied.

## Cost

Low: hours of CPU, no GPU needed. The NLP/LP subproblems are tiny.

## Risks

* Constructions are rigid — they will not beat a locally optimised packing at
  small `n`. Use them only where `n` is large or where no better source exists.
* Parameter enumeration can explode; cap the discrete search per `n`.

## Plan

* `src/constructions/` — one module per family, each exposing
  `build(n) -> (L, centres, angles)` and `params` for the enumerator.
* `src/constructions/enumerate.py` — the loop above, writing
  `submissions/constructions.csv` + a JSON of which family won per `n`.
* Reuse `tools/score.py` to validate every output.

## Decisions needed from you

1. Should constructions be optimised for **`L`** or for **robustness**
   (minimal clearance) as seeds? I recommend `L` with a `1e-9` clearance.
2. Do you want me to scrape **Ellsworth's SVG catalogue** to calibrate the
   grammar against the true records for n ≤ 324?
3. Priority between F4/F5 (staircase cores) and F6 (L‑cut gadgets)? F6 composes
   better with A04; F5 is stronger standalone.
