# A11 — Rigid rotations + minimal enclosing square

**Priority: medium. A free, always‑on post‑process; small but guaranteed.**

## Why

The metric measures the bounding box of the corner cloud. The container has no
orientation, so for *any* layout there is a global rotation `φ` that may reduce
the bounding box. Most notebooks ignore this and leave `φ = 0`.

Additionally, entire *subsets* (a block of the packing) can be rotated rigidly
if the freed space allows a better fit — the move that AlphaEvolve found
decisive for hexagons.

## Method

### 11.1 Global rotation (exact, cheap)

For corner set `P = {p_k}`, the bounding box after rotation `φ` is

```
w(φ) = max_k (p_k·e_x(φ)) − min_k (p_k·e_x(φ))
h(φ) = max_k (p_k·e_y(φ)) − min_k (p_k·e_y(φ))
L(φ) = max(w(φ), h(φ))
```

`w` and `h` are piecewise sinusoidal with breakpoints where the argmax/argmin
changes (at most `O(n²)` breakpoints). So:

1. Enumerate candidate `φ` from the breakpoints (angles between pairs of corner
   points).
2. Between breakpoints, `L` is `max` of two sinusoids → minimise analytically or
   by golden‑section.

Then **re‑optimise `φ` jointly with the squeeze** — a rotation changes which
walls bind, so alternate: rotate → quench (A06) → rotate.

### 11.2 Rigid subset rotation

Pick a connected subset `S` (a "block" of the layout, e.g. a `3×4` group of
squares), rotate it by `φ`, translate to the best free position, quench. Accept
if `L` decreases. This is a large, discrete move that single‑square kicks cannot
replicate.

### 11.3 Whole‑layout 90°/mirror symmetries

Free relabelling and symmetry: canonicalise each layout (sort by angle, fix
mirror) before hashing, so the search does not waste time on duplicates.

## Expected gain

* Global rotation alone: `1e-5…1e-4` per puzzle where it applies (mostly tilted
  layouts that are not yet at the optimal orientation). Total **~1e-3**.
* Rigid subset rotations: `1e-4…1e-3` per successful move; total **~5e-3**.
* The canonicalisation saves 5–20 % of search time (duplicate basins).

## Cost

Very low. Rotation is `O(n log n)`; subset rotation is one quench.

## Risks

* Rotating can break contacts that the LP (A08) had just resolved — always
  re‑quench and re‑validate.
* The breakpoint enumeration is `O(n²)`; for `n = 1000` that is 1e6 candidates —
  use a coarse scan (`Δφ = 1e-4`) then refine locally.

## Plan

* `src/rotate/global.py` — breakpoint enumeration + golden section + alternation
  with the squeeze.
* `src/rotate/block.py` — subset selection (connected components, grid blocks),
  rotation, best‑placement, acceptance.
* `src/rotate/canonical.py` — hashing for deduplication.

## Decisions needed from you

1. Should global rotation be applied to **every** puzzle (safe, slow) or only
   where the layout is non‑axis‑aligned (fast)?
2. Block size for subset rotation: `2×2`, `3×3`, or adaptive by `n`?
