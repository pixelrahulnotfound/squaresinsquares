# A10 — Subset transfer + closure algebra

**Priority: high. Cheapest structural trick: it reuses an improvement found at
one `n` for its neighbours, for free.**

## Why

Any subset of a valid packing is valid (delete squares, keep the box). So a good
packing at `n+1` is automatically a valid — usually mediocre — packing at `n`,
and a good packing at `n` is a *seed* for `n+1`. Combined with a squeeze, this
propagates improvements along the `n` axis at almost no cost.

`kozykappa` (22.4753) and `malyshevdanil` (22.3078) both use this, and
`skarin`'s Santa‑2025 write‑up formalises the closure.

## The closure rules

```
(C1) monotonicity   L(n) ≤ L(m)  for all m ≥ n
                    (take any n squares from the m-packing)
(C2) frame          L(n + j(2⌊L⌋ + j)) ≤ L + j
                    (wrap the packing in a j-wide frame: each side gains
                     ⌊L⌋ + j cells, i.e. 2j⌊L⌋ + j² squares)
(C3) scaling        L(a·n) ≤ a·L(n)   (a×a array of copies, weak)
(C4) guillotine     L(n1 + n2) ≤ max(L1, L2) if the two packings can be
                    placed side by side / stacked in the box
```

`(C1)` is the important one: apply it as a **running minimum from above**
(`L[n] = min(L[n], L[n+1], …)`), which is what `malyshevdanil.choose` does with
its 6‑layer forward/backward DP.

## Transfer operators (the real content)

### T1 — delete‑and‑squeeze (`n+1 → n`)
Delete the square whose removal least increases the "hole" (largest clearance,
or a corner/edge square), then A06 squeeze. The freed space lets the box shrink
by `f`.

### T2 — insert‑and‑squeeze (`n → n+1`)
Find the largest empty disc (clearance field) and insert a square there at the
locally best angle; quench.

### T3 — shrink‑by‑k (`n+k → n`)
Delete the `k` squares closest to the boundary (they are "cheapest" to remove),
then squeeze. Works surprisingly well when `k/n` is small.

### T4 — uniform compaction (`kozykappa`)
After a transfer, contract all centres toward the centroid by `λ` and quench;
`λ` is found by bisection on feasibility. This is a cheap global squeeze that
often beats local descent.

### T5 — best‑of‑neighbourhood DP
```
for n in descending order:
    for m in {n+1, n+2, n+3}:
        cand = transfer(best[m], m -> n)   # T1/T3
        best[n] = min(best[n], cand)
```

## Expected gain

* `kozykappa` reports ~0.09 from transfer + compaction on top of the public
  catalogue.
* On top of the current portfolio: **0.005–0.02**, mostly in the 325–1000 band
  where many puzzles are still near‑grid.

## Cost

Low: a transfer is one delete + one quench (A06), i.e. seconds. `1000 × 3`
transfers is a few hours on 20 cores.

## Risks

* Deleting from an *optimal* packing destroys it — only apply where there is
  slack (`L > grid` or `L` far above the construction bound).
* `(C1)` must be applied with the *same* `L` semantics (bounding box, not
  container), otherwise the running minimum is wrong.
* Transfers can produce invalid layouts if the repair is skipped.

## Plan

* `src/transfer/ops.py` — T1–T4, each `(layout, n_from, n_to) -> layout`.
* `src/transfer/closure.py` — C1–C4 + the running‑minimum DP.
* `src/transfer/clearance.py` — largest‑empty‑disc / clearance field (numba).

## Decisions needed from you

1. How many neighbours per `n` (`m − n ∈ {1,2,3}` or up to 10)? More = better
   but linear cost.
2. Should the closure DP run **before** search (to fill holes cheaply) or
   **after** (to propagate wins)? I recommend both: once at the start, once
   after each search campaign.
