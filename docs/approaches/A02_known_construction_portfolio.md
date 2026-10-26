# A02 — Known‑construction portfolio + per‑n merge

**Priority: must‑have (already implemented, score 22.304027).**

## Why

Before writing a single line of search, ~99.5 % of all public knowledge can be
absorbed for free. The merge is also the *safety net*: every later experiment
is judged against it, and the submission is always ≥ this score.

## Sources, in order of value

| source | puzzles | what it gives |
|---|---:|---|
| `external/couzo/certificates/best` | 720 | exact rational witnesses, n ≤ 1000 |
| `assets/kmmmvs.../submission.csv` | 1000 | certified public portfolio 22.3078 |
| `assets/kmmmvs.../per_puzzle.csv` | 1000 | provenance: which notebook, which method |
| Ellsworth catalogue (Friedman/Ellsworth SVGs) | ~324 | records, 33 digits |
| `external/couzo/certificates/constructions` | 1000 | constructions to n = 10 000 (weaker) |

## Method

```
best[n] = argmin over sources of  L(source[n])     (after validation)
```

Rules that matter:

1. **Validate before merging.** A source may contain an invalid puzzle that
   still has small `L`. `tools/score.py` checks SAT + metric.
2. **Prefer certificates on ties.** Couzo's rational witnesses certify at any
   precision and give provenance; a float packing that ties may be undecidable.
3. **Compare `L`, not the score**, per puzzle (monotone in `L` for fixed `n`).
4. **Never merge on rounded values.** Compare at 17 digits, then re‑run the
   official metric on the merged file.

## Expected gain

* grid → public portfolio: **−6.63**
* public portfolio → + Couzo: **−0.00375** (done; 112 puzzles improved)

## Cost

Minutes. Done.

## Plan (remaining work)

* `src/portfolio.py` — a registry of sources with loader, validator, priority.
  Adding a new solver output is one entry, and the merge is recomputed.
* Track **provenance per puzzle** (which source, which method) so we know where
  the remaining slack is; `assets/.../per_puzzle.csv` is the template.
* Keep `submissions/best.csv` as the rolling champion.

## Decisions needed from you

1. Do we also harvest **Ellsworth's catalogue** for n ≤ 324 (scrape the Packing
   Center SVGs, 33 digits)? It is likely to beat the public portfolio on some
   mid‑n puzzles — cheap and high value. Requires network scraping.
2. Should `best.csv` include Couzo constructions (weaker but complete) for the
   280 n where no certificate exists, or keep grid there until we search?
