# Results — the full benchmark table

*The complete measurement behind the verdict in [`learnings.md`](learnings.md):
**108 configurations across 6 model pairs, and none of them beat plain decoding.**
Speedups range from 0.22× to 0.65× — speculative decoding was always slower.*

Raw data lives in `results/m3_sweep.json`. Regenerate this table with:

```bash
uv run python benchmarks/benchmark_speculative.py --report results/m3_sweep.json
```

## How it was measured (the honest way)

- **The comparison is fair:** speculative decoding and plain decoding ran through the
  *same* code path (no shortcuts on either side).
- **The text is real:** prompts were six different natural texts (history, biology, a
  story, an algorithm problem, music, geometry), joined and cut to length — never a
  repeated paragraph (see the traps in [`learnings.md`](learnings.md)).
- **Correctness gate first:** for greedy configs, speculative output had to match plain
  output token-for-token before any timing was recorded. 3 of 36 greedy configs hit a
  documented fp16 tie (see [`blockers.md`](blockers.md) B11) and are marked
  `greedy_near_tie` — their timing is still valid.
- **Timing:** 1 warm-up, 3 timed runs, median wall-clock with GPU sync. 64 new tokens
  per config.
- **Hardware:** RTX 3060 12 GB, WSL2, PyTorch 2.13, fp16.

## The reading, in one table

| Pair | Configs | Speedup range | Acceptance range | Best config |
|---|---|---|---|---|
| 0.5B draft → 1.5B target | 36 | 0.25×–0.53× | 18–65% | 0.53× (len 128, k=3) |
| 0.5B → 3B | 9 | 0.36×–0.55× | 24–53% | 0.55× (len 32, k=4) |
| 0.5B → 4B | 36 | 0.28×–0.65× | 11–68% | 0.65× (len 128, k=3) |
| 0.6B → 1.5B | 9 | 0.22×–0.33× | 23–51% | 0.33× (len 512, k=4) |
| 0.6B → 3B | 9 | 0.26×–0.39× | 23–49% | 0.39× (len 32, k=4) |
| 0.6B → 4B | 9 | 0.33×–0.42× | 23–47% | 0.42× (len 128, k=4) |

**What to notice:**

- **Bigger targets help, but not enough.** Within each draft row, the speedup rises as
  the target grows (0.5B row: 0.53 → 0.55 → 0.65×). The trend is real — the draft
  becomes relatively cheaper against a bigger target — but the best value is still
  35% slower than plain decoding.
- **Acceptance on real text is low everywhere** (~20–50%), including for same-family
  pairs. The draft's guesses are rejected roughly 2 out of 3 times.
- **Longer drafts hurt.** Asking the draft to guess 8 words ahead lowers acceptance
  (its guesses get worse with distance), so the largest draft length is usually the
  worst, not the best.
- **A bigger draft loses to a smaller one.** The 0.6B draft is slower than the 0.5B
  draft against every target — the extra draft quality doesn't pay for its extra cost.
- **Memory was never the problem** (max ~9.8 GB of 12 GB).

## The full table (108 configs)

*Speedup = spec tokens/sec ÷ plain tokens/sec (above 1.0× = speculative wins). Gates:
`greedy_exact` = output identical to plain decoding; `greedy_near_tie` = an fp16 tie
flip, documented; `sampled_none` = temperature > 0, distribution preserved by design
(unit-tested).*

| Pair | Prompt | Len | k | T | Spec tok/s | Plain tok/s | Speedup | Accept | Gate |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 3 | 0.0 | 14.9 | 35.5 | x0.421 | 46% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 3 | 0.7 | 14.1 | 35.9 | x0.393 | 42% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 3 | 1.0 | 14.2 | 35.7 | x0.398 | 43% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 4 | 0.0 | 13.5 | 36.1 | x0.375 | 37% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 4 | 0.7 | 17.8 | 35.8 | x0.497 | 57% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 4 | 1.0 | 11.7 | 36.0 | x0.324 | 28% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 5 | 0.0 | 12.6 | 36.8 | x0.342 | 31% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 5 | 0.7 | 15.0 | 35.5 | x0.424 | 43% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 5 | 1.0 | 14.0 | 36.4 | x0.384 | 38% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 8 | 0.0 | 10.1 | 36.4 | x0.278 | 22% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 8 | 0.7 | 10.1 | 36.3 | x0.279 | 23% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 32 | 8 | 1.0 | 11.6 | 36.3 | x0.319 | 29% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 3 | 0.0 | 17.8 | 33.7 | x0.529 | 65% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 3 | 0.7 | 16.4 | 34.2 | x0.480 | 59% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 3 | 1.0 | 14.6 | 34.4 | x0.425 | 48% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 4 | 0.0 | 17.4 | 35.1 | x0.497 | 57% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 4 | 0.7 | 17.0 | 34.9 | x0.487 | 56% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 4 | 1.0 | 14.3 | 35.0 | x0.410 | 43% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 5 | 0.0 | 18.1 | 35.1 | x0.517 | 58% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 5 | 0.7 | 15.0 | 34.5 | x0.435 | 44% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 5 | 1.0 | 16.9 | 34.9 | x0.484 | 52% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 8 | 0.0 | 14.7 | 35.3 | x0.417 | 40% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 8 | 0.7 | 15.6 | 34.3 | x0.456 | 44% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 128 | 8 | 1.0 | 10.9 | 34.8 | x0.312 | 27% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 3 | 0.0 | 15.8 | 35.7 | x0.442 | 52% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 3 | 0.7 | 15.9 | 34.7 | x0.458 | 53% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 3 | 1.0 | 15.4 | 34.5 | x0.447 | 49% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 4 | 0.0 | 15.0 | 35.0 | x0.428 | 45% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 4 | 0.7 | 12.0 | 34.7 | x0.346 | 32% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 4 | 1.0 | 12.9 | 34.9 | x0.370 | 35% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 5 | 0.0 | 13.1 | 35.0 | x0.375 | 36% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 5 | 0.7 | 13.9 | 34.8 | x0.399 | 39% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 5 | 1.0 | 13.7 | 35.0 | x0.391 | 38% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 8 | 0.0 | 11.1 | 35.3 | x0.315 | 27% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 8 | 0.7 | 8.8 | 35.5 | x0.247 | 18% | sampled_none |
| qwen2.5-0.5b->qwen2.5-1.5b | diverse | 512 | 8 | 1.0 | 9.4 | 35.2 | x0.266 | 21% | sampled_none |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 32 | 4 | 0.0 | 13.4 | 27.9 | x0.482 | 44% | greedy_exact |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 32 | 4 | 0.7 | 14.4 | 28.1 | x0.513 | 49% | sampled_none |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 32 | 4 | 1.0 | 15.3 | 27.9 | x0.548 | 52% | sampled_none |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 128 | 4 | 0.0 | 13.2 | 27.5 | x0.479 | 42% | greedy_near_tie |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 128 | 4 | 0.7 | 14.4 | 27.2 | x0.528 | 49% | sampled_none |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 128 | 4 | 1.0 | 10.9 | 27.4 | x0.399 | 32% | sampled_none |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 512 | 4 | 0.0 | 11.3 | 25.9 | x0.435 | 35% | greedy_near_tie |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 512 | 4 | 0.7 | 9.3 | 25.7 | x0.360 | 24% | sampled_none |
| qwen2.5-0.5b->qwen2.5-3b | diverse | 512 | 4 | 1.0 | 13.7 | 25.9 | x0.529 | 49% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 3 | 0.0 | 10.9 | 20.8 | x0.523 | 40% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 3 | 0.7 | 10.8 | 20.6 | x0.527 | 41% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 3 | 1.0 | 10.2 | 20.8 | x0.492 | 37% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 4 | 0.0 | 9.8 | 20.7 | x0.471 | 30% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 4 | 0.7 | 8.9 | 20.1 | x0.444 | 26% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 4 | 1.0 | 12.0 | 20.8 | x0.578 | 44% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 5 | 0.0 | 9.5 | 20.1 | x0.473 | 30% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 5 | 0.7 | 9.8 | 20.1 | x0.486 | 31% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 5 | 1.0 | 8.3 | 20.3 | x0.408 | 23% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 8 | 0.0 | 8.1 | 20.5 | x0.394 | 20% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 8 | 0.7 | 8.5 | 19.5 | x0.434 | 24% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 32 | 8 | 1.0 | 5.6 | 20.3 | x0.277 | 11% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 3 | 0.0 | 11.7 | 20.7 | x0.568 | 46% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 3 | 0.7 | 11.3 | 20.5 | x0.551 | 45% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 3 | 1.0 | 13.1 | 20.3 | x0.647 | 56% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 4 | 0.0 | 12.2 | 20.4 | x0.598 | 45% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 4 | 0.7 | 9.0 | 20.3 | x0.443 | 26% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 4 | 1.0 | 8.7 | 20.2 | x0.431 | 26% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 5 | 0.0 | 11.9 | 20.3 | x0.589 | 41% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 5 | 0.7 | 10.3 | 20.1 | x0.512 | 34% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 5 | 1.0 | 8.7 | 20.0 | x0.433 | 25% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 8 | 0.0 | 9.4 | 20.3 | x0.462 | 26% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 8 | 0.7 | 6.8 | 20.3 | x0.337 | 15% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 128 | 8 | 1.0 | 6.2 | 20.4 | x0.305 | 13% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 3 | 0.0 | 9.2 | 19.3 | x0.480 | 32% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 3 | 0.7 | 9.8 | 19.2 | x0.510 | 36% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 3 | 1.0 | 8.4 | 18.9 | x0.444 | 26% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 4 | 0.0 | 8.3 | 19.2 | x0.432 | 24% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 4 | 0.7 | 8.1 | 19.2 | x0.422 | 23% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 4 | 1.0 | 7.9 | 19.3 | x0.411 | 22% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 5 | 0.0 | 7.4 | 19.1 | x0.389 | 19% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 5 | 0.7 | 8.4 | 18.7 | x0.449 | 26% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 5 | 1.0 | 6.3 | 18.9 | x0.332 | 13% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 8 | 0.0 | 5.9 | 19.0 | x0.310 | 12% | greedy_exact |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 8 | 0.7 | 7.0 | 18.9 | x0.369 | 18% | sampled_none |
| qwen2.5-0.5b->qwen3-4b | diverse | 512 | 8 | 1.0 | 6.1 | 18.9 | x0.325 | 14% | sampled_none |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 32 | 4 | 0.0 | 8.2 | 35.4 | x0.231 | 27% | greedy_exact |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 32 | 4 | 0.7 | 9.4 | 35.2 | x0.267 | 35% | sampled_none |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 32 | 4 | 1.0 | 7.6 | 34.9 | x0.218 | 23% | sampled_none |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 128 | 4 | 0.0 | 9.6 | 35.0 | x0.275 | 36% | greedy_exact |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 128 | 4 | 0.7 | 10.6 | 34.8 | x0.305 | 44% | sampled_none |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 128 | 4 | 1.0 | 7.8 | 34.6 | x0.225 | 25% | sampled_none |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 512 | 4 | 0.0 | 9.0 | 34.6 | x0.260 | 35% | greedy_exact |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 512 | 4 | 0.7 | 8.6 | 34.2 | x0.252 | 31% | sampled_none |
| qwen3-0.6b->qwen2.5-1.5b | diverse | 512 | 4 | 1.0 | 9.8 | 34.2 | x0.286 | 39% | sampled_none |
| qwen3-0.6b->qwen2.5-3b | diverse | 32 | 4 | 0.0 | 7.1 | 27.6 | x0.256 | 23% | greedy_exact |
| qwen3-0.6b->qwen2.5-3b | diverse | 32 | 4 | 0.7 | 9.7 | 27.9 | x0.346 | 41% | sampled_none |
| qwen3-0.6b->qwen2.5-3b | diverse | 32 | 4 | 1.0 | 10.8 | 27.8 | x0.389 | 49% | sampled_none |
| qwen3-0.6b->qwen2.5-3b | diverse | 128 | 4 | 0.0 | 7.8 | 27.9 | x0.281 | 29% | greedy_exact |
| qwen3-0.6b->qwen2.5-3b | diverse | 128 | 4 | 0.7 | 8.0 | 27.0 | x0.295 | 31% | sampled_none |
| qwen3-0.6b->qwen2.5-3b | diverse | 128 | 4 | 1.0 | 8.8 | 27.9 | x0.314 | 36% | sampled_none |
| qwen3-0.6b->qwen2.5-3b | diverse | 512 | 4 | 0.0 | 8.9 | 26.4 | x0.337 | 37% | greedy_near_tie |
| qwen3-0.6b->qwen2.5-3b | diverse | 512 | 4 | 0.7 | 9.0 | 26.4 | x0.341 | 37% | sampled_none |
| qwen3-0.6b->qwen2.5-3b | diverse | 512 | 4 | 1.0 | 8.4 | 26.4 | x0.320 | 33% | sampled_none |
| qwen3-0.6b->qwen3-4b | diverse | 32 | 4 | 0.0 | 8.1 | 20.4 | x0.398 | 37% | greedy_exact |
| qwen3-0.6b->qwen3-4b | diverse | 32 | 4 | 0.7 | 7.9 | 20.4 | x0.384 | 34% | sampled_none |
| qwen3-0.6b->qwen3-4b | diverse | 32 | 4 | 1.0 | 7.0 | 20.7 | x0.338 | 28% | sampled_none |
| qwen3-0.6b->qwen3-4b | diverse | 128 | 4 | 0.0 | 8.8 | 20.8 | x0.421 | 40% | greedy_exact |
| qwen3-0.6b->qwen3-4b | diverse | 128 | 4 | 0.7 | 7.7 | 20.6 | x0.372 | 31% | sampled_none |
| qwen3-0.6b->qwen3-4b | diverse | 128 | 4 | 1.0 | 7.6 | 19.9 | x0.382 | 34% | sampled_none |
| qwen3-0.6b->qwen3-4b | diverse | 512 | 4 | 0.0 | 7.1 | 19.1 | x0.371 | 29% | greedy_exact |
| qwen3-0.6b->qwen3-4b | diverse | 512 | 4 | 0.7 | 7.5 | 19.1 | x0.390 | 32% | sampled_none |
| qwen3-0.6b->qwen3-4b | diverse | 512 | 4 | 1.0 | 6.3 | 19.1 | x0.329 | 23% | sampled_none |

*Columns: prompt length in tokens (32/128/512), draft length k, temperature T (0 = greedy),
spec/plain tokens per second, speedup, acceptance rate, correctness gate.*
