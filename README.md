# Speculative decoding: a measured walkthrough of why it doesn't win at small scale

*I implemented vanilla speculative decoding from scratch, verified it was correct,
benchmarked it honestly across 108 configurations and 6 model pairs — and it lost
every single time (0.22×–0.65×, where 1.0× is break-even). This repository is the
story of that journey: what I built, the traps that nearly fooled me, and exactly
why the trick fails on this class of hardware.*

## The story in one paragraph

Speculative decoding promises 2–3× faster text generation for free: a small "draft"
model guesses the next few words, and a big "target" model just checks the guesses
instead of thinking from scratch. It only pays off if two things are both true —
**the draft is much cheaper per thought**, and **the draft usually agrees with the
target**. I measured both on a consumer GPU:

- the 3×-smaller draft was only **~13% cheaper** per thought (fixed per-thought
  overhead eats the size advantage), and
- on real text the target **rejected ~2 out of every 3** draft guesses.

Two broken bets, no free lunch. The interesting part is *how* I got to measure that
honestly — three benchmark traps nearly produced a fake win first.

## How it works, in one picture

```
                  ONE ITERATION OF SPECULATIVE DECODING

   prompt ─────────────────┐
                          ▼
                 ┌──────────────────┐
                 │      DRAFT       │   small model — makes k quick
                 │  proposes x1..xk │   guesses (k cheap "thoughts")
                 └────────┬─────────┘
                          │  x1  x2  …  xk
                          ▼
                 ┌──────────────────┐
                 │      TARGET      │   big model — checks all k guesses
                 │  verifies all k  │   in ONE parallel pass (one
                 │  in a single pass│   expensive "thought")
                 └────────┬─────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
       all k accepted            rejected at guess j
             │                         │
             ▼                         ▼
  keep x1..xk plus a bonus     keep x1..xj-1 plus the
  word from the target         target's own correction word
             │                         │
             └───────────┬─────────────┘
                         ▼
             append what was kept, repeat
             until the output is long enough
```

The trick only pays off if the draft's guesses are cheap **and** usually accepted
— the two conditions this repo measured (see [`docs/learnings.md`](docs/learnings.md)).

## The journey

1. **The promise.** The papers say it's a free 2–3×. The recipe looks like an
   afternoon project. → *I built the model plumbing and a reference oracle.*
2. **Building it right is harder than it looks.** A "simple" implementation quietly
   breaks in four different ways (non-greedy references, a changed library API,
   tensor-rank bugs, a wrong rollback length). The only safety net is a strict
   correctness gate against an independent implementation. → *21/21 tests, three-way
   verified.*
3. **Measuring honestly changes everything.** Three benchmark traps — a flattering
   repeated-prompt, a padding trap, and a floating-point tie — nearly produced a fake
   "1.01× win." With real text and honest prompts, the verdict flipped. → *108-config
   sweep: no configuration wins.*
4. **The verdict and what would fix it.** A bigger target, a draft trained to predict
   the target, or eliminating the fixed per-thought overhead — with numbers for each.

The full plain-English explanation (with analogies) is in
[**`docs/learnings.md`**](docs/learnings.md).

## Try it yourself

Requires an NVIDIA GPU (I used an RTX 3060, WSL2, Python via `uv`). Model weights
download on first run.

```bash
uv run pytest tests/                                    # the correctness suite
uv run python benchmarks/benchmark_speculative.py \
    --pairs qwen2.5-0.5b:qwen2.5-1.5b                   # a quick benchmark slice
```

The benchmark harness sweeps prompt length × draft length × temperature and writes
`results/m3_sweep.json`; `--report` prints the markdown table.

## Learn more

| Document | What it is |
|---|---|
| [`docs/learnings.md`](docs/learnings.md) | The analysis in plain English — the idea, the three traps, the two broken bets, what would make it win, and a glossary. |
| [`docs/blockers.md`](docs/blockers.md) | Every bug I hit, in *symptom → root cause → fix → lesson* form, plus my debugging playbook. |
| [`docs/results.md`](docs/results.md) | The complete 108-config benchmark table and how to read it. |

## Layout

```
config/        model catalog + benchmark sweep settings
src/           the implementation (models, speculative loop)
tests/         correctness + equivalence tests
benchmarks/    the M3 sweep harness + probes
docs/          the story (learnings, blockers, results)
results/       raw benchmark JSONs (gitignored)
m4/            the next chapter: training an EAGLE-style draft head (in progress)
```

Each folder has a short `README.md`; start with [`src/`](src/README.md) to read
the code, [`tests/`](tests/README.md) to read the verification, and
[`benchmarks/`](benchmarks/README.md) to reproduce the measurements.

