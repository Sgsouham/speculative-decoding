# benchmarks — the measurement harness

Everything needed to reproduce the numbers in
[`../results/speculative-decoding.md`](../results/speculative-decoding.md). GPU
required (WSL2). Model weights come from your HF cache (`HF_HOME`).

- **`benchmark_speculative.py`** — the main benchmark sweep harness. Sweeps
  prompt length × draft length × temperature for any model pair. It verifies
  correctness *first* (greedy configs must match plain decoding token-for-token
  before any timing), then times both paths with CUDA-synced medians and writes
  incremental, resumable JSON. Useful flags:
  - `--pairs qwen2.5-0.5b:qwen2.5-1.5b` — which draft:target pair(s)
  - `--prompt-lens 32,128,512` / `--ks 3,4,5,8` / `--temps 0,0.7,1.0` — the grid
  - `--max-new 64` / `--warmup 1` / `--reps 3` — decode length + timing discipline
  - `--report results/speculative_sweep.json` — print the markdown table from a run
  - `--compile` — `torch.compile` both models before timing (the fixed-overhead
    lever from [`docs/learnings.md`](../docs/learnings.md))
- **`benchmark_smoke.py`** — the earlier methodology probe (single-config checks).
  It owns the shared timing discipline, the diverse prompt set, and the
  incremental-write pattern that the sweep harness reuses.
- **`benchmark_eagle.py`** — the Phase-1 GPU probe: EAGLE decode with the
  trained draft head. Three gates (token-identical correctness, acceptance,
  timing) × prompt × k; results in `results/eagle_probe.json`; `--report`
  prints the pooled summary. See [`../results/eagle-training.md`](../results/eagle-training.md).
- **`probe_compile.py`** — measured how much `torch.compile` cuts the fixed
  per-thought overhead (the "commute" cost in the analysis).
- **`debug_near_tie.py`** — the diagnostic used to prove that a suspicious
  benchmark "failure" was actually a floating-point tie, not a bug (the
  floating-point-tie entry in [`docs/blockers.md`](../docs/blockers.md)).

The prompts are six varied natural texts, **concatenated and truncated** to the
target length — never a repeated paragraph (padding re-inflates acceptance and
would flatter the draft; see the traps in the analysis).
