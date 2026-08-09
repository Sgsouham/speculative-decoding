# config — model catalog + sweep settings

`default.yaml` holds everything that controls which models and which experiments
run:

- **Model catalog** — the draft models (`qwen2.5-0.5b`, `qwen3-0.6b`) and the
  target models (`qwen2.5-1.5b`, `qwen2.5-3b`, `qwen3-4b`). Any draft pairs with
  any target (they share a tokenizer), so all six pairs are benchmarkable.
- **Active model** — the one-line switch: change `model.draft` / `model.target`
  and every test and benchmark follows.
- **Decode defaults** — draft length k, temperature, max new tokens, seed.
- **Benchmark sweep** — the M3 grid (prompt lengths, draft lengths,
  temperatures) that `benchmarks/benchmark_speculative.py` uses by default.

> Note: Qwen3 chat models have **no `-Instruct` suffix** in their repo IDs
> (`Qwen/Qwen3-4B`, not `Qwen/Qwen3-4B-Instruct`) — see
> [`docs/blockers.md`](../docs/blockers.md) B7.
