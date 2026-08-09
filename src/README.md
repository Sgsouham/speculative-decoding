# src — the implementation

Three small modules:

- **`config.py`** — reads `config/default.yaml` and resolves model aliases
  (e.g. `qwen2.5-0.5b`) to Hugging Face repo IDs. This is where the one-line
  model switch lives.
- **`models.py`** — loads models in fp16/eval mode; `ModelHandle` wraps a model
  with its own KV cache (its running memory) plus reset/rollback semantics; and
  forces greedy decoding so every `generate()` call follows the same policy as
  our hand-rolled loop (the default policy is *not* greedy — see
  [`docs/blockers.md`](../docs/blockers.md) B4).
- **`speculative.py`** — the core: `speculative_decode()` (the draft → verify →
  accept/reject → resample loop from the README diagram) and
  `autoregressive_decode()` (the plain, non-speculative baseline through the
  *same* machinery, so comparisons are fair). The correctness-critical line is
  the resample — `max(0, target_prob − draft_prob)`, normalized — which is what
  makes the scheme provably output-identical to the target model alone.

Everything here is verified by the suite in [`../tests/`](../tests/).
