# src — the implementation

Modules:

- **`config.py`** — reads `config/default.yaml` and resolves model aliases
  (e.g. `qwen2.5-0.5b`) to Hugging Face repo IDs. This is where the one-line
  model switch lives.
- **`models.py`** — loads models in fp16/eval mode; `ModelHandle` wraps a model
  with its own KV cache (its running memory) plus reset/rollback semantics; and
  forces greedy decoding so every `generate()` call follows the same policy as
  our hand-rolled loop (the default policy is *not* greedy — see
  [`docs/blockers.md`](../docs/blockers.md) — the non-greedy-default bug).
- **`speculative.py`** — the vanilla core: `speculative_decode()` (the draft →
  verify → accept/reject → resample loop from the README diagram) and
  `autoregressive_decode()` (the plain, non-speculative baseline through the
  *same* machinery, so comparisons are fair). The correctness-critical line is
  the resample — `max(0, target_prob − draft_prob)`, normalized — which is what
  makes the scheme provably output-identical to the target model alone.
- **`eagle_head.py`** — the EAGLE-1-style draft head architecture (FC + decoder
  layers, warm-started from the target's own layers) and its decode path.
- **`eagle_decode.py`** — the EAGLE decode engine: draft chain, verify, accept/
  reject with the trained head, plus checkpoint loading.
- **`cache_utils.py`** — the chunk-index parser shared by the draft-head data
  pipeline and trainer.
- **`train_eagle_head.py`** — trains the draft head on the cached features
  (see [`../results/eagle-training.md`](../results/eagle-training.md)).
- **`collect_eagle_features.py`** — the feature-cache pipeline: runs the target
  over WikiText-2 and caches its hidden states for training.

Everything here is verified by the suite in [`../tests/`](../tests/).
