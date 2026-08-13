# config — model catalog + sweep settings

`default.yaml` holds everything that controls which models and which experiments
run. Loaded by `src/config.py` (`load_config()`); every key below can be
overridden per-run with the relevant CLI flag.

## `model` — the active draft/target pair

The one-line switch: change `model.draft` / `model.target` and every test and
benchmark follows (the drop-in requirement). Any draft pairs with any target —
they share one tokenizer (151,669-token Qwen vocab), so no cross-tokenizer
mapping is needed.

```yaml
model:
  draft: qwen2.5-0.5b     # active draft alias
  target: qwen2.5-1.5b    # active target alias
```

## `models.drafts` / `models.targets` — the catalog

Alias → Hugging Face repo ID. Benchmarks sweep the cross product of drafts ×
targets unless a specific pair is passed (`--pairs`). Add a model here and it
becomes available everywhere (tests, benchmarks, the eagle scripts) — provided
it shares the Qwen tokenizer.

```yaml
models:
  drafts:
    qwen2.5-0.5b: Qwen/Qwen2.5-0.5B-Instruct
    qwen3-0.6b:   Qwen/Qwen3-0.6B              # Qwen3 chat = no -Instruct suffix
  targets:
    qwen2.5-1.5b: Qwen/Qwen2.5-1.5B-Instruct
    qwen2.5-3b:   Qwen/Qwen2.5-3B-Instruct
    qwen3-4b:     Qwen/Qwen3-4B
```

> Note: Qwen3 chat models have **no `-Instruct` suffix** in their repo IDs
> (`Qwen/Qwen3-4B`, not `Qwen/Qwen3-4B-Instruct`) — see
> [`docs/blockers.md`](../docs/blockers.md) — the model-ID quirk.

## `draft_head` — the EAGLE chapter's model + paths

Defaults for `src/collect_eagle_features.py` (the feature-cache pipeline) and
`src/train_eagle_head.py` (the head trainer). Each script's CLI flags default to
these values; passing a flag overrides the config. `cache` is written by the
collector and read by the trainer; `out` holds the head weights +
`train_report.json`; `logdir` is the TensorBoard root.

```yaml
draft_head:
  model: qwen2.5-3b                 # target alias whose hidden states the head learns
  cache: data/draft-head/wikitext2  # feature cache dir (collector output / trainer input)
  out: data/draft-head              # head weights + train_report.json
  logdir: data/draft-head/runs      # TensorBoard log root
```

## `runtime` — dtype/device/environment pins

```yaml
runtime:
  dtype: fp16             # both models in fp16 (12 GB VRAM budget)
  device: cuda            # GPU work only in WSL2 Ubuntu
```

`USE_HUB_KERNELS` (transformers 5.x runtime hub-kernel fetch, default YES) is
pinned OFF in `src/__init__.py` + `tests/conftest.py` — network-independent,
reproducible benchmarks. Override with `USE_HUB_KERNELS=YES` if ever needed.

## `decode` — default decode parameters

Used as defaults by the hand-rolled loop and the equivalence tests:

```yaml
decode:
  draft_length: 4       # k candidate tokens proposed per step
  temperature: 0.0      # 0.0 = greedy; > 0 = sampled (exercises resample path)
  max_new_tokens: 128   # default decode length for equivalence tests
  seed: 42              # fixed seed -> reproducible equivalence comparisons
```

## `benchmark` — the sweep grid

The default grid for `benchmarks/benchmark_speculative.py` (36 configs per
pair = 3 prompt lengths × 4 draft lengths × 3 temperatures):

```yaml
benchmark:
  model_pairs:          # sweep these (draft, target) pairs — cross product by default
    drafts: [qwen2.5-0.5b, qwen3-0.6b]
    targets: [qwen2.5-1.5b, qwen2.5-3b, qwen3-4b]
  sweep:
    prompt_lengths: [32, 128, 512]
    draft_lengths: [3, 4, 5, 8]
    temperatures: [0.0, 0.7, 1.0]
    decode_lengths: [64, 128, 256]
  timing:
    warmup: 3           # full-model runs: warmup 3 / reps 5
    reps: 5
    unit_reps: 50       # per-op (accept/reject math) benches get >= 50 reps
  metrics: [decode_tok_s, acceptance_rate, ms_per_token, e2e_tok_s, peak_vram_mb]
  output_dir: results
```

The full sweep is expensive (~216 runs of ~5–20 s each): split it with
`--pairs` and let incremental writes + resume stitch the file together.

## `oracle` — the third reference implementation

```yaml
oracle:
  hf_assistant_model: true   # transformers generate(assistant_model=...) as 3rd reference
```

Enables the three-way agreement check (hand-rolled == plain HF greedy == HF's
own `assistant_model` speculative decoding) in the test suite.
