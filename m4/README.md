# m4 — the next chapter: a trainable draft head

M1–M3 measured *why* vanilla speculative decoding loses on this hardware: the
draft disagrees with the target on ~2 of every 3 guesses, so most of its cheap
work gets thrown away. M4 tries the obvious fix — **train the draft to predict
the target** — with an EAGLE-style draft head (arXiv 2401.15077): a small
network that reads the target's own hidden features and predicts the next token
the target would pick.

This folder is **Phase 0** of that: the data pipeline and the head's first
training probe. It is work in progress — the private plan and lessons live in
`docs/internal/` (not published). The cache and training outputs go to `data/`
(gitignored — GBs of tensors).

## Scripts

- **`collect_features.py`** — runs the target model over WikiText-2 and caches
  its hidden states (the "features" the head learns from). Resumable; writes
  ~100K-token chunk files to `data/m4/wikitext2/`.
  ```bash
  HF_HOME=/path/to/hf-cache uv run python m4/collect_features.py --max-tokens 500000
  ```
  Omit `--max-tokens` to process the whole corpus (~3M tokens).

- **`train_head.py`** — trains the head (a linear feature map + one decoder
  layer) on the cached features to predict the target's next token. Writes
  weights, a JSON report (rewritten after every epoch — crash-safe), and
  TensorBoard events to `data/m4/`.
  ```bash
  HF_HOME=/path/to/hf-cache uv run python m4/train_head.py --epochs 5
  uv run tensorboard --logdir data/m4/runs --port 6006   # live curves
  ```
  `--resume` continues from the best checkpoint; `--self-test` runs a tiny
  synthetic wiring check (CPU-only).

## The gate

The head is judged on **`greedy_agreement`** — the fraction of positions where
the head's top-1 token equals the target's top-1 token. That is the number that
decides whether a trained draft is good enough to actually speed up decoding
(the same acceptance problem M1–M3 measured, now attacked at the source).

| vanilla baseline | PASS | BORDERLINE | best so far |
|---|---|---|---|
| ~0.35 | ≥ 0.50 | 0.42–0.50 | 0.473 (epoch 49) |
