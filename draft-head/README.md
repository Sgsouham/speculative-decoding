# draft-head — the next chapter: a trainable draft head

M1–M3 measured *why* vanilla speculative decoding loses on this hardware: the
draft disagrees with the target on ~2 of every 3 guesses, so most of its cheap
work gets thrown away. This chapter tries the obvious fix — **train the draft
to predict the target** — with an EAGLE-style draft head (arXiv 2401.15077): a
small network that reads the target's own hidden features and predicts the next
token the target would pick.

This folder is **Phase 0** of that: the data pipeline and the head's first
training probe. It is work in progress — the private plan and lessons live in
`docs/internal/` (not published). The cache and training outputs go to `data/`
(gitignored — GBs of tensors).

## Scripts

- **`collect_features.py`** — runs the target model over WikiText-2 and caches
  its hidden states (the "features" the head learns from). Resumable; writes
  ~100K-token chunk files to `data/draft-head/wikitext2/`.
  ```bash
  HF_HOME=/path/to/hf-cache uv run python draft-head/collect_features.py --max-tokens 500000
  ```
  Omit `--max-tokens` to process the whole corpus (~3M tokens).

- **`train_head.py`** — trains the head (a linear feature map + N decoder
  layers, warm-started from the target's own top layers) on the cached
  features to predict the target's next token. Writes weights, a JSON report
  (rewritten after every epoch — crash-safe), TensorBoard events, and a
  committable run section to `results/draft-head-training.md`.
  ```bash
  HF_HOME=/path/to/hf-cache uv run python draft-head/train_head.py --epochs 5
  HF_HOME=/path/to/hf-cache uv run python draft-head/train_head.py --loss eagle --epochs 10  # paper loss
  uv run tensorboard --logdir data/draft-head/runs --port 6006   # live curves
  ```
  - `--n-layers 2` stacks two decoder layers (the capacity probe — see
    below). `--resume` continues from the best checkpoint of the SAME depth;
    `--self-test` runs a tiny synthetic wiring check (CPU-only).
  - `--loss mse|eagle` picks the objective. `mse` (default) = pure feature
    MSE — what all the Phase-0 runs used. `eagle` = the paper's **actual**
    loss (arXiv 2401.15077 §3.2): Smooth L1 on the features + 0.1 ×
    cross-entropy between the frozen decode path's logits and the true token
    two ahead (`--ce-weight` tunes the 0.1). This is the objective lever we
    reached only after data and depth both plateaued.

## The gate

The head is judged on **`greedy_agreement`** — the fraction of positions where
the head's top-1 token equals the target's top-1 token. That is the number that
decides whether a trained draft is good enough to actually speed up decoding
(the same acceptance problem M1–M3 measured, now attacked at the source).

| vanilla baseline | PASS | BORDERLINE | best so far |
|---|---|---|---|
| ~0.35 | ≥ 0.50 | 0.42–0.50 | 0.485 (1-layer, e96) — CE objective probe in flight (0.468 @ e6) |

## What the training so far taught us

Three experiments, one pattern — the same shape of lesson as M1–M3, one level
further in:

1. **Data was the first real lever.** 500K → 3.0M tokens lifted agreement
   0.376 → 0.485 while `val_mse` kept falling.
2. **Depth was not a lever.** The 2-layer head (163M params) made features
   measurably better (`val_mse` lower, faster) but agreement barely moved.
   Features saturate long before argmax agreement does: agreement is a
   knife-edge comparison in a 151,936-way space, and the corpus itself is only
   ~51% predictable for the target (`target_top1_acc` = 0.514) — so part of the
   wall is irreducible.
3. **The objective is the remaining lever.** We trained what the features
   *are* (MSE) but measured whether the target *agrees* (argmax). The paper
   trains both at once (`--loss eagle`). The full history lives in
   `results/draft-head-training.md` — committed, so the numbers ship with the repo.
