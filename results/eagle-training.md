# Draft-head (EAGLE) training — the trained-draft chapter

The main story measured *why* vanilla speculative decoding loses on this
hardware: the draft disagrees with the target on ~2 of every 3 guesses, so most
of its cheap work gets thrown away. This chapter tries the obvious fix — **train
the draft to predict the target** — with an EAGLE-style draft head
(arXiv 2401.15077): a small network that reads the target's own hidden features
and predicts the next token the target would pick. The architecture lives in
[`src/eagle_head.py`](../src/eagle_head.py); this document is the gate and the
results.

## The gate

The head is judged on **`greedy_agreement`** — the fraction of positions where
the head's top-1 token equals the target's top-1 token. That is the number that
decides whether a trained draft is good enough to actually speed up decoding
(the same acceptance problem the main story measured, now attacked at the
source).

| vanilla baseline | PASS | BORDERLINE | best so far |
|---|---|---|---|
| ~0.35 | ≥ 0.50 | 0.42–0.50 | **0.490** (1-layer CE, e26) — all levers explored; build the engine |

## What the training taught us

Three experiments, one pattern — the same shape of lesson as the main story,
one level further in:

1. **Data was the first real lever.** 500K → 3.0M tokens lifted agreement
   0.376 → 0.485 while `val_mse` kept falling.
2. **Depth was not a lever.** The 2-layer head (163M params) made features
   measurably better (`val_mse` lower, faster) but agreement barely moved.
   Features saturate long before argmax agreement does: agreement is a
   knife-edge comparison in a 151,936-way space, and the corpus itself is only
   ~51% predictable for the target (`target_top1_acc` = 0.514) — so part of the
   wall is irreducible.
3. **The objective was the hidden lever — and it's now closed.** We trained
   what the features *are* (MSE) but measured whether the target *agrees*
   (argmax); the paper trains both at once (`--loss eagle`). CE compressed
   the climb ~10× (0.480 @ e10 vs 0.426 @ e10 for MSE) yet hit the same
   ceiling — best **0.490 @ e26**, 95% of the target's own 0.514
   predictability. The wall is corpus + target ambiguity, not the head, the
   loss, or training time.

## The scripts

- **`src/collect_eagle_features.py`** — runs the target model over WikiText-2
  and caches its hidden states (the "features" the head learns from).
  Resumable; writes ~100K-token chunk files to the cache dir.
  ```bash
  HF_HOME=/path/to/hf-cache uv run python src/collect_eagle_features.py --max-tokens 500000
  ```
  Omit `--max-tokens` to process the whole corpus (~3M tokens).

- **`src/train_eagle_head.py`** — trains the head (a linear feature map + N
  decoder layers, warm-started from the target's own top layers) on the cached
  features to predict the target's next token. Writes weights, a JSON report
  (rewritten after every epoch — crash-safe), TensorBoard events, and appends
  each completed run's section to THIS document.
  ```bash
  HF_HOME=/path/to/hf-cache uv run python src/train_eagle_head.py --epochs 5
  HF_HOME=/path/to/hf-cache uv run python src/train_eagle_head.py --loss eagle --epochs 10  # paper loss
  uv run tensorboard --logdir data/draft-head/runs --port 6006   # live curves
  ```
  - `--n-layers 2` stacks two decoder layers (the capacity probe — see
    below). `--resume` continues from the best checkpoint of the SAME depth;
    `--self-test` runs a tiny synthetic wiring check (CPU-only).
  - `--loss mse|eagle` picks the objective. `mse` (default) = pure feature
    MSE — what all the Phase-0 runs used. `eagle` = the paper's **actual**
    loss (arXiv 2401.15077 §3.2): Smooth L1 on the features + 0.1 ×
    cross-entropy between the frozen decode path's logits and the true token
    two ahead (`--ce-weight` tunes the 0.1).

---

# Run log (appended by `src/train_eagle_head.py`)

The heavy per-epoch detail lives in the gitignored `data/draft-head/train_report.json`
(GBs of tensors sit in `data/`, so it can never be committed). This file is the
small, tracked record: **`src/train_eagle_head.py` appends a new section here after every
completed run** (fresh or resume), so the training story ships to GitHub without
the cache.

**What `greedy_agreement` means:** the fraction of positions where the head's
top-1 token equals the target's top-1 token — the acceptance-relevant stat for
greedy speculative decoding. Vanilla (untrained) draft baseline: **~0.35**.
Gate: **PASS ≥ 0.50 · BORDERLINE 0.42–0.50 · STOP < 0.42**.

---

## History before this log existed (reconstructed from the gitignored reports)

| Run | head | loss | epochs | best greedy_agreement | verdict |
|---|---|---|---|---|---|
| 500K-token cache (Phase-0 gate) | 1-layer (85M) | mse | 1–20 | **0.376** @ e20 | STOP → data lever |
| 3.0M-token cache (full corpus) | 1-layer (85M) | mse | 1–100 | **0.485** @ e96 | BORDERLINE |
| 3.0M-token cache | 2-layer (163M) | mse | 1–40 | **0.474** @ e40 | BORDERLINE |

**What these three runs taught us** (the full story is in the intro above):

- More data was the first real lever: 500K → 3.0M tokens lifted the ceiling
  0.376 → 0.485 while `val_mse` kept falling.
- More depth was NOT a lever: a second full decoder layer made the features
  measurably better (`val_mse` lower, faster) yet agreement barely moved
  (+0.007 at matched epochs). The wall was never the head's size — features
  saturate long before argmax agreement does, because agreement is a knife-edge
  comparison in a 151,936-way space, and the corpus itself is only ~51%
  predictable (`target_top1_acc` = 0.514).
- Which leaves **the objective** as the untested lever: we trained pure feature
  MSE but measured argmax agreement. The EAGLE paper's actual loss is
  **Smooth L1 + 0.1 × cross-entropy** (`--loss eagle`) — that is what the runs
  below probe.

## eagle_head_20260812_170017 · fresh · n_layers 1 · eagle (Smooth L1 + 0.1×CE)
- model qwen2.5-3b · lr 0.0005 · seed 42 · head = FC + 1 decoder layer (warm from target's top 1) (85,467,648 params)
- data: 2,984,960 cached → 2,686,976 train / 297,984 val · epochs 1–10
- target_top1_acc (corpus predictability ceiling): 0.514
- **best greedy_agreement 0.480 @ epoch 10** · gate: **BORDERLINE**

| epoch | train_mse | val_mse | top1_acc | greedy_agreement | Δ |
|---|---|---|---|---|---|
| 1 | 3.39 | 23.08 | 0.280 | 0.387 | — |
| 2 | 2.96 | 21.10 | 0.306 | 0.426 | +0.040 |
| 3 | 2.79 | 20.17 | 0.319 | 0.446 | +0.019 |
| 4 | 2.69 | 19.79 | 0.325 | 0.453 | +0.007 |
| 5 | 2.61 | 18.92 | 0.332 | 0.462 | +0.009 |
| 6 | 2.55 | 18.39 | 0.335 | 0.468 | +0.007 |
| 7 | 2.51 | 18.00 | 0.340 | 0.474 | +0.006 |
| 8 | 2.47 | 17.62 | 0.343 | 0.479 | +0.005 |
| 9 | 2.43 | 17.77 | 0.343 | 0.478 | -0.002 |
| 10 | 2.40 | 17.34 | 0.344 | 0.480 | +0.002 |

## eagle_head_20260812_170017 · resume · n_layers 1 · eagle (Smooth L1 + 0.1×CE)
- model qwen2.5-3b · lr 0.0005 · seed 42 · head = FC + 1 decoder layer (warm from target's top 1) (85,467,648 params)
- data: 2,984,960 cached → 2,686,976 train / 297,984 val · epochs 11–30
- target_top1_acc (corpus predictability ceiling): 0.514
- **best greedy_agreement 0.490 @ epoch 26** · gate: **BORDERLINE**

| epoch | train_mse | val_mse | top1_acc | greedy_agreement | Δ |
|---|---|---|---|---|---|
| 11 | 2.39 | 17.22 | 0.346 | 0.478 | — |
| 12 | 2.36 | 16.97 | 0.348 | 0.484 | +0.005 |
| 13 | 2.34 | 17.03 | 0.348 | 0.482 | -0.001 |
| 14 | 2.32 | 16.85 | 0.349 | 0.483 | +0.000 |
| 15 | 2.31 | 16.80 | 0.349 | 0.484 | +0.002 |
| 16 | 2.29 | 16.79 | 0.349 | 0.482 | -0.002 |
| 17 | 2.28 | 16.68 | 0.350 | 0.483 | +0.000 |
| 18 | 2.26 | 16.62 | 0.352 | 0.489 | +0.006 |
| 19 | 2.25 | 16.50 | 0.351 | 0.486 | -0.003 |
| 20 | 2.24 | 16.65 | 0.348 | 0.480 | -0.006 |
| 21 | 2.24 | 16.60 | 0.352 | 0.484 | +0.005 |
| 22 | 2.23 | 16.46 | 0.351 | 0.482 | -0.003 |
| 23 | 2.22 | 16.45 | 0.351 | 0.478 | -0.003 |
| 24 | 2.21 | 16.44 | 0.353 | 0.487 | +0.008 |
| 25 | 2.20 | 16.43 | 0.352 | 0.482 | -0.004 |
| 26 | 2.19 | 16.33 | 0.355 | 0.490 | +0.008 |
| 27 | 2.19 | 16.28 | 0.354 | 0.488 | -0.002 |
| 28 | 2.18 | 16.28 | 0.355 | 0.489 | +0.000 |
| 29 | 2.17 | 16.43 | 0.354 | 0.486 | -0.003 |
| 30 | 2.16 | 16.27 | 0.354 | 0.483 | -0.003 |
