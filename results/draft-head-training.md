# draft-head — training results (committable log)

The heavy per-epoch detail lives in the gitignored `data/draft-head/train_report.json`
(GBs of tensors sit in `data/`, so it can never be committed). This file is the
small, tracked record: **`train_head.py` appends a new section here after every
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

**What these three runs taught us** (full story in `draft-head/README.md`):

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
