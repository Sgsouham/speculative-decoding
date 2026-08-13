# results — the measured outcomes

Two experiments, two documents:

## Part 1 — Speculative decoding

[**`speculative-decoding.md`**](speculative-decoding.md) — the full 108-config
benchmark table and how to read it: every (draft, target) pair × prompt length ×
draft length × temperature measured honestly, and why no configuration won on
this hardware. The story behind the numbers lives in
[`../docs/learnings.md`](../docs/learnings.md).

- Raw JSON: **`speculative_sweep.json`** — the authoritative dataset. Regenerate
  the markdown table with:
  ```bash
  uv run python benchmarks/benchmark_speculative.py --report results/speculative_sweep.json
  ```
- Other `*.json` files — earlier measurements (the compile probe, the robustness
  checks, the smoke-methodology runs). Preserved for reproducibility.

## Part 2 — Draft-head (EAGLE) training

[**`eagle-training.md`**](eagle-training.md) — the trained-draft chapter: the
gate the head is judged on, what each training experiment taught us, and the
full per-epoch run log. The architecture lives in
[`../src/eagle_head.py`](../src/eagle_head.py); the scripts that produce these
numbers are [`../src/train_eagle_head.py`](../src/train_eagle_head.py) and
[`../src/collect_eagle_features.py`](../src/collect_eagle_features.py).

---

The JSON files are **gitignored by design** (ephemeral run output; `.gitkeep`
keeps the folder in the repo). The tracked, human-readable versions of the
results are the two markdown documents above.
