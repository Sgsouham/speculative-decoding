# results — raw benchmark data

- **`m3_sweep.json`** — the 108-config M3 sweep, the authoritative dataset
  behind [`docs/results.md`](../docs/results.md). Regenerate the human-readable
  table with:

  ```bash
  uv run python benchmarks/benchmark_speculative.py --report results/m3_sweep.json
  ```

- Other `*.json` files — earlier measurements (the torch.compile probe, the
  robustness checks, the smoke-methodology runs). They are preserved for
  reproducibility and historical comparison.

The JSON files are **gitignored by design** (ephemeral run output; `.gitkeep`
keeps the folder in the repo). The tracked, human-readable version of the
results lives in `docs/results.md`.
