# Repo 02 — speculative-decoding

Vanilla speculative decoding implemented from scratch (~300 lines): a small **draft** model
proposes candidate tokens, the **target** model verifies them in one parallel forward pass,
and an accept/reject + resample step preserves the target distribution *exactly*.

**Status:** 🟢 M0–M3 done (21/21 tests; **108-config M3 sweep — honest null result: vanilla
speculative decoding does not win on this hardware**, x0.22–0.65). Results:
[`results/Results.md`](results/Results.md) · measured analysis:
[`docs/why-speculative-is-slower-at-small-scale.md`](docs/why-speculative-is-slower-at-small-scale.md) ·
Full build plan: [`plan.md`](plan.md).

## Environment (GPU work in WSL2 Ubuntu only)

```bash
uv run pytest tests/                                                 # M0–M2: 21/21 green
uv run python benchmarks/benchmark_speculative.py --pairs qwen2.5-0.5b:qwen3-4b   # M3 sweep
```

## Layout

```
config/        default.yaml (model ids, draft k, temperature, benchmark sweep)
src/           implementation (M1–M4)
tests/         correctness + equivalence tests (M2)
benchmarks/    benchmark + validation harness (M3)
results/       JSON runs + consolidated Results.md (M3)
```

> Canonical env is managed by `uv` (`pyproject.toml` + `uv.lock` committed).
