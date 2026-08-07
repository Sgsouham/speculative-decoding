# Repo 02 — speculative-decoding

Vanilla speculative decoding implemented from scratch (~300 lines): a small **draft** model
proposes candidate tokens, the **target** model verifies them in one parallel forward pass,
and an accept/reject + resample step preserves the target distribution *exactly*.

**Status:** 🟡 M0 (scaffold + env) — in progress. Full build plan: [`plan.md`](plan.md).

## Environment (GPU work in WSL2 Ubuntu only)

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"   # must print True
uv run pytest tests/                                                # after M1+
uv run python benchmarks/benchmark_speculative.py                   # after M3
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


Unfinished, Will be updating this more. Current status, Spec decoding doesnt offer more speedup than the vanilla AR based decoding.
