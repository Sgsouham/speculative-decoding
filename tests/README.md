# tests — the correctness suite

**21 tests, GPU required (run in WSL2).** The suite's job is to prove the
hand-rolled speculative loop is **output-identical** to plain decoding — the
claim that makes every number in this repo meaningful. Numbers are never
reported before they're verified.

What it checks:

- **Three-way agreement:** hand-rolled speculative == plain HF greedy == HF's
  own speculative decoding (`assistant_model`) — token-identical.
- **All draft lengths** (k = 1, 2, 4, 8) match plain decoding.
- **Sampled decoding** (temperature > 0) is reproducible with a fixed seed, and
  the `max(0, p_target − p_draft)` resample line is unit-tested against a tiny
  artificial model pair.
- **Cache rollback semantics:** cropping the KV cache mid-decode and continuing
  matches a fresh decode.

Run it:

```bash
uv run pytest tests/ -q
```

> Note on strictness: these tests assert exact token equality. Exact logit ties
> can occasionally flip a token between code paths (fp16 rounding — see
> [`docs/blockers.md`](../docs/blockers.md) B11); the benchmark harness handles
> that case explicitly, while this suite stays strict because the default model
> pair has never hit one.
