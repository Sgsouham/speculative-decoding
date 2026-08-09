# Blockers — every bug I hit, and what it taught me

*A debugging log in the format **symptom → root cause → fix → lesson**. These are the
mistakes that made building a "simple" 300-line algorithm take real work — each one
was a silent failure that would have corrupted my results if I hadn't been testing
against an independent reference.*

**Status legend:** ✅ resolved · 🔶 worked around

---

## M0 — Environment

### B7 · The model IDs I remembered don't exist ✅
- **Symptom:** `Repository Not Found` for `Qwen/Qwen3-0.6B-Instruct`.
- **Root cause:** Qwen3 dropped the `-Instruct` suffix — chat models are just
  `Qwen/Qwen3-0.6B` / `Qwen/Qwen3-4B` (only Qwen2.5 keeps `-Instruct`).
- **Fix:** verified against the Hugging Face API instead of trusting memory.
- **Lesson:** don't trust model-ID memory; the registry API is one request away.
  Gated or renamed repos return 401 — the search API shows what actually exists.

---

## M1 — Model plumbing

### B5 · Floating-point drift between two valid code paths ✅
- **Symptom:** step-by-step cached decoding and one-shot decoding disagreed on logits
  by up to 0.047 — on 60% of values.
- **Root cause:** fp16 (half-precision) rounding. Running the same math in a different
  order produces ~1-ULP (unit-in-the-last-place) differences. Not a cache bug.
- **Fix:** assert **token identity** strictly (which word was picked), and compare
  numeric values only within a computed tolerance.
- **Lesson:** compare *tokens* strictly, *numbers* loosely. A real cache bug flips the
  chosen word; fp16 noise doesn't.

---

## M2 — The core loop

### B1 · A library function silently returned `None` ✅
- **Symptom:** the KV cache (running memory) became `None` after a rollback, and every
  later step recomputed from empty — wrong output everywhere.
- **Root cause:** transformers 5.x refactored the cache API: `crop()` now **mutates in
  place and returns `None`** (4.x returned a new object). My code stored the `None`.
- **Fix:** handle both behaviors (assign the result only when it's not `None`).
- **Lesson:** read the library's actual source instead of assuming the old API.
  transformers 5.x is a moving target — the "API drift is real" warning was right.

### B2 & B3 · One-element tensors corrupted the model ✅
- **Symptom:** `RuntimeError: tensor a (14) must match tensor b (64)` inside the
  rotary-embedding math — on BOTH the correction path and the bonus path.
- **Root cause:** indexing `tensor[0, i]` collapses the shape to 1-D. A 1-D tensor
  handed to the model confuses its layout assumptions, and the error's numbers (14
  heads × 64 head-dim) named the exact model.
- **Fix:** use `tensor[:, i:i+1]` (keeps 2-D) everywhere.
- **Lesson:** when feeding tensors to a model, always know the exact rank. `[i]`,
  `[:, i]`, and `[:, i:i+1]` are different shapes. Print `.shape` when in doubt.

### B4 · The "greedy" reference wasn't greedy ✅
- **Symptom:** my hand-rolled decoder matched Hugging Face's for 10 tokens, then
  picked a *different word* at position 10 even though the raw scores had a clear
  winner. Separately, my own decoder produced a degenerate repetition loop while HF
  produced sensible text.
- **Root cause:** the model repo ships a `generation_config` that bakes in **sampling**
  (temperature 0.7, top-k, top-p, repetition penalty) — so `generate()` wasn't greedy
  at all. My decoder and HF were following different policies.
- **Fix:** force greedy everywhere (explicit kwargs + normalizing the loaded config),
  so the reference and the implementation agree on policy.
- **Lesson:** never assume `generate()` without arguments is greedy — check the model's
  config. Also: a degenerate repetition loop in your own decoder is a tell that you're
  conditioning on wrong/truncated context. Comparing against an independent
  implementation surfaced a discrepancy my self-consistent tests could never see.

### B8 · A throwaway script collided with the wrapper ✅
- **Symptom:** `got multiple values for keyword argument 'do_sample'`.
- **Root cause:** my debug script passed `do_sample=False` while the wrapper
  hardcodes it.
- **Fix:** don't pass it in the script.
- **Lesson:** debug scripts and library wrappers can collide; keep the wrapper's API
  surface explicit.

### B9 · A rollback used the wrong length and silently dropped history ✅
- **Symptom:** after the rope fixes, all greedy tests failed with
  `speculative != autoregressive` — the speculative decoder ran cleanly but produced
  *different tokens* than the reference on the same engine.
- **Root cause:** the rejection path rolled the cache back to `prompt_length + accepted`
  instead of `current_sequence_length + accepted`. That is correct only on iteration 1;
  on later iterations it **silently dropped every earlier iteration's tokens**, so the
  model conditioned on truncated history.
- **Why it hid:** every debug script exercised only *one* rollback; the multi-iteration
  case was never tested until the real suite ran to completion.
- **Fix:** track the accepted-sequence length before each iteration's drafts and roll
  back to that.
- **Lesson:** off-by-N bugs hide until multi-iteration paths run. When a rollback or
  truncation is involved, the length must be the **current** sequence length, never a
  constant captured at init. And: two paths on the same engine disagreeing is *always* a
  state-alignment bug, never floating-point noise (see B11 for the one exception).

---

## M3 — Benchmarking honestly

### B11 · The greedy gate crashed on a floating-point tie ✅
- **Symptom:** the benchmark crashed: speculative ≠ autoregressive at one output
  position, with 24 tokens diverging downstream.
- **Root cause:** an **exact tie** in fp16 — two candidate words scored *exactly* the
  same. The speculative path computes that position in one big parallel pass while the
  reference computes it one step at a time; a ~1-ULP rounding difference flipped which
  side of the tie each path recorded. One flip then cascades: once the paths disagree,
  conditioning diverges and many later tokens differ. Verified against two independent
  reference implementations (both agreed with the reference path), so it was rounding,
  not a bug.
- **Fix:** the benchmark gate now checks the margin at the first divergence against a
  4-ULP tolerance. Within tolerance → record the row as `near-tie` (with the evidence)
  and keep the timing. A real, large-margin divergence still crashes loudly.
- **Lesson:** B9's "same engine always means a state bug" has a documented exception:
  **exact ties can legitimately flip across execution orders.** Before chasing a state
  bug, check the top-2 margin at the first divergence. Also: the number of divergent
  tokens is NOT the severity signal — the first margin is.
- **Bonus lesson (prompt honesty):** never pad benchmark prompts by repeating a
  paragraph — it re-inflates acceptance toward ~100% (a "diverse" prompt repeated is
  still a flattering prompt). Fill long prompts by concatenating different texts.

---

## Debugging playbook (what actually worked)

1. **Isolate with sectioned scripts.** Split the stack into sections (multi-token
   append, rollback+re-append, draft sequence, target multi-append, full loop). The
   failing layer was always one section apart from the passing ones.
2. **Cache-vs-one-shot comparison.** The most powerful check: a cached walk must
   reproduce a one-shot forward's logits. It proved the cache machinery correct,
   narrowing bugs to input shapes and decode policy.
3. **Compare against an independent implementation.** Hand-rolled vs Hugging Face
   `generate()` surfaced B4 — a discrepancy our self-consistent tests could never see
   (both paths agreed perfectly while BOTH were wrong relative to HF's actual policy).
4. **Read the library source.** `grep` the actual implementation; it settled API
   questions in seconds.
5. **Read the error's numbers.** "14 vs 64" was two model-specific constants — a
   fingerprint that named the failing model.
6. **Check env vars and configs.** `USE_HUB_KERNELS` and the model's `generation_config`
   both silently change numerics. For reproducible benchmarks, pin them.
7. **Reproduce manually, then fix the source.** The debug script's own indexing bug
   initially *looked* like a library bug — always diff the debug script against the
   real code path.
