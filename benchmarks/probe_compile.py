"""probe_compile.py — prototype: can torch.compile kill the ~20 ms fixed
per-forward overhead at decode shapes? (TODO 6, Aug 8)

Hypothesis under test (docs/learnings.md — the fixed-overhead finding):
batch-1 decode forwards are dominated by fixed kernel-launch + Python-dispatch
overhead (~19-20 ms) that does NOT shrink with model size. If torch.compile
(inductor fusion; `reduce-overhead` adds CUDA graphs) collapses it, the draft
becomes weight-bound (~3x cheaper than the target) and speculative decoding
wins even at k=4.

Methodology (mirrors the doc's per-forward measurement, appendix):
  - warm KV cache at fixed L (prompt 128 tokens)
  - eager vs compiled (mode=default, mode=reduce-overhead) 1-token and 4-token
    forwards; crop the cache back to L between reps (untimed); CUDA-sync'd
    median over reps (defaults: warmup 5 / reps 30)
  - correctness gate FIRST: compiled vs eager must be argmax-identical
    (tokens strict, logits loose — the suite's fp16-honest rule). A wrong
    path is never timed.

Note on CUDA graphs: HF's DynamicCache grows via `torch.cat` per step, and
`reduce-overhead` replay requires identical input tensor addresses. We record
the outcome (works / wrong / errors) rather than assume.

Usage (WSL2):
  HF_HOME=/mnt/d/projects/hf-cache HF_HUB_OFFLINE=1 uv run python \
    benchmarks/probe_compile.py --role draft --alias qwen2.5-0.5b \
    --modes eager,default,reduce-overhead --out results/probe_compile_05b.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Scripts aren't covered by pyproject's `pythonpath = [\".\"]` (pytest-only) —
# bootstrap the repo root so `from src...` works from any invocation dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_model_id
from src.models import load_model

# Diverse natural text (same as benchmark_smoke DIVERSE_PROMPTS[0]) — a real
# prompt, not the flattering repeated paragraph.
PROMPT_TEXT = (
    "The steam engine transformed industry in the eighteenth century, replacing water and animal power with reliable mechanical energy. "
    "James Watt's separate condenser dramatically improved fuel efficiency, and by the early nineteenth century railways connected distant cities, "
    "reshaping trade, migration, and the very sense of time itself. Engineers kept pushing higher pressures and better materials, "
    "and each improvement unlocked new possibilities in factories, shipping, and mining. The consequences were social as much as technological: "
    "new working patterns, new cities, and new inequalities that reformers would spend decades trying to address."
)


def median_wall_time(fn, warmup: int, reps: int) -> float:
    """CUDA-synchronized wall-clock median over reps (Repo 01 discipline)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def prefill_cache(model, tokenizer, text: str, device: str, max_tokens: int):
    """Warm the KV cache at a fixed length; returns (cache, prompt_ids)."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)[:, :max_tokens]
    with torch.no_grad():
        out = model(ids, use_cache=True)
    return out.past_key_values, ids


def step(model, cache, tok) -> None:
    """One forward with the given cache (mutates the cache in place)."""
    with torch.no_grad():
        model(tok, past_key_values=cache, use_cache=True)


def crop_back(cache, L: int) -> None:
    """Restore the cache to length L (5.x mutates in place; 4.x returns new)."""
    if cache is None:
        return
    result = cache.crop(L)
    if result is not None:
        cache = result


def run_gate(eng, ref, cache_factory, tok, L: int, steps: int = 3):
    """compiled vs eager over `steps` single-token steps (crop-back each).

    Returns (argmax_identical, max_logit_diff).
    """
    ce, cr = cache_factory(), cache_factory()
    ok = True
    max_diff = 0.0
    for _ in range(steps):
        with torch.no_grad():
            oe = eng(tok, past_key_values=ce, use_cache=True)
            orr = ref(tok, past_key_values=cr, use_cache=True)
        ok = ok and bool(torch.equal(oe.logits[:, -1:].argmax(dim=-1), orr.logits[:, -1:].argmax(dim=-1)))
        max_diff = max(max_diff, float((oe.logits - orr.logits).abs().max()))
        crop_back(ce, L)
        crop_back(cr, L)
    return ok, max_diff


def timed_forward(eng, cache, tok, L: int, warmup: int, reps: int) -> float:
    """Median ms for one forward at fixed cache length L (crop-back between reps)."""
    for _ in range(warmup):
        step(eng, cache, tok)
        crop_back(cache, L)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        crop_back(cache, L)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(eng, cache, tok)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def bench_mode(raw_model, cache_factory, tok, tok4, L: int, mode: str, warmup: int, reps: int) -> dict:
    """Time one mode (eager | default | reduce-overhead) with a gate first."""
    if mode == "eager":
        eng = raw_model
        status = "ok"
    else:
        print(f"  compiling mode={mode} ...", flush=True)
        eng = torch.compile(raw_model, mode=mode, dynamic=False)
        status = "ok"

    # --- correctness gate (never time a wrong path) ---
    gate_ok, max_diff = run_gate(eng, raw_model, cache_factory, tok, L)
    print(f"  gate (mode={mode}): argmax_identical={gate_ok} max_logit_diff={max_diff:.4f}", flush=True)
    if not gate_ok:
        return {"mode": mode, "status": "gate_failed", "gate_ok": False,
                "max_logit_diff": max_diff, "ms_1tok": None, "ms_4tok": None}

    ms1 = timed_forward(eng, cache_factory(), tok, L, warmup, reps)
    print(f"  timed (mode={mode}): 1-token {ms1:.2f} ms", flush=True)
    ms4 = timed_forward(eng, cache_factory(), tok4, L, warmup, reps)
    print(f"  timed (mode={mode}): 4-token {ms4:.2f} ms", flush=True)
    return {"mode": mode, "status": status, "gate_ok": True,
            "max_logit_diff": max_diff, "ms_1tok": ms1, "ms_4tok": ms4}


def main():
    ap = argparse.ArgumentParser(description="torch.compile per-forward overhead probe")
    ap.add_argument("--role", choices=["draft", "target"], default="draft")
    ap.add_argument("--alias", default="qwen2.5-0.5b", help="model alias from config/default.yaml")
    ap.add_argument("--modes", default="eager,default,reduce-overhead",
                    help="comma-separated: eager, default, reduce-overhead")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="results/probe_compile.json")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cfg = load_config()
    repo_id = resolve_model_id(cfg, args.role, args.alias)[1]
    print(f"probe: role={args.role} alias={args.alias} repo={repo_id} modes={modes} "
          f"prompt={args.prompt_len} warmup={args.warmup} reps={args.reps}", flush=True)

    model, tokenizer = load_model(repo_id)
    cache, ids = prefill_cache(model, tokenizer, PROMPT_TEXT, model.device, args.prompt_len)
    # Crop target must be the ACTUAL cached length (the tokenizer may produce
    # fewer tokens than args.prompt_len) — cropping to a larger length is a
    # no-op and the cache would grow every rep (shape drift -> Dynamo
    # recompiles -> invalid timing).
    L = ids.shape[1]
    print(f"  prompt tokens: {L}", flush=True)
    tok = ids[:, -1:]                 # [1, 1] — continue from the last prompt token
    tok4 = ids[:, -4:]                # [1, 4]

    def cache_factory():
        return prefill_cache(model, tokenizer, PROMPT_TEXT, model.device, L)[0]

    results = []
    out_path = Path(args.out)
    if out_path.exists():  # resume — skip already-recorded modes
        try:
            results = json.loads(out_path.read_text(encoding="utf-8")).get("results", [])
        except Exception:
            results = []
    done = {r["mode"] for r in results}

    for mode in modes:
        if mode in done:
            print(f"  skip mode={mode} (already recorded)", flush=True)
            continue
        try:
            r = bench_mode(model, cache_factory, tok, tok4, L, mode, args.warmup, args.reps)
        except Exception as e:  # compile/capture failures are findings, not crashes
            r = {"mode": mode, "status": f"error: {type(e).__name__}: {str(e)[:200]}",
                 "gate_ok": False, "max_logit_diff": None, "ms_1tok": None, "ms_4tok": None}
            print(f"  mode={mode} ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
        r.update({"role": args.role, "alias": args.alias, "repo_id": repo_id})
        results.append(r)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "meta": {
                "probe": "torch.compile fixed-overhead hypothesis (TODO 6)",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "engine": "src/models.py load_model",
                "prompt_len": L,
                "timing": f"warmup {args.warmup} / reps {args.reps}, median wall w/ CUDA sync",
            },
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"  incremental write -> {out_path}", flush=True)

    print(f"wrote {out_path}")
    for r in results:
        print(f"  {r.get('alias')} {r['mode']:>16}: "
              f"{r.get('ms_1tok', '—') if r.get('ms_1tok') is not None else '—':>7} ms/1tok "
              f"{r.get('ms_4tok', '—') if r.get('ms_4tok') is not None else '—':>7} ms/4tok "
              f"gate={r.get('gate_ok')} [{r.get('status')}]")


if __name__ == "__main__":
    main()
