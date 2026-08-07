"""benchmark_smoke.py — quick decode-methodology probe (M3 prep, Aug 7).

Sanity-checks the timing discipline BEFORE building the full M3 harness:
  - speculative vs target-only autoregressive decode through the SAME engines
    (identical timing path — plan §7 baseline requirement)
  - greedy output-equivalence gate (correctness first — suite convention §6.4)
  - warmup + CUDA-sync'd reps, MEDIAN reported (Repo 01 discipline)
  - peak VRAM via torch.cuda.max_memory_allocated()

Scope: 3 (draft, target) pairs × 1 config each (greedy, k=4, 64 tokens).
Full 36-config sweep belongs to the M3 harness.

Usage (WSL2):
  HF_HOME=/mnt/d/projects/hf-cache uv run python benchmarks/benchmark_smoke.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

# Scripts aren't covered by pyproject's `pythonpath = ["."]` (pytest-only) —
# bootstrap the repo root so `from src...` works from any invocation dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_model_id
from src.models import ModelHandle, load_model
from src.speculative import autoregressive_decode, speculative_decode

# All catalog combos (plan §4, config `models.drafts` × `models.targets`):
#   0.5B→1.5B (default pair), 0.5B→3B, 0.5B→4B, 0.6B→1.5B, 0.6B→3B, 0.6B→4B
# Every draft pairs with every target (shared 151,669-token vocab).
SMOKE_PAIRS = [
    ("qwen2.5-0.5b", "qwen2.5-1.5b"),
    ("qwen2.5-0.5b", "qwen2.5-3b"),
    ("qwen2.5-0.5b", "qwen3-4b"),
    ("qwen3-0.6b", "qwen2.5-1.5b"),
    ("qwen3-0.6b", "qwen2.5-3b"),
    ("qwen3-0.6b", "qwen3-4b"),
]

PROMPT_TEXT = (
    "Artificial intelligence is the simulation of human intelligence processes by "
    "machines, especially computer systems. Specific applications include expert "
    "systems, natural language processing, speech recognition and machine vision. "
    "The field was founded on the claim that a central property of humans is the "
    "intelligence that can be described precisely enough that a machine can simulate it. "
)


def make_prompt(tokenizer, n_tokens: int, device: str) -> torch.Tensor:
    """Natural-text prompt padded (repeated) to exactly n_tokens."""
    ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    while ids.shape[1] < n_tokens:
        ids = torch.cat([ids, ids[:, : n_tokens - ids.shape[1]]], dim=1)
    return ids[:, :n_tokens]


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


def bench_pair(cfg, draft_alias: str, target_alias: str, args) -> dict:
    draft_id = resolve_model_id(cfg, "draft", draft_alias)[1]
    target_id = resolve_model_id(cfg, "target", target_alias)[1]

    draft_model, _ = load_model(draft_id)
    target_model, tokenizer = load_model(target_id)
    draft = ModelHandle(draft_model, tokenizer)
    target = ModelHandle(target_model, tokenizer)
    prompt = make_prompt(tokenizer, args.prompt_len, draft.device)

    def run_spec():
        return speculative_decode(
            draft, target, prompt, args.max_new,
            draft_length=args.k, temperature=0.0, seed=42,
        )

    def run_ar():
        return autoregressive_decode(target, prompt, args.max_new, temperature=0.0)

    # --- correctness gate: greedy spec == greedy AR (token-identical) ---
    torch.cuda.reset_peak_memory_stats()
    sd_out, stats = run_spec()
    torch.cuda.synchronize()
    ar_out, _ = run_ar()
    torch.cuda.synchronize()
    equiv = bool(torch.equal(sd_out, ar_out))
    if not equiv:
        raise AssertionError(
            f"{draft_alias}->{target_alias}: speculative != autoregressive "
            f"(divergence at token {int((sd_out != ar_out).nonzero(as_tuple=True)[1][0]) if (sd_out != ar_out).any() else '?'})"
        )

    # --- timed runs ---
    t_spec = median_wall_time(run_spec, args.warmup, args.reps)
    t_ar = median_wall_time(run_ar, args.warmup, args.reps)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    n = args.max_new
    accepted = stats["accepted"]
    proposed = stats["proposed"]
    result = {
        "pair": f"{draft_alias}->{target_alias}",
        "draft": draft_alias,
        "target": target_alias,
        "prompt_len": args.prompt_len,
        "k": args.k,
        "max_new": n,
        "equiv_ok": equiv,
        "spec_tok_s": n / t_spec,
        "ar_tok_s": n / t_ar,
        # speedup = spec over target-only AR: >1 means speculative wins
        "speedup": (n / t_spec) / (n / t_ar) if t_ar > 0 else 0.0,
        "spec_ms_per_token": 1000 * t_spec / n,
        "ar_ms_per_token": 1000 * t_ar / n,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "accepted_per_step": stats["accepted_per_step"],
        "peak_vram_mb": peak_vram_mb,
    }
    print(
        f"{result['pair']:<22} spec {result['spec_tok_s']:6.1f} tok/s | "
        f"ar {result['ar_tok_s']:6.1f} tok/s | x{result['speedup']:4.2f} | "
        f"acc {result['acceptance_rate']:5.0%} | {peak_vram_mb:6.0f} MB | "
        f"equiv {'OK' if equiv else 'FAIL'}"
    )

    del draft_model, target_model, draft, target
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description="Decode-methodology smoke benchmark")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--k", default="4",
                    help="comma-separated draft lengths to sweep, e.g. 3,4,5,8")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pairs", default=None,
                    help="comma-separated draft:target alias pairs, e.g. "
                         "qwen2.5-0.5b:qwen3-4b (default: all SMOKE_PAIRS)")
    ap.add_argument("--out", default="results/smoke_benchmark.json")
    args = ap.parse_args()

    if args.pairs:
        pairs = [tuple(s.strip().split(":")) for s in args.pairs.split(",") if s.strip()]
    else:
        pairs = SMOKE_PAIRS
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    cfg = load_config()
    print(f"smoke: pairs={len(pairs)} prompt={args.prompt_len} k={ks} "
          f"max_new={args.max_new} warmup={args.warmup} reps={args.reps}")
    results = []
    for k in ks:
        args.k = k
        for d, t in pairs:
            results.append(bench_pair(cfg, d, t, args))

    meta = {
        "engine": "src/speculative.py (hand-rolled)",
        "dtype": "fp16",
        "temperature": 0.0,
        "k_sweep": ks,
        "timing": f"warmup {args.warmup} / reps {args.reps}, median wall w/ CUDA sync",
        "env": {"USE_HUB_KERNELS": os.environ.get("USE_HUB_KERNELS", "unset")},
    }
    payload = {"meta": meta, "pairs": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
