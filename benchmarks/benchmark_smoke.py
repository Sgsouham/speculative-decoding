"""benchmark_smoke.py — quick decode-methodology probe (M3 prep, Aug 7).

Sanity-checks the timing discipline BEFORE building the full M3 harness:
  - speculative vs target-only autoregressive decode through the SAME engines
    (identical timing path — plan §7 baseline requirement)
  - greedy output-equivalence gate (correctness first — suite convention §6.4)
  - warmup + CUDA-sync'd reps, MEDIAN reported (Repo 01 discipline)
  - peak VRAM via torch.cuda.max_memory_allocated()

Scope: 3 (draft, target) pairs × 1 config each (greedy, k=4, 64 tokens).
Full 36-config sweep belongs to the M3 harness.

--compile / --compile-mode: torch.compile BOTH engines before timing, so the
spec-vs-AR comparison is compiled vs compiled (Aug 8 probe, analysis doc §9).
The first call pays compile time (absorbed by the equiv gate); steady-state
reps are timed. The equiv gate doubles as an end-to-end compiled-correctness
check.

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

# Diverse natural prompts for acceptance-rate honesty (the repeated paragraph
# above inflates acceptance on near-deterministic continuation).
DIVERSE_PROMPTS = [
    "The steam engine transformed industry in the eighteenth century, replacing water and animal power with reliable mechanical energy. "
    "James Watt's separate condenser dramatically improved fuel efficiency, and by the early nineteenth century railways connected distant cities, "
    "reshaping trade, migration, and the very sense of time itself. Engineers kept pushing higher pressures and better materials, "
    "and each improvement unlocked new possibilities in factories, shipping, and mining. The consequences were social as much as technological: "
    "new working patterns, new cities, and new inequalities that reformers would spend decades trying to address.",
    "Photosynthesis converts sunlight into chemical energy, powering nearly all life on Earth. Inside the chloroplast, pigments capture photons and "
    "drive the splitting of water, releasing oxygen as a byproduct. The resulting electrons flow through a chain of proteins, producing energy carriers "
    "that the Calvin cycle uses to fix carbon dioxide into sugars. Plants, algae, and some bacteria perform this chemistry at remarkable efficiency, "
    "and scientists study it closely because a deeper understanding could improve crop yields and inspire artificial systems for clean energy.",
    "In a quiet coastal town, a lighthouse keeper named Elias had spent forty years watching the sea. Each evening he climbed the spiral stairs, "
    "trimmed the lamp, and checked the mechanism that turned the great beam. He knew the currents, the shoals, and the rhythms of the fishermen who "
    "depended on his light. One autumn a storm rolled in faster than any he had seen, and as the waves rose, he noticed a small boat drifting far "
    "beyond the channel markers. He made a choice in that moment that would be retold in the town for generations.",
    "Consider the following algorithmic problem. You are given a sorted array of integers and a target value, and you must find the position of the "
    "target if it exists, returning negative one otherwise. A linear scan is simple but runs in linear time. A better approach compares the middle "
    "element, discards half of the array at every step, and finishes in logarithmic time. This binary search requires careful handling of the boundaries, "
    "and off-by-one errors are common even among experienced programmers, so a robust implementation should be verified against brute force on small inputs.",
    "The modern piano is the product of more than three centuries of incremental invention. Bartolomeo Cristofori built the first instrument with hammers "
    "that struck strings rather than plucking them, around the year 1700, and called it a gravicembalo col piano e forte. The design spread slowly across "
    "Europe, and builders gradually enlarged the range, strengthened the frame with iron, and refined the action so that repeated notes became possible. "
    "By the nineteenth century the piano had become the centerpiece of the bourgeois parlor and the concert hall alike, and composers from Beethoven to "
    "Liszt pushed both its expressive range and its mechanical limits. Its influence on how music was composed, published, and performed can hardly be "
    "overstated.",
    "Geometry began as a practical science of measuring land and building structures, but it became something far more abstract in the hands of the "
    "Greeks. Euclid organized the known results into a single deductive system around 300 BCE, starting from five postulates and deriving hundreds of "
    "propositions from them by pure logic. The parallel postulate, the fifth and most controversial, resisted proof for two thousand years; attempts to "
    "prove it led mathematicians to discover hyperbolic and elliptic geometries in the nineteenth century, showing that space itself need not be "
    "Euclidean. This realization opened the door to non-Euclidean physics, and eventually to the curved spacetime of general relativity. The story is a "
    "reminder that axioms are choices, not truths forced upon us by the world.",
]


def make_prompt(tokenizer, text: str, n_tokens: int, device: str) -> torch.Tensor:
    """Tokenize `text`, padded (repeated) to exactly n_tokens."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    while ids.shape[1] < n_tokens:
        ids = torch.cat([ids, ids[:, : n_tokens - ids.shape[1]]], dim=1)
    return ids[:, :n_tokens]


def write_out(path: str, meta: dict, results: list) -> None:
    """Write the results JSON (mkdir-safe)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "pairs": results}, f, indent=2)


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


def bench_pair(cfg, draft_alias: str, target_alias: str, args, prompt_text: str = PROMPT_TEXT, prompt_name: str = "repeat") -> dict:
    draft_id = resolve_model_id(cfg, "draft", draft_alias)[1]
    target_id = resolve_model_id(cfg, "target", target_alias)[1]

    draft_model, _ = load_model(draft_id)
    target_model, tokenizer = load_model(target_id)
    if args.compile:  # compile BOTH engines — identical timing path discipline
        draft_model = torch.compile(draft_model, mode=args.compile_mode)
        target_model = torch.compile(target_model, mode=args.compile_mode)
    draft = ModelHandle(draft_model, tokenizer)
    target = ModelHandle(target_model, tokenizer)
    prompt = make_prompt(tokenizer, prompt_text, args.prompt_len, draft.device)

    def run_spec():
        return speculative_decode(
            draft, target, prompt, args.max_new,
            draft_length=args.k, temperature=0.0, seed=42,
        )

    def run_ar():
        return autoregressive_decode(target, prompt, args.max_new, temperature=0.0)

    # --- correctness gate: greedy spec == greedy AR (token-identical) ---
    # (also validates the compiled path end-to-end when --compile is set)
    torch.cuda.reset_peak_memory_stats()
    try:
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
    except AssertionError:
        raise  # real correctness failure — never swallow
    except Exception as e:
        if not args.compile:
            raise  # eager path: failures crash loudly, never silently recorded
        # compiled-mode failures (e.g. reduce-overhead CUDA-graph replay error)
        # are findings, not crashes — record and skip timing
        print(f"  {draft_alias}->{target_alias}: compiled path FAILED — "
              f"{type(e).__name__}: {str(e)[:120]}")
        del draft_model, target_model, draft, target
        torch.cuda.empty_cache()
        return {
            "pair": f"{draft_alias}->{target_alias}",
            "draft": draft_alias,
            "target": target_alias,
            "prompt": prompt_name,
            "prompt_len": args.prompt_len,
            "k": args.k,
            "max_new": args.max_new,
            "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "equiv_ok": False,
            "status": f"compile_error: {type(e).__name__}: {str(e)[:160]}",
        }

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
        "prompt": prompt_name,
        "prompt_len": args.prompt_len,
        "k": args.k,
        "max_new": n,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
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
    tag = f"compiled:{args.compile_mode}" if args.compile else "eager"
    print(
        f"{result['pair']:<22} [{tag:<14}] spec {result['spec_tok_s']:6.1f} tok/s | "
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
    ap.add_argument("--prompt-mode", choices=["repeat", "diverse"], default="repeat",
                    help="repeat = single repeated paragraph; diverse = all DIVERSE_PROMPTS")
    ap.add_argument("--prompt-idx", default=None,
                    help="comma-separated indices into DIVERSE_PROMPTS, e.g. 0,1 (default: all)")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated draft:target alias pairs, e.g. "
                         "qwen2.5-0.5b:qwen3-4b (default: all SMOKE_PAIRS)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile both models (mode=--compile-mode) — compiled "
                         "end-to-end spec vs AR. Raises Dynamo's recompile limit so the "
                         "per-decode DynamicCache layer-growth does not silently fall "
                         "back to eager; the first decode (the equiv gate) pays the "
                         "compile, later decodes reuse inductor's on-disk cache.")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead"],
                    help="torch.compile mode for --compile. reduce-overhead = CUDA graphs; "
                         "known to error on the raw HF DynamicCache path (probe_compile.py, Aug 8).")
    ap.add_argument("--out", default="results/smoke_benchmark.json")
    args = ap.parse_args()

    if args.pairs:
        pairs = [tuple(s.strip().split(":")) for s in args.pairs.split(",") if s.strip()]
    else:
        pairs = SMOKE_PAIRS
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    if args.compile:
        # The raw HF loop re-creates its DynamicCache per decode (lazy per-layer
        # growth -> ~num_layers distinct graph shapes). Default cache_size_limit
        # (8) is exhausted by the first prefill, silently falling back to eager —
        # which would fake the 'compiled' timing as eager. Raise it so the first
        # decode compiles every frame (absorbed by the equiv gate) and later
        # decodes hit inductor's on-disk cache.
        torch._dynamo.config.cache_size_limit = 64
    cfg = load_config()
    print(f"smoke: pairs={len(pairs)} prompt={args.prompt_len} k={ks} "
          f"max_new={args.max_new} warmup={args.warmup} reps={args.reps} "
          f"compile={args.compile_mode if args.compile else 'eager'}")
    if args.prompt_mode == "diverse":
        idxs = [int(x) for x in args.prompt_idx.split(",")] if args.prompt_idx else range(len(DIVERSE_PROMPTS))
        prompts = [(DIVERSE_PROMPTS[i], f"diverse-{i}") for i in idxs]
    else:
        prompts = [(PROMPT_TEXT, "repeat")]
    meta = {
        "engine": "src/speculative.py (hand-rolled)",
        "dtype": "fp16",
        "temperature": 0.0,
        "k_sweep": ks,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "timing": f"warmup {args.warmup} / reps {args.reps}, median wall w/ CUDA sync",
        "env": {"USE_HUB_KERNELS": os.environ.get("USE_HUB_KERNELS", "unset")},
    }
    results = []
    for k in ks:
        args.k = k
        for d, t in pairs:
            for text, name in prompts:
                results.append(bench_pair(cfg, d, t, args, prompt_text=text, prompt_name=name))
                write_out(args.out, meta, results)  # incremental — timeout-safe
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
