"""benchmark_eagle.py — Phase 1 GPU probe: EAGLE decode with the trained draft head.

Measures the trained Phase-0 draft head (best greedy_agreement 0.490 @ e26, CE
loss — data/draft-head/head_fc.pt + head_layers.pt) against target-only AR
decode through the SAME engines (identical timing path — no `generate()`
shortcut on either side). Three gates, in order (plan §6,
docs/internal/phase1-engine-plan.md):

  1. CORRECTNESS: greedy EAGLE == greedy AR, token-identical, on the eval
     prompts. A divergence at an fp16 near-tie (top1-top2 gap within the
     4-ULP bound) is the documented plan §8 risk — recorded as a finding
     and timed anyway. A LARGE divergence is a real alignment bug and crashes
     loudly (the same discriminator as the benchmark-sweep harness).
  2. ACCEPTANCE: chain acceptance + tau (avg tokens accepted per verify pass)
     vs the ~0.35 vanilla baseline and the 0.490 teacher-forced gate.
     Chained acceptance is expected BELOW 0.490 (self-chained features drift).
  3. TIMING: EAGLE vs AR tok/s, plus the component breakdown (verify pass /
     draft chain / AR token, plan §5) so the verdict is decomposable: the
     fixed-overhead risk measured in the benchmark sweep — per-call launch
     cost on the head forwards can eat the win even at decent acceptance.

Eval prompts: the 6 diverse natural texts (no repeated-paragraph flattery),
each truncated to --prompt-len via concat-truncate (never pad-repeat).

Usage (WSL2):
  HF_HOME=/mnt/d/projects/hf-cache uv run python benchmarks/benchmark_eagle.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

# Bootstrap paths so `from src...` works from any invocation dir.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_smoke import DIVERSE_PROMPTS, median_wall_time, write_out  # noqa: E402
from benchmark_speculative import ar_logit_gap, build_concat_prompt  # noqa: E402

from src.config import load_config, resolve_model_id  # noqa: E402
from src.eagle_decode import EagleHeadProvider, eagle_speculative_decode, load_head_checkpoint  # noqa: E402
from src.models import ModelHandle, load_model  # noqa: E402
from src.speculative import autoregressive_decode  # noqa: E402

# Plan §6 gate thresholds
ACCEPT_PASS = 0.45      # chained acceptance at/above -> PASS (tune draft length first)
ACCEPT_BORDER = 0.35    # below -> STOP (chained acceptance worse than vanilla)
VANILLA_BASELINE = 0.35
GATE_AGREEMENT = 0.490  # the teacher-forced Phase-0 gate (best head, e26)


def acceptance_verdict(acc: float) -> str:
    if acc >= ACCEPT_PASS:
        return "PASS"
    if acc >= ACCEPT_BORDER:
        return "BORDERLINE"
    return "STOP"


def component_timings(target: ModelHandle, provider: EagleHeadProvider,
                      prompt: torch.Tensor, k: int, warmup: int, reps: int) -> tuple[float, float]:
    """(verify_ms, draft_chain_ms) — one EAGLE iteration decomposed (plan §5).

    verify_ms: one parallel target forward over k proposal positions (the
    per-iteration target cost, ~ memory-bound / insensitive to k).
    draft_chain_ms: one full k-step head chain (growing buffer + top layer +
    lm_head per step) — the per-iteration draft cost.
    The observed eagle ms/iteration should be ~ verify + chain; the harness
    prints both so any gap (e.g. cache-growth effects) is visible.
    """
    device = prompt.device
    block = torch.zeros(1, k, dtype=torch.long, device=device)

    target.reset()
    target.forward(prompt)

    def verify():
        target.forward(block)

    target.reset()
    prefill = target.forward(prompt, capture_hidden=True)
    feats = prefill.hidden_states[-2]
    f_seed = feats[:, -2:-1, :]
    t_seed = prompt[:, -1:]

    def chain():
        provider.draft(f_seed, t_seed, k, temperature=0.0)

    return 1000 * median_wall_time(verify, warmup, reps), 1000 * median_wall_time(chain, warmup, reps)


def bench_prompt(cfg, args, target: ModelHandle, head, tokenizer,
                 text: str, prompt_name: str, k: int) -> dict:
    """One (prompt, k) config: all three gates on an already-loaded pair+head."""
    n = args.max_new
    prompt = build_concat_prompt(tokenizer, [text], args.prompt_len, target.device)
    prompt_len = prompt.shape[1]  # actual length (a concat can fall short)

    def run_eagle():
        return eagle_speculative_decode(
            target, head, prompt, n, draft_length=k, temperature=0.0, seed=args.seed)

    def run_ar():
        return autoregressive_decode(target, prompt, n, temperature=0.0)

    # --- gate 1: greedy EAGLE == greedy AR (token-identical) ---
    equiv_note = None
    torch.cuda.reset_peak_memory_stats()
    try:
        eag_out, stats = run_eagle()
        torch.cuda.synchronize()
        ar_out, _ = run_ar()
        torch.cuda.synchronize()
        if torch.equal(eag_out, ar_out):
            equiv, gate = True, "greedy_exact"
        else:
            diff = (eag_out != ar_out).nonzero(as_tuple=True)[1]
            n_div = int(diff.numel())
            first = int(diff[0])
            gap_bound = ar_logit_gap(target, prompt, first)
            near_tie = gap_bound is not None and gap_bound[0] <= gap_bound[1]
            if near_tie:
                # fp16 tie flip in the parallel verify path (the documented
                # plan §8 near-tie) — a finding, not a bug: output equivalent
                # within fp16 noise.
                equiv, gate = "near_tie", "greedy_near_tie"
                equiv_note = (f"near-tie fp16 flip at token {first} "
                              f"(top1-top2 gap {gap_bound[0]:.4f} <= 4-ULP bound "
                              f"{gap_bound[1]:.4f}); {n_div} divergent token(s) "
                              f"downstream")
                print(f"    {equiv_note} — recorded, timing valid", flush=True)
            else:
                raise AssertionError(
                    f"EAGLE != AR ({n_div} divergent token(s), first at {first}; "
                    f"AR-path top1-top2 gap "
                    f"{gap_bound[0]:.4f} > 4-ULP bound {gap_bound[1]:.4f} "
                    f"=> real alignment bug, never swallowed)" if gap_bound else
                    f"EAGLE != AR ({n_div} divergent token(s), first at {first} — "
                    f"could not evaluate the tie bound => real alignment bug)")
    except AssertionError:
        raise  # real correctness failure — crash loudly, never record
    except Exception as e:
        raise  # eager path: failures crash loudly (no --compile path here yet)

    # --- gate 3 timing (gate-1 run doubles as the acceptance source) ---
    t_eag = median_wall_time(run_eagle, args.warmup, args.reps)
    t_ar = median_wall_time(run_ar, args.warmup, args.reps)
    t_verify, t_chain = component_timings(target, EagleHeadProvider(head, target.model, target.device),
                                          prompt, k, args.warmup, args.reps)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    # --- gate 2 acceptance (from the gate-1 run's stats) ---
    accepted = stats["accepted"]
    proposed = stats["proposed"]
    acc = accepted / proposed if proposed else 0.0
    tau = statistics.mean(stats["accepted_per_step"]) if stats["accepted_per_step"] else 0.0
    iterations = stats["iterations"]

    eag_ms_iter = 1000 * t_eag / iterations if iterations else 0.0
    result = {
        "prompt": prompt_name,
        "prompt_len": prompt_len,
        "k": k,
        "max_new": n,
        "seed": args.seed,
        "gate": gate,
        "equiv_ok": equiv,
        "equiv_note": equiv_note,
        "status": "ok",
        # timing
        "eag_tok_s": n / t_eag,
        "ar_tok_s": n / t_ar,
        "speedup": (n / t_eag) / (n / t_ar) if t_ar > 0 else 0.0,
        "eag_ms_per_token": 1000 * t_eag / n,
        "ar_ms_per_token": 1000 * t_ar / n,
        "eag_ms_per_iter": eag_ms_iter,
        "verify_ms": t_verify,
        "draft_chain_ms": t_chain,
        "model_ms_per_iter": t_verify + t_chain,  # plan §5 decomposition
        # acceptance
        "acceptance_rate": acc,
        "accepted": accepted,
        "proposed": proposed,
        "tau": tau,
        "iterations": iterations,
        "accepted_per_step": stats["accepted_per_step"],
        "accept_verdict": acceptance_verdict(acc),
        "peak_vram_mb": peak_vram_mb,
    }
    print(
        f"  {prompt_name:<10} k {k} | eag {result['eag_tok_s']:6.1f} | "
        f"ar {result['ar_tok_s']:6.1f} tok/s | x{result['speedup']:4.2f} | "
        f"acc {result['acceptance_rate']:5.0%} | τ {result['tau']:4.2f} | "
        f"iter {t_verify:.0f}+{t_chain:.0f}={result['model_ms_per_iter']:.0f} "
        f"vs {eag_ms_iter:.0f}ms | {result['gate']}"
    )
    return result


def render_summary(results: list[dict], meta: dict) -> str:
    """Pooled per-k summary table (the probe's headline numbers)."""
    ok = [r for r in results if r.get("status") == "ok"]
    lines = ["| k | prompts | EAGLE tok/s | AR tok/s | Speedup | Acc | τ | Gate |",
             "|---|---|---|---|---|---|---|---|"]
    for k in sorted({r["k"] for r in ok}):
        rs = [r for r in ok if r["k"] == k]
        n = len(rs)
        eag = statistics.mean(r["eag_tok_s"] for r in rs)
        ar = statistics.mean(r["ar_tok_s"] for r in rs)
        acc = sum(r["accepted"] for r in rs) / sum(r["proposed"] for r in rs)
        tau = statistics.mean(r["tau"] for r in rs)
        gate = "all exact" if all(r["gate"] == "greedy_exact" for r in rs) else \
            " / ".join(sorted({r["gate"] for r in rs}))
        lines.append(
            f"| {k} | {n} | {eag:.1f} | {ar:.1f} | x{eag / ar:.3f} | {acc:.0%} "
            f"| {tau:.2f} | {gate} |")
    verdict = "**WIN** (>1.0×)" if any(
        r["speedup"] > 1.0 for r in ok) else "**no win yet** (<=1.0× everywhere)"
    lines += [
        "",
        f"*Head: best greedy_agreement {meta.get('head_agreement', '?')} (teacher-forced "
        f"gate ref). Chained acceptance vs vanilla baseline {VANILLA_BASELINE} "
        f"(plan §6: PASS >= {ACCEPT_PASS}, STOP < {ACCEPT_BORDER}). Speedup verdict: {verdict}*",
    ]
    return "\n".join(lines)


def main():
    dh = load_config().get("draft_head", {})  # config/default.yaml → CLI defaults
    ap = argparse.ArgumentParser(description="Phase 1 EAGLE decode probe (plan §6)")
    ap.add_argument("--model", default=dh.get("model", "qwen2.5-3b"),
                    help="target alias in config/default.yaml")
    ap.add_argument("--head-dir", default=dh.get("out", "data/draft-head"),
                    help="dir with head_fc.pt + head_layers.pt")
    ap.add_argument("--n-layers", type=int, default=1, help="decoder layers in the checkpoint")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--ks", default="3,4,5,8", help="comma-separated draft lengths")
    ap.add_argument("--max-new", type=int, default=128, help="generated tokens per config")
    ap.add_argument("--prompt-idx", default=None,
                    help="comma-separated indices into DIVERSE_PROMPTS (default: all)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/eagle_probe.json")
    ap.add_argument("--report", default=None, metavar="JSON",
                    help="render the pooled summary from an existing results file and exit")
    args = ap.parse_args()

    if args.report:
        payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
        # write_out stores results under "pairs" (benchmark_smoke convention)
        print(render_summary(payload.get("pairs", []), payload.get("meta", {})))
        return

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    idxs = [int(x) for x in args.prompt_idx.split(",")] if args.prompt_idx else range(len(DIVERSE_PROMPTS))
    prompts = [(DIVERSE_PROMPTS[i], f"diverse-{i}") for i in idxs]

    cfg = load_config()
    target_id = resolve_model_id(cfg, "target", args.model)[1]
    print(f"loading target {args.model} ({target_id}) ...", flush=True)
    target_model, tokenizer = load_model(target_id)
    target = ModelHandle(target_model, tokenizer)
    head = load_head_checkpoint(target_model, args.head_dir, n_layers=args.n_layers, device=target.device)
    print(f"head loaded from {args.head_dir} ({args.n_layers} layer(s), "
          f"{sum(p.numel() for p in head.parameters()):,} params, {head.fc.weight.dtype})",
          flush=True)

    meta = {
        "harness": "benchmarks/benchmark_eagle.py (Phase 1 probe)",
        "engine": "src/eagle_decode.py (hand-rolled EAGLE-1 chain)",
        "dtype": "fp16",
        "target": args.model,
        "head_dir": args.head_dir,
        "n_layers": args.n_layers,
        "head_agreement": GATE_AGREEMENT,
        "vanilla_baseline": VANILLA_BASELINE,
        "accept_thresholds": {"PASS": ACCEPT_PASS, "STOP": ACCEPT_BORDER},
        "prompts": [n for _, n in prompts],
        "prompt_len": args.prompt_len,
        "ks": ks,
        "max_new": args.max_new,
        "seed": args.seed,
        "timing": f"warmup {args.warmup} / reps {args.reps}, median wall w/ CUDA sync",
        "env": {"USE_HUB_KERNELS": os.environ.get("USE_HUB_KERNELS", "unset")},
    }
    print(f"EAGLE probe: prompts={len(prompts)} k={ks} prompt_len={args.prompt_len} "
          f"max_new={args.max_new} warmup={args.warmup} reps={args.reps}", flush=True)

    results = []
    for k in ks:
        for text, name in prompts:
            r = bench_prompt(cfg, args, target, head, tokenizer, text, name, k)
            results.append(r)
            write_out(args.out, meta, results)  # incremental — timeout-safe
    try:
        del target_model, target
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"\nwrote {args.out} ({len(results)} configs)")
    print(render_summary(results, meta))


if __name__ == "__main__":
    main()
