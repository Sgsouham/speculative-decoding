"""benchmark_speculative.py — M3: the 36-config sweep harness (plan §7).

Measures speculative vs target-only autoregressive decode through the SAME
engines (identical timing path — no `generate()` shortcut on either side) over:

    prompt_len {32, 128, 512} × draft k {3, 4, 5, 8} × temperature {0, 0.7, 1.0}

= 36 configs per (draft, target) pair. Diverse natural prompts are the DEFAULT —
the repeated paragraph inflates acceptance (~80% vs ~35% on real text; robustness
check Aug 8, analysis doc §7). Long prompts are built by CONCATENATING the
diverse texts and truncating (build_concat_prompt) — never by repeating a
paragraph, which would re-inflate acceptance. `--prompt-mode repeat` restores
the old single text for A/B.

Methodology (inherited from benchmark_smoke.py, the validated probe):
  - warmup + CUDA-sync'd reps, MEDIAN reported (Repo 01 discipline)
  - greedy correctness gate FIRST: spec == AR token-identical; failures crash
    loudly (numbers are never reported before they're verified, plan §6.4).
    Exception: a divergence at an fp16 near-tie (top1-top2 gap within the 4-ULP
    bound, B5) is the documented plan §8 risk — recorded as gate=greedy_near_tie
    and timed anyway (a 1-ULP reordering in the verify path flips a tied argmax;
    the output is equivalent within fp16 tie noise). First seen on 0.5B→3B
    (Aug 9, debug_near_tie.py: both HF oracles agreed with AR at the tie).
  - sampled configs (temp > 0): no exact gate — the max(0, p_t - p_d)_+ resample
    is unit-tested (plan §6.2); recorded as equiv_ok: null
  - peak VRAM via torch.cuda.max_memory_allocated()
  - incremental JSON writes after EVERY config (timeout-safe), with resume:
    already-recorded configs are skipped, so a sweep can be split across calls

Budget discipline: each pair is loaded ONCE and all its configs run before
unloading (a 4B load is ~4-5 min over /mnt/d; re-loading per config would
dominate). A full 6-pair × 36-config sweep is ~216 runs of ~5-20 s each —
split it with --pairs and let incremental writes + resume stitch the file.

--compile / --compile-mode: torch.compile BOTH engines once per pair before
timing, so spec-vs-AR is compiled vs compiled (Aug 8 probe, analysis doc §9).
The greedy gate doubles as the end-to-end compiled-correctness check;
compile-mode failures are recorded as findings (status), never swallowed.

--report <json>: render a markdown summary table from an existing results file
(the Results.md deliverable, plan §7) without touching the GPU.

Usage (WSL2):
  HF_HOME=/mnt/d/projects/hf-cache HF_HUB_OFFLINE=1 uv run python \
    benchmarks/benchmark_speculative.py --pairs qwen2.5-0.5b:qwen2.5-1.5b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import torch

# Bootstrap paths so `from src...` works from any invocation dir.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# Shared, already-validated helpers + prompts live in the smoke probe.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_smoke import (  # noqa: E402
    DIVERSE_PROMPTS,
    PROMPT_TEXT,
    SMOKE_PAIRS,
    make_prompt,
    median_wall_time,
    write_out,
)

from src.config import load_config, resolve_model_id  # noqa: E402
from src.models import ModelHandle, load_model  # noqa: E402
from src.speculative import autoregressive_decode, speculative_decode  # noqa: E402


def build_concat_prompt(tokenizer, texts: list[str], n_tokens: int, device: str) -> torch.Tensor:
    """Concatenate texts (natural boundaries) and truncate to exactly n_tokens.

    The honest long-prompt builder: for prompt lengths beyond one diverse text
    we CONCATENATE more diverse texts rather than repeat a paragraph. Repeating
    (make_prompt) makes the continuation near-deterministic and re-inflates
    acceptance — the exact artifact the Aug 8 robustness check exposed. Short
    lengths truncate, which is fine (open-ended continuation).
    """
    ids = torch.cat([tokenizer(t, return_tensors="pt").input_ids for t in texts], dim=1).to(device)
    return ids[:, :n_tokens]


def ar_logit_gap(target: ModelHandle, prompt, pos: int) -> tuple[float, float] | None:
    """(top1-top2 gap, 4-ULP tie bound) at output position `pos` per the AR path.

    Replays the AR decode up to `pos` (the position that PREDICTS output token
    `pos`) and reports how close the top-2 logits were. If the gap is within the
    fp16 4-ULP bound (B5), a 1-ULP reordering in the parallel verify path can
    legitimately flip the argmax — the documented plan §8 near-tie risk. Returns
    None if `pos` is outside the decode range.
    """
    n_prompt = prompt.shape[1]
    if pos < n_prompt:
        return None
    target.reset()
    target.forward(prompt)
    for _ in range(n_prompt, pos):
        tok = target.pending_logits.argmax(dim=-1)
        target.forward(tok)
    row = target.pending_logits[0, 0].float()
    top2 = row.topk(2).values
    gap = float(top2[0] - top2[1])
    bound = 4 * float(torch.finfo(torch.float16).eps) * float(row.abs().max())
    return gap, bound


def _tag(args) -> str:
    return f"compiled:{args.compile_mode}" if args.compile else "eager"


def _config_key(draft_alias, target_alias, prompt_name, prompt_len, k, temperature, args) -> str:
    """Uniquely identify a timed config — incl. timing params, so a re-run with
    different --max-new/--warmup/--reps/--seed is NOT silently skipped as done."""
    return (
        f"{draft_alias}->{target_alias}|{prompt_name}|{prompt_len}|{k}|{temperature}"
        f"|{args.compile_mode if args.compile else 'eager'}"
        f"|{args.max_new}|{args.warmup}|{args.reps}|{args.seed}"
    )


def bench_config(draft: ModelHandle, target: ModelHandle, tokenizer, args,
                 draft_alias: str, target_alias: str, k: int, temperature: float,
                 text: str, prompt_name: str, prompt_len: int) -> dict:
    """One (k, temperature, prompt_len) config on an already-loaded pair."""
    greedy = temperature <= 0.0
    n = args.max_new
    if args.prompt_mode == "diverse":
        # concat-truncate, never pad-repeat (see build_concat_prompt docstring)
        prompt = build_concat_prompt(tokenizer, args.diverse_texts, prompt_len, draft.device)
    else:
        prompt = make_prompt(tokenizer, text, prompt_len, draft.device)
    prompt_len = prompt.shape[1]  # ACTUAL length — a concat can fall short of the target
    key = _config_key(draft_alias, target_alias, prompt_name, prompt_len, k, temperature, args)
    pair = f"{draft_alias}->{target_alias}"

    def run_spec():
        return speculative_decode(
            draft, target, prompt, n,
            draft_length=k, temperature=temperature, seed=args.seed,
        )

    def run_ar():
        return autoregressive_decode(target, prompt, n, temperature=temperature, seed=args.seed)

    # --- correctness gate (never time a wrong path) ---
    equiv_note = None
    torch.cuda.reset_peak_memory_stats()
    try:
        sd_out, stats = run_spec()
        torch.cuda.synchronize()
        ar_out, _ = run_ar()
        torch.cuda.synchronize()
        if greedy:
            if torch.equal(sd_out, ar_out):
                equiv, gate = True, "greedy_exact"
            else:
                # plan §6.4: a divergence at an fp16 near-tie (top1-top2 gap
                # within the 4-ULP bound, B5) is a documented risk, not a bug —
                # the verify path's parallel forward reorders fp16 accumulation
                # and flips a tied argmax. Record it as a finding and keep
                # timing (output is equivalent within fp16 tie noise). A LARGE
                # divergence is a real alignment bug and must crash loudly.
                # A single tie flip at the FIRST divergence cascades: once the
                # paths pick different tokens, conditioning diverges and many
                # downstream tokens differ. The discriminator is the top1-top2
                # gap AT THE FIRST divergence on the canonical (AR) path — the
                # debug (Aug 9) confirmed gap == 0.0 there and both HF oracles
                # agreed with AR. Only n_div is reported; the gap decides.
                diff = (sd_out != ar_out).nonzero(as_tuple=True)[1]
                n_div = int(diff.numel())
                first = int(diff[0])
                gap_bound = ar_logit_gap(target, prompt, first)
                near_tie = gap_bound is not None and gap_bound[0] <= gap_bound[1]
                if near_tie:
                    equiv, gate = "near_tie", "greedy_near_tie"
                    equiv_note = (f"near-tie fp16 flip at token {first} "
                                  f"(top1-top2 gap {gap_bound[0]:.4f} <= 4-ULP bound "
                                  f"{gap_bound[1]:.4f}); {n_div} divergent token(s) "
                                  f"downstream of the flip")
                    print(f"    {equiv_note} — recorded, timing valid", flush=True)
                else:
                    raise AssertionError(
                        f"{pair}: speculative != autoregressive ({n_div} divergent "
                        f"token(s), first at {first}; AR-path top1-top2 gap "
                        f"{gap_bound[0]:.4f} > 4-ULP bound {gap_bound[1]:.4f} "
                        f"=> real alignment bug, never swallowed)" if gap_bound else
                        f"{pair}: speculative != autoregressive ({n_div} divergent "
                        f"token(s), first at {first} — could not evaluate the tie "
                        f"bound => real alignment bug, never swallowed)"
                    )
        else:
            equiv = None  # sampled: distribution-preservation is unit-tested (§6.2)
            gate = "sampled_none"
    except AssertionError:
        raise  # real correctness failure — crash loudly, never record
    except Exception as e:
        if not args.compile:
            raise  # eager path: failures crash loudly
        # compiled-mode failures (e.g. reduce-overhead CUDA-graph replay error)
        # are findings, not crashes — record and skip timing
        print(f"  {key}: compiled path FAILED — {type(e).__name__}: {str(e)[:120]}")
        return {
            "key": key,
            "pair": pair, "draft": draft_alias, "target": target_alias,
            "prompt": prompt_name, "prompt_len": prompt_len, "k": k, "temperature": temperature,
            "max_new": n, "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "gate": "compile_error", "equiv_ok": False,
            "status": f"compile_error: {type(e).__name__}: {str(e)[:160]}",
        }

    # --- timed runs ---
    t_spec = median_wall_time(run_spec, args.warmup, args.reps)
    t_ar = median_wall_time(run_ar, args.warmup, args.reps)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    accepted = stats["accepted"]
    proposed = stats["proposed"]
    return {
        "key": key,
        "pair": pair, "draft": draft_alias, "target": target_alias,
        "prompt": prompt_name, "prompt_len": prompt_len, "k": k, "temperature": temperature,
        "max_new": n, "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "gate": gate, "equiv_ok": equiv, "equiv_note": equiv_note, "status": "ok",
        "spec_tok_s": n / t_spec,
        "ar_tok_s": n / t_ar,
        # speedup = spec over target-only AR: >1 means speculative wins
        "speedup": (n / t_spec) / (n / t_ar) if t_ar > 0 else 0.0,
        "spec_ms_per_token": 1000 * t_spec / n,
        "ar_ms_per_token": 1000 * t_ar / n,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "iterations": stats["iterations"],
        "accepted_per_step": stats["accepted_per_step"],
        "peak_vram_mb": peak_vram_mb,
    }


def bench_pair(cfg, draft_alias: str, target_alias: str, args, prompts,
               out_path: Path, meta: dict, resume: bool = True) -> list[dict]:
    """Load one (draft, target) pair ONCE, run all its pending configs.

    resume=True (default): skip configs already recorded in out_path, appending
    to the file. resume=False: start fresh — results=[] and the first write
    overwrites the file wholesale.
    """
    results = []
    if resume and out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8")).get("pairs", [])
        except Exception:
            results = []
    done = {r["key"] for r in results if r.get("key")}

    keys = [
        _config_key(draft_alias, target_alias, name, plen, k, temp, args)
        for plen, k, temp, (_, name) in product(args.prompt_lens, args.ks, args.temps, prompts)
    ]
    pending = [k for k in keys if k not in done]
    if not pending:
        print(f"{draft_alias}->{target_alias}: all configs already recorded — skipping load")
        return results
    print(f"{draft_alias}->{target_alias}: loading pair ({len(pending)} pending configs) ...",
          flush=True)

    draft_id = resolve_model_id(cfg, "draft", draft_alias)[1]
    target_id = resolve_model_id(cfg, "target", target_alias)[1]
    draft_model, _ = load_model(draft_id)
    target_model, tokenizer = load_model(target_id)
    if args.compile:  # compile BOTH engines once per pair — identical timing path discipline
        draft_model = torch.compile(draft_model, mode=args.compile_mode)
        target_model = torch.compile(target_model, mode=args.compile_mode)
    draft = ModelHandle(draft_model, tokenizer)
    target = ModelHandle(target_model, tokenizer)

    try:
        for plen, k, temp, (text, name) in product(args.prompt_lens, args.ks, args.temps, prompts):
            key = _config_key(draft_alias, target_alias, name, plen, k, temp, args)
            if key in done:
                continue
            r = bench_config(draft, target, tokenizer, args, draft_alias, target_alias,
                             k, temp, text, name, plen)
            results.append(r)
            done.add(key)
            write_out(str(out_path), meta, results)  # incremental — timeout-safe
            print(
                f"  [{_tag(args):<14}] len {plen:3d} k {k} T {temp:>3} "
                f"spec {r.get('spec_tok_s', 0):6.1f} | ar {r.get('ar_tok_s', 0):6.1f} tok/s | "
                f"x{r.get('speedup', 0):4.2f} | acc {r.get('acceptance_rate', 0):5.0%} | "
                f"{r.get('peak_vram_mb', 0):6.0f} MB | {r.get('status', '?')}",
                flush=True,
            )
    finally:
        del draft_model, target_model, draft, target
        torch.cuda.empty_cache()
    return results


def render_markdown(results: list[dict], meta: dict) -> str:
    """Markdown summary table (the Results.md deliverable, plan §7)."""
    rows = sorted(
        (r for r in results if r.get("status") == "ok"),
        key=lambda r: (r["pair"], r["prompt_len"], r["k"], r["temperature"]),
    )
    lines = ["| Pair | Prompt | Len | k | T | Spec tok/s | AR tok/s | Speedup | Acc | VRAM MB | Gate |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc = f"{r['acceptance_rate']:.0%}" if r.get("acceptance_rate") is not None else "—"
        lines.append(
            f"| {r['pair']} | {r['prompt']} | {r['prompt_len']} | {r['k']} | {r['temperature']} "
            f"| {r['spec_tok_s']:.1f} | {r['ar_tok_s']:.1f} | x{r['speedup']:.3f} | {acc} "
            f"| {r['peak_vram_mb']:.0f} | {r['gate']} |"
        )
    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        lines.append("")
        lines.append("**Recorded failures (findings, not crashes):**")
        for r in failed:
            lines.append(f"- `{r['key']}` — {r['status']}")
    lines.append("")
    lines.append(f"*{len(rows)} configs × pairs from {meta.get('harness', 'M3 sweep')}*")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="M3 full 36-config sweep harness (plan §7)")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated draft:target alias pairs, e.g. "
                         "qwen2.5-0.5b:qwen3-4b (default: all 6 SMOKE_PAIRS)")
    ap.add_argument("--prompt-lens", default="32,128,512",
                    help="comma-separated prompt lengths (plan §7)")
    ap.add_argument("--ks", default="3,4,5,8",
                    help="comma-separated draft lengths k")
    ap.add_argument("--temps", default="0,0.7,1.0",
                    help="comma-separated temperatures (0 = greedy)")
    ap.add_argument("--max-new", type=int, default=128,
                    help="generated tokens per config (decode length)")
    ap.add_argument("--prompt-mode", choices=["diverse", "repeat"], default="diverse",
                    help="diverse = natural prompts (DEFAULT — honest acceptance); "
                         "repeat = the old flattering repeated paragraph")
    ap.add_argument("--prompt-idx", default=None,
                    help="comma-separated indices into DIVERSE_PROMPTS to CONCATENATE "
                         "per prompt_len, e.g. 0,1 (default: all 4)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
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
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore configs already recorded in --out (default: resume)")
    ap.add_argument("--out", default="results/m3_sweep.json")
    ap.add_argument("--report", default=None, metavar="JSON",
                    help="render a markdown summary table from an existing results file and exit")
    args = ap.parse_args()

    if args.report:
        payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print(render_markdown(payload.get("pairs", []), payload.get("meta", {})))
        return

    if args.pairs:
        pairs = [tuple(s.strip().split(":")) for s in args.pairs.split(",") if s.strip()]
    else:
        pairs = SMOKE_PAIRS
    args.prompt_lens = [int(x) for x in args.prompt_lens.split(",") if x.strip()]
    args.ks = [int(x) for x in args.ks.split(",") if x.strip()]
    args.temps = [float(x) for x in args.temps.split(",") if x.strip()]
    if args.prompt_mode == "diverse":
        idxs = [int(x) for x in args.prompt_idx.split(",")] if args.prompt_idx else list(range(len(DIVERSE_PROMPTS)))
        # all selected texts are concatenated per prompt_len (build_concat_prompt)
        args.diverse_texts = [DIVERSE_PROMPTS[i] for i in idxs]
        prompts = [(DIVERSE_PROMPTS[0], "diverse-concat")]  # text unused in diverse mode
    else:
        prompts = [(PROMPT_TEXT, "repeat")]

    if args.compile:
        # The raw HF loop re-creates its DynamicCache per decode (lazy per-layer
        # growth -> ~num_layers distinct graph shapes). Default cache_size_limit
        # (8) is exhausted by the first prefill, silently falling back to eager —
        # which would fake the 'compiled' timing as eager. Raise it so the first
        # decode compiles every frame (absorbed by the equiv gate) and later
        # decodes hit inductor's on-disk cache.
        torch._dynamo.config.cache_size_limit = 64

    cfg = load_config()
    out_path = Path(args.out)
    meta = {
        "harness": "benchmarks/benchmark_speculative.py (M3, plan §7)",
        "engine": "src/speculative.py (hand-rolled)",
        "dtype": "fp16",
        "prompt_mode": args.prompt_mode,
        "prompt_idxs": idxs if args.prompt_mode == "diverse" else None,
        "sweep": {"prompt_lengths": args.prompt_lens, "draft_lengths": args.ks,
                  "temperatures": args.temps},
        "max_new": args.max_new,
        "seed": args.seed,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "timing": f"warmup {args.warmup} / reps {args.reps}, median wall w/ CUDA sync",
        "env": {"USE_HUB_KERNELS": os.environ.get("USE_HUB_KERNELS", "unset")},
    }
    n_configs = len(pairs) * len(args.prompt_lens) * len(args.ks) * len(args.temps) * len(prompts)
    print(f"M3 sweep: pairs={len(pairs)} configs/pair={len(args.prompt_lens) * len(args.ks) * len(args.temps) * len(prompts)} "
          f"total={n_configs} max_new={args.max_new} compile={args.compile_mode if args.compile else 'eager'} "
          f"prompts={[n for _, n in prompts]}")

    if not args.no_resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get("pairs", [])
            print(f"resume: {len(existing)} configs already in {out_path}")
        except Exception:
            pass

    for d, t in pairs:
        bench_pair(cfg, d, t, args, prompts, out_path, meta, resume=not args.no_resume)

    results = json.loads(out_path.read_text(encoding="utf-8")).get("pairs", [])
    print(f"\nwrote {out_path} ({len(results)} configs)")
    print(render_markdown(results, meta))


if __name__ == "__main__":
    main()
