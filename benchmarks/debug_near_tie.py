"""debug_near_tie.py — diagnose the M3 0.5B->3B greedy divergence at token 149.

The greedy gate crashed (spec != AR, same engines) at generated token 21 on the
diverse-concat prompt at len 128. Two hypotheses (plan §8 / B9):
  H1 fp16 near-tie: the verify pass computes k+1 logits in one parallel forward,
     AR one at a time — different accumulation order -> ~1 ULP differences; if
     the top-2 gap at that position is tiny, argmax flips.
  H2 real state-alignment bug (cache drift): would fail regardless of prompt.

Decision rule: compare hand-rolled spec + AR against the TWO independent HF
oracles (greedy, and greedy+assistant_model). A near-tie flips only the
hand-rolled-vs-hand-rolled comparison (both HF paths agree with one side within
ULP noise); a real bug breaks HF agreement too. Also replay AR position-by-
position to record the top1-top2 logit gap at the divergence point.

Usage (WSL2):
  HF_HOME=/mnt/d/projects/hf-cache HF_HUB_OFFLINE=1 uv run python \
    benchmarks/debug_near_tie.py --target qwen2.5-3b --prompt-len 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_smoke import DIVERSE_PROMPTS  # noqa: E402
from benchmark_speculative import build_concat_prompt  # noqa: E402
from src.config import load_config, resolve_model_id  # noqa: E402
from src.models import ModelHandle, load_model  # noqa: E402
from src.speculative import autoregressive_decode, speculative_decode  # noqa: E402


def replay_ar_gaps(target, prompt, n: int, n_prompt: int) -> list[tuple[int, float]]:
    """One AR pass recording (position, top1-top2 logit gap) per generated token."""
    target.reset()
    target.forward(prompt)
    gaps = []
    for j in range(n):
        row = target.pending_logits[0, 0]  # [V]
        top2 = row.float().topk(2).values
        gaps.append((n_prompt + j, float(top2[0] - top2[1])))
        tok = row.argmax(dim=-1, keepdim=True).unsqueeze(0)
        target.forward(tok)
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qwen2.5-3b")
    ap.add_argument("--draft", default="qwen2.5-0.5b")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    draft_id = resolve_model_id(cfg, "draft", args.draft)[1]
    target_id = resolve_model_id(cfg, "target", args.target)[1]
    print(f"loading {args.draft} + {args.target} ...", flush=True)
    d_model, tok = load_model(draft_id)
    t_model, _ = load_model(target_id)
    draft = ModelHandle(d_model, tok)
    target = ModelHandle(t_model, tok)

    prompt = build_concat_prompt(tok, DIVERSE_PROMPTS, args.prompt_len, target.device)
    n = args.max_new
    n_prompt = prompt.shape[1]

    sd, sstats = speculative_decode(draft, target, prompt, n, draft_length=args.k, temperature=0.0)
    ar, _ = autoregressive_decode(target, prompt, n, temperature=0.0)
    print(f"spec == ar (hand-rolled): {torch.equal(sd, ar)}")
    print(f"  spec accepted: {sstats['accepted']}/{sstats['proposed']} "
          f"({sstats['accepted'] / sstats['proposed']:.0%})")

    pos = None
    if not torch.equal(sd, ar):
        diff = (sd != ar).nonzero(as_tuple=True)[1]
        pos = int(diff[0])
        print(f"  FIRST DIVERGENCE at output token {pos} (generated token {pos - n_prompt})")
        print(f"  spec window: {sd[0, pos - 2:pos + 3].tolist()}")
        print(f"  ar   window: {ar[0, pos - 2:pos + 3].tolist()}")

        # --- independent HF oracles (which side do they agree with?) ---
        ref_g = target.generate(prompt, max_new_tokens=n)
        ref_a = target.generate(prompt, max_new_tokens=n, assistant_model=draft.model)
        print(f"  HF-greedy == spec: {torch.equal(sd, ref_g)} | HF-greedy == ar: {torch.equal(ar, ref_g)}")
        print(f"  HF-assisted == spec: {torch.equal(sd, ref_a)} | HF-assisted == ar: {torch.equal(ar, ref_a)}")
        hf_div = (ref_g != ref_a).nonzero(as_tuple=True)[1]
        if hf_div.numel():
            print(f"  (note: the two HF paths diverge from each other at {int(hf_div[0])} — "
                  f"also consistent with a near-tie in the shared engine)")

        # --- logit gap at the divergence point (near-tie evidence) ---
        gaps = replay_ar_gaps(target, prompt, n, n_prompt)
        gap_at = dict(gaps).get(pos)
        min_gap = min(g for _, g in gaps)
        print(f"  AR top1-top2 logit gap at divergence token: {gap_at:.4f}")
        print(f"  min gap over all {len(gaps)} generated tokens: {min_gap:.4f}")
        if gap_at is not None and gap_at < 0.1:
            print("  VERDICT: near-tie (fp16 ULP reordering flips argmax) — documented plan §8 risk")
        else:
            print("  VERDICT: gap is LARGE — NOT a near-tie; treat as a real alignment bug (B9-style)")
    else:
        print("  no divergence — pass")

    del draft, target, d_model, t_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
