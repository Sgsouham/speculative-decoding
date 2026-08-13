"""eagle_decode.py — EAGLE-1-style speculative decode with the Phase-0 draft head.

The draft head (src/eagle_head.py) predicts the target's next second-to-top
feature from the fused pair (f_i, e(t_{i+1})); the target's frozen top layer +
LM head turn the predicted feature into the draft distribution (paper
arXiv 2401.15077 §3.1). This module is the decode side: it mirrors
src/speculative.py's loop (SAME accept/reject/resample math — the
distribution-preservation claim is unchanged), with the draft source swapped
from a small LM to the head's chain, and per-phase re-seeding from the
target's ACTUAL features so drift never accumulates across phases.

Alignment (verified against the paper, Aug 13 — see
docs/internal/phase1-engine-plan.md §2): with the target cache at length M,
the last known pair is (f_{M-2}, t_{M-1}) — feature ONE BEHIND the last
token. The chain proposes tokens for positions M..M+k-1, verified against the
target's distributions at those positions (pending logits p_M verify proposal
#1) — exactly vanilla's alignment, no off-by-one.

Seed-feature bookkeeping (plan §2.3, the seed-alignment trap): every target forward
runs with capture_hidden=True; after a phase of n_acc accepted proposals,
the next seed feature is the verify pass's row n_acc-1 (or the previous
f_last when n_acc == 0), and the new f_last is the correction/bonus forward's
last row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eagle_head import EagleDraftHead, run_top_layer  # noqa: E402
from src.models import ModelHandle  # noqa: E402
from src.speculative import _normalize_probs, _residual_distribution  # noqa: E402


# --------------------------------------------------------------------------
# Draft provider
# --------------------------------------------------------------------------
class EagleHeadProvider:
    """The draft chain over the trained head (stateless per phase).

    Wraps the head + the target's frozen machinery (embedding, top layer +
    norm via run_top_layer, LM head). Each phase is seeded with the target's
    ACTUAL feature + token; the chain then proposes k tokens, feeding each
    step its own predicted feature and sampled token (the paper's "sampling
    outcomes" input). The head sees the phase's GROWING pair buffer (causal
    attention over it), matching its training-time causal semantics.
    """

    def __init__(self, head: EagleDraftHead, target_model, device):
        self.head = head.eval()
        self.target_model = target_model
        self.device = device
        self.embed = target_model.model.embed_tokens
        self.lm_head = target_model.lm_head

    def draft(self, f_seed: Tensor, t_seed: Tensor, k: int,
              temperature: float = 0.0, rng=None) -> tuple[Tensor, list[Tensor]]:
        """Chain k draft tokens -> (tokens [1, k], draft_probs [1,1,V] rows).

        f_seed [1, 1, h]: target's ACTUAL second-to-top feature at index M-2.
        t_seed [1, 1]:    target's ACTUAL last token at index M-1.
        greedy (temperature <= 0): probs is [] (argmax acceptance).
        sampled: probs are the per-step fp32 softmax rows for min(1, p_t/p_d).
        """
        greedy = temperature <= 0.0
        bf = f_seed.clone()       # growing feature buffer [1, j, h]
        bt = t_seed.clone()       # growing token buffer  [1, j]
        tokens: list[Tensor] = []
        probs: list[Tensor] = []
        for _ in range(k):
            emb = self.embed(bt)                       # [1, j, h]
            fused = torch.cat([bf, emb], dim=-1)       # [1, j, 2h]
            f_hat = self.head(fused)                   # [1, j, h] (causal)
            top = run_top_layer(self.target_model, f_hat[:, -1:], self.device)
            logits = self.lm_head(top)                 # [1, 1, V]
            if greedy:
                tok = logits.argmax(dim=-1)            # [1, 1]
            else:
                p = _normalize_probs(logits, temperature)
                tok = torch.multinomial(p.view(1, -1), 1, generator=rng)
                probs.append(p)
            tokens.append(tok)
            bf = torch.cat([bf, f_hat[:, -1:]], dim=1)  # chain predicted feature
            bt = torch.cat([bt, tok], dim=1)            # chain sampled token
        return torch.cat(tokens, dim=1), probs


def _next_seed_feature(n_acc: int, vfeats: Tensor, f_last: Tensor) -> Tensor:
    """The seed feature f at index M'-2 after a phase (plan §2.3).

    n_acc >= 1: the accepted proposals' last position (index M+n_acc-1) was
    covered by the verify pass -> row n_acc-1 of vfeats.
    n_acc == 0: the seed is the pre-phase last feature (index M-1 = M'-2).
    Pure helper so the alignment is unit-testable.
    """
    return vfeats[:, n_acc - 1:n_acc] if n_acc >= 1 else f_last


# --------------------------------------------------------------------------
# The decode loop
# --------------------------------------------------------------------------
@torch.no_grad()
def eagle_speculative_decode(
    target: ModelHandle,
    head: EagleDraftHead,
    prompt_ids: Tensor,
    max_new_tokens: int,
    draft_length: int = 4,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[Tensor, dict]:
    """EAGLE decode with the trained draft head. Returns (output_ids, stats).

    Drop-in analogue of src/speculative.py's speculative_decode — same
    signature shape, same accept/reject/resample math, same stats keys. The
    target's cache is the only cache (the head is stateless per phase).
    """
    greedy = temperature <= 0.0
    device = prompt_ids.device
    rng = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    if prompt_ids.shape[1] < 2:
        raise SystemExit("prompt too short — need >= 2 tokens for the seed pair (f_{M-2}, t_{M-1})")

    provider = EagleHeadProvider(head, target.model, device)

    target.reset()
    prefill = target.forward(prompt_ids, capture_hidden=True)
    feats = prefill.hidden_states[-2]         # [1, M, h] — second-to-top layer
    f_last = feats[:, -1:, :]                 # f at index M-1
    f_seed = feats[:, -2:-1, :]               # f at index M-2 (the seed pair's feature)
    t_seed = prompt_ids[:, -1:]               # t at index M-1 (the seed pair's token)

    output = prompt_ids.clone()
    n_generated = 0
    stats = {
        "iterations": 0,
        "proposed": 0,
        "accepted": 0,
        "resampled": 0,
        "bonus": 0,
        "accepted_per_step": [],
    }

    while n_generated < max_new_tokens:
        C_len = output.shape[1]               # cache length M at phase start
        k = min(draft_length, max_new_tokens - n_generated)
        if k < 1:
            break
        stats["iterations"] += 1

        # ---- 1. draft chain: k proposals for positions M..M+k-1 ----
        draft_tokens, draft_probs = provider.draft(f_seed, t_seed, k, temperature, rng)

        # ---- 2. verify all k in ONE parallel target forward ----
        p0 = target.pending_logits            # target's p_M
        tout = target.forward(draft_tokens, capture_hidden=True)
        vfeats = tout.hidden_states[-2]       # [1, k, h] — features at M..M+k-1
        t_rows = torch.cat([p0, tout.logits], dim=1)  # [1, k+1, V]: x_1..x_k + bonus

        # ---- 3. accept/reject — identical math to src/speculative.py ----
        if greedy:
            target_argmax = t_rows[:, :-1].argmax(dim=-1)   # [1, k]
            accepted = target_argmax == draft_tokens
            n_acc = int((~accepted).nonzero(as_tuple=True)[1][0]) if (~accepted).any() else k
            correction = target_argmax[:, n_acc:n_acc + 1] if n_acc < k else None
        else:
            t_probs = _normalize_probs(t_rows, temperature)  # [1, k+1, V]
            d_probs = torch.cat(draft_probs, dim=1)          # [1, k, V]
            p_t_x = t_probs[:, :k].gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)
            p_d_x = d_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)
            ratio = torch.clamp(p_t_x / p_d_x.clamp_min(1e-12), max=1.0)
            u = torch.rand(k, device=device, generator=rng)
            accepted = u < ratio[0]
            n_acc = int((~accepted).nonzero(as_tuple=True)[0][0]) if (~accepted).any() else k
            correction = None
            if n_acc < k:
                residual = _residual_distribution(t_probs[0, n_acc], draft_probs[n_acc][0])
                correction = torch.multinomial(residual.view(1, -1), 1, generator=rng)

        # ---- 4. append + cache rollback + seed refresh (plan §2.3) ----
        kept = list(draft_tokens.split(1, dim=1))[:n_acc]
        if n_acc < k:
            kept.append(correction)
            stats["resampled"] += 1
            target.crop_cache(C_len + n_acc)
            t_out = target.forward(correction, capture_hidden=True)
            f_seed = _next_seed_feature(n_acc, vfeats, f_last)
        else:
            # all k accepted — bonus from the target's own distribution
            if greedy:
                bonus = t_rows[:, -1:].argmax(dim=-1)
            else:
                bonus = torch.multinomial(t_probs[0, -1].view(1, -1), 1, generator=rng)
            kept.append(bonus)
            stats["bonus"] += 1
            t_out = target.forward(bonus, capture_hidden=True)
            f_seed = _next_seed_feature(k, vfeats, f_last)
        f_last = t_out.hidden_states[-2][:, -1:]   # f at new last index M'-1
        t_seed = kept[-1]                          # t at new last index M'-1

        new_tokens = torch.cat(kept, dim=1)        # [1, n_acc + 1]
        stats["proposed"] += k
        stats["accepted"] += n_acc
        stats["accepted_per_step"].append(n_acc)

        n_generated += new_tokens.shape[1]
        output = torch.cat([output, new_tokens], dim=1)

    output = output[:, : prompt_ids.shape[1] + max_new_tokens]
    return output, stats


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------
def load_head_checkpoint(model, out_dir: str | Path = "data/draft-head",
                         n_layers: int = 1, device: str = "cuda") -> EagleDraftHead:
    """Build the head from the target's top N layers + the best checkpoint.

    Loads head_fc.pt + head_layers.pt (the Phase-0 best, fp32), casts the head
    to the model's dtype (fp16 at deploy), and shares the target's rotary so
    positional embeddings are identical to the decode path.
    """
    out_dir = Path(out_dir)
    head = EagleDraftHead(
        model.config.hidden_size,
        [model.model.layers[-i] for i in range(1, n_layers + 1)],
        model.model.rotary_emb,
    )
    fc_path, layer_path = out_dir / "head_fc.pt", out_dir / "head_layers.pt"
    if not (fc_path.exists() and layer_path.exists()):
        raise SystemExit(f"no head checkpoint in {out_dir} — run src/train_eagle_head.py first")
    head.fc.load_state_dict(torch.load(fc_path, weights_only=True))
    sd = torch.load(layer_path, weights_only=True)
    n_saved = len({k.split(".", 1)[0] for k in sd})
    if n_saved != n_layers:
        raise SystemExit(f"checkpoint has {n_saved} decoder layer(s) but n_layers={n_layers} "
                         f"— architecture mismatch")
    head.layers.load_state_dict(sd)
    head = head.to(next(model.parameters()).dtype).to(device).eval()
    head.rotary_emb = model.model.rotary_emb  # shared, frozen, read-only at inference
    return head


# --------------------------------------------------------------------------
# CPU self-test (no GPU needed)
# --------------------------------------------------------------------------
def _synthetic_target(seed: int = 7):
    """Tiny Qwen2 model on CPU — same recipe as train_eagle_head.py's self-test."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=500, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, rope_theta=10000.0, torch_dtype="float32",
    )
    model = Qwen2ForCausalLM(cfg).eval()
    return model, cfg


def _assert_accounting(stats: dict, max_new: int) -> None:
    """Token-accounting invariant: every phase emits n_acc accepted proposals
    + exactly one more token (correction or bonus), so the total emitted is
    sum(accepted_per_step) + #phases. resampled/bonus count PHASES (not
    tokens) — a full-accept phase contributes k accepted + 1 bonus."""
    assert stats["accepted"] == sum(stats["accepted_per_step"]), stats
    assert sum(stats["accepted_per_step"]) + stats["resampled"] + stats["bonus"] == max_new, stats
    assert stats["accepted"] <= stats["proposed"], stats


def run_self_test() -> None:
    """Structural wiring test on a synthetic CPU model: shapes, seed
    bookkeeping, determinism, and token accounting across both cache paths.

    Everything is seeded, so the path mix is deterministic — the real
    acceptance + token-identical gates run on GPU (benchmark_eagle.py).
    """
    model, cfg = _synthetic_target()

    # a) seed-feature alignment helper (plan §2.3) — the seed-alignment trap
    vfeats = torch.randn(1, 4, cfg.hidden_size)
    f_last = torch.randn(1, 1, cfg.hidden_size)
    assert torch.equal(_next_seed_feature(0, vfeats, f_last), f_last)
    assert torch.equal(_next_seed_feature(2, vfeats, f_last), vfeats[:, 1:2])
    assert torch.equal(_next_seed_feature(4, vfeats, f_last), vfeats[:, 3:4])

    # b) draft provider contract: shapes + greedy determinism
    head = EagleDraftHead(cfg.hidden_size, [model.model.layers[-1]], model.model.rotary_emb)
    provider = EagleHeadProvider(head, model, torch.device("cpu"))
    f_seed = torch.randn(1, 1, cfg.hidden_size)
    t_seed = torch.tensor([[3]])
    toks, probs = provider.draft(f_seed, t_seed, k=5, temperature=0.0)
    assert toks.shape == (1, 5) and toks.dtype == torch.int64
    assert probs == []
    toks2, _ = provider.draft(f_seed, t_seed, k=5, temperature=0.0)
    assert torch.equal(toks, toks2), "greedy draft must be deterministic"

    # c) decode loop end-to-end: warm (untrained) head — an untrained head
    #    mostly rejects on the 500-vocab synthetic, so exercise both cache
    #    paths across two heads and check accounting, not the accept mix.
    prompt = torch.tensor([[11, 22, 33, 44]])
    out, stats = eagle_speculative_decode(
        ModelHandle(model, None), head, prompt, max_new_tokens=8,
        draft_length=4, temperature=0.0)
    assert out.shape == (1, prompt.shape[1] + 8)
    assert stats["iterations"] >= 2
    _assert_accounting(stats, 8)

    # determinism end-to-end
    out2, _ = eagle_speculative_decode(
        ModelHandle(model, None), head, prompt, max_new_tokens=8,
        draft_length=4, temperature=0.0)
    assert torch.equal(out, out2)

    # d) broken head (non-identity FC) -> draft diverges -> partial-accept
    #    path (crop + correction + f_seed = f_last) is guaranteed exercised.
    head_bad = EagleDraftHead(cfg.hidden_size, [model.model.layers[-1]], model.model.rotary_emb)
    with torch.no_grad():
        head_bad.fc.weight.zero_()
        head_bad.fc.bias.zero_()
    out3, stats3 = eagle_speculative_decode(
        ModelHandle(model, None), head_bad, prompt, max_new_tokens=8,
        draft_length=4, temperature=0.0)
    assert out3.shape == (1, prompt.shape[1] + 8)
    assert stats3["resampled"] >= 1, f"expected partial-accept phases, got {stats3}"
    _assert_accounting(stats3, 8)

    print(f"eagle self-test OK: shapes ✓ seed-alignment ✓ determinism ✓ "
          f"accounting ✓ (warm: acc={stats['accepted']}/prop={stats['proposed']}, "
          f"resampled={stats['resampled']}, bonus={stats['bonus']}; "
          f"broken: resampled={stats3['resampled']})")


if __name__ == "__main__":
    run_self_test()
