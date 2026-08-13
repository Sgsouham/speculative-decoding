"""Core vanilla speculative decoding loop.

Algorithm (Leviathan et al. 2023 / Chen et al. 2023):
  1. Draft: the small model proposes k candidate tokens autoregressively
     (its own KV cache stays warm across steps).
  2. Verify: the target runs ONE parallel forward pass over the k candidates
     (its own cache) — a single call yields k+1 distributions: one for each
     draft token plus the "bonus" token that follows the last draft token.
  3. Accept/reject:
       - greedy (temperature == 0): deterministic — accept while the target's
         argmax equals the draft token; on first mismatch emit the target argmax.
       - sampled (temperature > 0): accept x_i with prob min(1, p_t(x_i)/p_d(x_i)).
  4. Resample: on rejection at position j, draw the correction from
     max(0, p_t - p_d)_+ normalized — the line that makes the scheme provably
     distribution-preserving.
  5. Rinse: continue from the last accepted position; both caches stay aligned
     (cropped to the accepted prefix on rejection, appended with the bonus on
     full acceptance).
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.models import ModelHandle


def _normalize_probs(logits: Tensor, temperature: float) -> Tensor:
    """Temperature-scaled softmax in fp32 (fp16-honest: prob math is fp32)."""
    return torch.softmax(logits.float() / max(temperature, 1e-9), dim=-1)


def _residual_distribution(p_t: Tensor, p_d: Tensor) -> Tensor:
    """max(0, p_t - p_d) normalized — the correction distribution on rejection.

    Pure function (unit-testable with an artificial draft/target pair, §6.2).
    """
    residual = torch.clamp(p_t - p_d, min=0.0)
    total = residual.sum()
    if total <= 0.0:  # numerical guard — unreachable on a genuine rejection
        return p_t
    return residual / total


@torch.no_grad()
def speculative_decode(
    draft: ModelHandle,
    target: ModelHandle,
    prompt_ids: Tensor,
    max_new_tokens: int,
    draft_length: int = 4,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[Tensor, dict]:
    """Vanilla speculative decoding. Returns (output_ids, stats).

    output_ids: [1, prompt_len + n] on the same device as prompt_ids.
    temperature == 0 -> deterministic greedy (argmax-match acceptance);
    temperature > 0  -> rejection-sampled (min(1, p_t/p_d) acceptance + residual).
    """
    greedy = temperature <= 0.0
    device = prompt_ids.device
    rng = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    draft.reset()
    target.reset()

    # Prefill both caches with the prompt; pending_logits then predicts the
    # first new token for each model.
    draft.forward(prompt_ids)
    target.forward(prompt_ids)
    n_ctx = prompt_ids.shape[1]

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
        C_len = output.shape[1]  # accepted-sequence length incl. prompt (pre-draft)
        k = min(draft_length, max_new_tokens - n_generated)
        if k < 1:
            break
        stats["iterations"] += 1

        # ---- 1. draft k tokens (autoregressive; draft cache stays warm) ----
        draft_tokens: list[Tensor] = []
        draft_probs: list[Tensor] = []  # fp32 [1, 1, V] rows (sampled mode only)
        for i in range(k):
            row = draft.pending_logits if i == 0 else draft.forward(draft_tokens[-1]).logits[:, -1:, :]
            if greedy:
                tok = row.argmax(dim=-1)  # [1, 1]
            else:
                probs = _normalize_probs(row, temperature)
                tok = torch.multinomial(probs.view(1, -1), 1, generator=rng)
                draft_probs.append(probs)
            draft_tokens.append(tok)
        draft.forward(draft_tokens[-1])  # align draft cache: ctx + x_1..x_k

        # ---- 2. verify all k positions in ONE parallel target forward ----
        p0 = target.pending_logits  # predicts x_1
        tout = target.forward(torch.cat(draft_tokens, dim=1))
        t_rows = torch.cat([p0, tout.logits], dim=1)  # [1, k+1, V]: x_1..x_k + bonus

        # ---- 3. accept/reject ----
        draft_ids = torch.cat(draft_tokens, dim=1)  # [1, k]
        if greedy:
            target_argmax = t_rows[:, :-1].argmax(dim=-1)  # [1, k]
            accepted = target_argmax == draft_ids
            n_acc = int((~accepted).nonzero(as_tuple=True)[1][0]) if (~accepted).any() else k
            # keep 2-D [1, 1] — 1-D input_ids corrupts the model's rope layout
            correction = target_argmax[:, n_acc : n_acc + 1] if n_acc < k else None
        else:
            t_probs = _normalize_probs(t_rows, temperature)  # [1, k+1, V]
            d_probs = torch.cat(draft_probs, dim=1)  # [1, k, V]
            p_t_x = t_probs[:, :k].gather(2, draft_ids.unsqueeze(-1)).squeeze(-1)  # [1, k]
            p_d_x = d_probs.gather(2, draft_ids.unsqueeze(-1)).squeeze(-1)  # [1, k]
            ratio = torch.clamp(p_t_x / p_d_x.clamp_min(1e-12), max=1.0)  # [1, k]
            u = torch.rand(k, device=device, generator=rng)
            accepted = u < ratio[0]
            n_acc = int((~accepted).nonzero(as_tuple=True)[0][0]) if (~accepted).any() else k
            correction = None
            if n_acc < k:
                residual = _residual_distribution(t_probs[0, n_acc], draft_probs[n_acc][0])
                correction = torch.multinomial(residual.view(1, -1), 1, generator=rng)

        # ---- 4. append accepted prefix + (correction | bonus); align caches ----
        kept = draft_tokens[:n_acc]
        if n_acc < k:
            kept.append(correction)
            stats["resampled"] += 1
            # caches hold C + x_1..x_k — roll back to the accepted prefix (which
            # includes ALL prior iterations' tokens: C_len, not n_ctx!), then append
            for handle in (draft, target):
                handle.crop_cache(C_len + n_acc)
                handle.forward(correction)
        else:
            # all k accepted — the bonus token comes from the target's own
            # distribution at position k+1 (what keeps the scheme exact)
            if greedy:
                bonus = t_rows[:, -1:].argmax(dim=-1)  # [1, 1] — keep 2-D
            else:
                bonus = torch.multinomial(t_probs[0, -1].view(1, -1), 1, generator=rng)
            kept.append(bonus)
            stats["bonus"] += 1
            # caches already hold ctx + x_1..x_k — append the target's bonus token
            for handle in (draft, target):
                handle.forward(bonus)

        new_tokens = torch.cat(kept, dim=1)  # [1, n_acc + 1]
        stats["proposed"] += k
        stats["accepted"] += n_acc
        stats["accepted_per_step"].append(n_acc)

        n_generated += new_tokens.shape[1]
        output = torch.cat([output, new_tokens], dim=1)

    output = output[:, : prompt_ids.shape[1] + max_new_tokens]
    return output, stats


@torch.no_grad()
def autoregressive_decode(
    target: ModelHandle,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[Tensor, dict]:
    """Plain target-only autoregressive decode through the SAME forward machinery
    (identical timing path — the §7 baseline; also the greedy equivalence reference)."""
    greedy = temperature <= 0.0
    device = prompt_ids.device
    rng = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    target.reset()
    target.forward(prompt_ids)
    output = prompt_ids.clone()
    for _ in range(max_new_tokens):
        row = target.pending_logits
        if greedy:
            tok = row.argmax(dim=-1)
        else:
            probs = _normalize_probs(row, temperature)
            tok = torch.multinomial(probs.view(1, -1), 1, generator=rng)
        target.forward(tok)
        output = torch.cat([output, tok], dim=1)
    return output, {"iterations": max_new_tokens}
