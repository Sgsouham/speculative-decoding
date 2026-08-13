"""eagle_head.py — the EAGLE-1-style draft head (architecture + decode path).

The head (arXiv 2401.15077 §3.2):
    FC(2h -> h)  +  N transformer decoder layers (default 1), deep-copied
    from the target's TOP N decoder layers. The target's embedding / top
    layers / norm / LM head are FROZEN and reused. Per position i:
    [f_i ; embed(t_{i+1})] -> f̂_{i+1}, and the frozen LM head maps the
    predicted feature to the draft distribution.

Lives in src/ so the decode engine, the tests, and the training script
(src/train_eagle_head.py) can all import it — the EAGLE code lives in src/
like every other module in this repo.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor


def causal_mask(seq: int, device, dtype) -> torch.Tensor:
    """Additive 4-D causal mask [1,1,S,S] (0 = attend, -inf = blocked)."""
    mask = torch.full((1, 1, seq, seq), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


class EagleDraftHead(nn.Module):
    """FC(2h -> h) + N decoder layers (copies of the target's TOP N layers).

    The target's embedding / top layers / norm / LM head are frozen and used
    externally by the training loop and the decode engine. FC is
    identity-initialized on the feature half so the head starts as the
    identity feature map; the decoder layers are warm-started from the
    target's own weights (EAGLE's trick), so a 2-layer head is a deeper copy
    of the target's final computation rather than random capacity.
    """

    def __init__(self, hidden: int, decoder_layers: list[nn.Module], rotary_emb: nn.Module):
        super().__init__()
        self.fc = nn.Linear(2 * hidden, hidden)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:, :hidden] = torch.eye(hidden)
            self.fc.bias.zero_()
        self.layers = nn.ModuleList(copy.deepcopy(l).float() for l in decoder_layers)  # fp32
        self.rotary_emb = rotary_emb
        self.hidden = hidden

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """fused [B, S, 2h] -> predicted next features [B, S, h] (f̂_{i+1}).

        Causal over the input sequence (each row attends to itself and the
        rows before it) — at decode time the engine feeds the phase's growing
        pair buffer, so later draft steps keep attention context (paper §3.1:
        the input is a feature sequence + an advanced token sequence).
        """
        x = self.fc(fused)
        b, s, _ = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary_emb(x, pos)
        mask = causal_mask(s, x.device, x.dtype)
        for layer in self.layers:
            out = layer(
                x,
                attention_mask=mask,
                position_ids=pos,
                position_embeddings=(cos, sin),
                use_cache=False,
            )
            x = out[0] if isinstance(out, tuple) else out
        return x


def run_top_layer_grad(model, x: torch.Tensor, device) -> torch.Tensor:
    """Decode path (frozen top layer + final norm) for the CE term of
    --loss eagle — the same shared last mile the eval gate uses, but
    grad-enabled so the CE gradient flows through it to the head's predicted
    feature. The target's params are requires_grad=False (frozen in main), so
    no gradients are stored for them. Casts x to the model's dtype (fp16 in
    production; fp32 in the CPU self-test)."""
    dt = next(model.model.layers[-1].parameters()).dtype
    b, s, _ = x.shape
    pos = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
    cos, sin = model.model.rotary_emb(x.to(dt), pos)
    mask = causal_mask(s, device, dt)
    out = model.model.layers[-1](
        x.to(dt), attention_mask=mask, position_ids=pos,
        position_embeddings=(cos, sin), use_cache=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return model.model.norm(out)


@torch.no_grad()
def run_top_layer(model, x: torch.Tensor, device) -> torch.Tensor:
    """no-grad wrapper of run_top_layer_grad for eval/precompute (identical
    decode path; the self-test's synthetic fp32 model works too, since the
    dtype cast is internal)."""
    return run_top_layer_grad(model, x, device)
