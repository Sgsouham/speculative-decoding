"""train_head.py — train the EAGLE-1-style draft head.

Consumes the feature cache produced by draft-head/collect_features.py
(data/draft-head/wikitext2/): fp16 second-to-top-layer features + token ids.

Head (arXiv 2401.15077 §3.2):
    FC(2h -> h)  +  N transformer decoder layers (default 1), deep-copied
    from the target's TOP N decoder layers (the layers right after the
    cached feature layer). The target's embedding / top layers / norm / LM
    head are FROZEN and reused. Per position i:  [f_i ; embed(t_{i+1})] ->
    f̂_{i+1}. FC is identity-initialized on the feature half ([f_i ; e] ->
    f_i), a principled warm start. --n-layers 2 is the capacity probe
    (Aug 11).

Loss (--loss, the objective lever — Aug 11):
    mse    pure feature MSE (the original Phase-0 loss; 100 epochs of this
           plateaued at ~0.485 greedy_agreement while val_mse kept falling).
    eagle  the paper's ACTUAL loss (arXiv 2401.15077 §3.2): Smooth L1 on the
           predicted features + ce_weight × cross-entropy between the frozen
           decode path's logits and the true token two ahead (t_{i+2}). The
           CE term optimizes the argmax agreement we gate on, not just the
           features. This is the probe after data (done) and depth (2-layer
           probe: negative) failed to move agreement.

Gate metric (plan §4 step 4a): GREEDY AGREEMENT = the fraction of positions
where the head's top-1 token equals the target's own top-1 token, measured
on a held-out tail of the cache. This is the acceptance-relevant statistic
for greedy speculative decoding — in greedy, a draft token is accepted iff
it is the target's argmax. Vanilla baseline: ~0.35 (M3, real text). If the
head is not clearly above that, STOP and reconsider (plan §4 gate).

Training precision: head params fp32 (small, ~50M — fp32 costs nothing and
avoids fp16 instability), cache features fp16 upcast per batch, MSE in fp32.
The engine casts the head to fp16 at deploy.

Outputs (--out, default data/draft-head/):
    head_fc.pt        FC state dict (fp32, best epoch)
    head_layers.pt    decoder-layer stack state dict (fp32, best epoch)
    train_report.json per-epoch val metrics + gate verdict

Usage (WSL2):
    HF_HOME=/mnt/d/projects/hf-cache uv run python draft-head/train_head.py --epochs 5
    HF_HOME=/mnt/d/projects/hf-cache uv run python draft-head/train_head.py --resume --epochs 5
    HF_HOME=/mnt/d/projects/hf-cache uv run python draft-head/train_head.py --loss eagle --epochs 10  # paper loss: Smooth L1 + 0.1*CE
    HF_HOME=/mnt/d/projects/hf-cache uv run python draft-head/train_head.py --self-test

--resume: continue from the BEST saved checkpoint (head_fc.pt /
head_layers.pt) and the epoch numbering in train_report.json, instead of
re-training from the identity init. A checkpoint's layer count must match
--n-layers (a 1-layer head cannot be resumed as 2-layer). Same LR unless --lr is passed; the best-checkpoint tracking
only overwrites the saved weights when greedy_agreement improves, so a resume
that doesn't beat the previous best leaves the checkpoint untouched.

Live stats: per-step train loss + per-epoch val metrics are written to
TensorBoard under --logdir (default data/draft-head/runs/<run-tag>/, flush every 5s).
A resume reuses the previous run's tag so the curves stay continuous. Watch:

    uv run tensorboard --logdir data/draft-head/runs --port 6006

...then open http://localhost:6006 (from the Windows browser; WSL shares
localhost).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import chunk_idx as _chunk_idx  # noqa: E402
from src.config import load_config, resolve_model_id  # noqa: E402
from src.models import load_model  # noqa: E402


# --------------------------------------------------------------------------
# Data + geometry helpers
# --------------------------------------------------------------------------
def make_pairs(feats: torch.Tensor, toks: torch.Tensor):
    """Pairs for one contiguous cached block.

    Cache positions 0..n-1 carry (feature f_i, token t_i). Pair i (0..n-2):
        input  = (f_i, t_{i+1})   -> fused [f_i; embed(t_{i+1})]
        target = f_{i+1}          (MSE)  and  t_{i+1}  (top-1 accuracy ref)
    The tail position (n-1) has no next token and is dropped — matching the
    cache design (collect_features.py drops one position per block).
    """
    return feats[:-1], toks[1:], feats[1:], toks[1:]


def causal_mask(seq: int, device, dtype) -> torch.Tensor:
    """Additive 4-D causal mask [1,1,S,S] (0 = attend, -inf = blocked)."""
    mask = torch.full((1, 1, seq, seq), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def load_chunks(cache_dir: Path, val_frac: float, block_len: int = 1024):
    """(train_blocks, val_blocks, total_tokens).

    The cache holds FLAT ~100K-token chunks (collect_features.py accumulates
    many 1024-token blocks into one file), but the head's causal attention is
    O(block_len²) — a whole chunk as one sequence OOMs instantly (Aug 10:
    "Tried to allocate 37 GiB" on the 50K-token val slice). So re-block every
    chunk into block_len-token sequences here. The original block boundaries
    are lost in the flat files; any consistent re-blocking is valid because
    every cached position already carries its own causally-complete feature.

    Deterministic holdout: the last `val_frac`-by-token blocks are
    validation, the rest training. Returns fp16-feature/int64-token blocks on
    CPU.
    """
    paths = sorted(cache_dir.glob("chunk_*.features.pt"), key=_chunk_idx)
    if not paths:
        raise SystemExit(f"no chunks in {cache_dir} — run draft-head/collect_features.py first")
    blocks = []
    for p in paths:
        feats = torch.load(p, weights_only=True)
        toks = torch.load(p.with_name(p.name.replace("features", "tokens")), weights_only=True)
        if feats.shape[0] != toks.shape[0]:
            raise SystemExit(f"{p.name}: feature/token count mismatch")
        n = feats.shape[0] // block_len
        for i in range(n):
            s, e = i * block_len, (i + 1) * block_len
            blocks.append((feats[s:e], toks[s:e]))
    if len(blocks) < 2:
        raise SystemExit(f"cache too small for a train/val split ({len(blocks)} block(s)) — "
                         f"run collect_features.py with a bigger budget first")
    total = sum(f.shape[0] for f, _ in blocks)
    val_n = max(1, min(int(total * val_frac) // block_len, len(blocks) - 1))
    val_blocks = blocks[-val_n:]
    train_blocks = blocks[:-val_n]
    return train_blocks, val_blocks, total


# --------------------------------------------------------------------------
# Model wiring
# --------------------------------------------------------------------------
class EagleDraftHead(nn.Module):
    """FC(2h -> h) + N decoder layers (copies of the target's TOP N layers).

    The target's embedding / top layers / norm / LM head are frozen and used
    externally by the training loop. FC is identity-initialized on the
    feature half so the head starts as the identity feature map; the decoder
    layers are warm-started from the target's own weights (EAGLE's trick), so
    a 2-layer head is a deeper copy of the target's final computation rather
    than random capacity.
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
        """fused [B, S, 2h] -> predicted next features [B, S, h] (f̂_{i+1})."""
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


def eagle_loss(head, fused, tgt_f, labels, model, device, ce_weight: float = 0.1):
    """EAGLE-style loss (arXiv 2401.15077 §3.2):
    L = SmoothL1(f̂_{i+1}, f_{i+1}) + ce_weight · CE(logits, t_{i+2}).

    Deliberate simplification vs the paper: L_cls there is a SOFT-target CE
    (CE between the target's distribution p_{i+2} and the predicted p̂_{i+2});
    we use hard labels (the actual cached token t_{i+2}) — the standard
    practical variant, and it costs one lm_head pass, not two.

    labels are the TRUE tokens TWO ahead (t_{i+2}) — the alignment the fixed
    top1_acc metric uses (head logits at i+1 are a causal prediction of
    t_{i+2}); logits[:, :-1] drops the last predicted position to match.
    Returns (total, reg, cls) for live logging."""
    pred = head(fused).float()
    reg = F.smooth_l1_loss(pred, tgt_f)
    top = run_top_layer_grad(model, pred, device)
    logits = model.lm_head(top)[:, :-1].reshape(-1, model.lm_head.out_features)
    cls = F.cross_entropy(logits, labels.to(device))
    return reg + ce_weight * cls, reg, cls


def run_self_test() -> None:
    """Wiring test on a tiny synthetic Qwen2 model — CPU only, seconds, no GPU."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    # default use_sliding_window=False -> all layers full_attention;
    # sliding_window/max_window_layers omitted (strict dataclass wants an int)
    cfg = Qwen2Config(
        vocab_size=500, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, rope_theta=10000.0, torch_dtype="float32",
    )
    model = Qwen2ForCausalLM(cfg).eval()
    h = cfg.hidden_size
    head = EagleDraftHead(h, [model.model.layers[-1]], model.model.rotary_emb)
    feats = torch.randn(48, h)
    toks = torch.randint(0, cfg.vocab_size, (48,))
    f, t_next, tgt_f, tgt_t = make_pairs(feats, toks)
    fused = torch.cat([f, model.model.embed_tokens(t_next)], dim=-1).unsqueeze(0)
    pred = head(fused)
    assert pred.shape == (1, f.shape[0], h), pred.shape
    loss = F.mse_loss(pred, tgt_f.unsqueeze(0))
    loss.backward()
    assert head.fc.weight.grad is not None
    assert head.layers[0].self_attn.q_proj.weight.grad is not None

    # 2-layer variant: stack the target's top 2 layers, same wiring. Use
    # detached copies of the inputs so its backward graph is independent of
    # the first head's (whose saved tensors were freed above).
    head2 = EagleDraftHead(h, [model.model.layers[-1], model.model.layers[-2]],
                           model.model.rotary_emb)
    pred2 = head2(fused.detach())
    assert pred2.shape == (1, f.shape[0], h), pred2.shape
    F.mse_loss(pred2, tgt_f.unsqueeze(0)).backward()
    assert head2.layers[1].self_attn.q_proj.weight.grad is not None

    # checkpoint round-trip: ModuleList state_dict save -> n_saved count -> load
    # (guards the resume path — the most bug-prone new code)
    import io

    buf = io.BytesIO()
    torch.save(head2.layers.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=True)
    n_saved = len({k.split(".", 1)[0] for k in sd})
    assert n_saved == 2, n_saved
    head3 = EagleDraftHead(h, [model.model.layers[-1], model.model.layers[-2]],
                           model.model.rotary_emb)
    head3.layers.load_state_dict(sd)
    assert torch.equal(head3.layers[1].self_attn.q_proj.weight,
                       head2.layers[1].self_attn.q_proj.weight)
    top = run_top_layer(model, tgt_f.unsqueeze(0), torch.device("cpu"))
    logits = model.lm_head(top)
    assert logits.shape == (1, tgt_f.shape[0], cfg.vocab_size), logits.shape
    agree = float((logits.argmax(-1)[0] == tgt_t).float().mean())

    # eagle-loss path: CE must reach the head THROUGH the frozen top layer +
    # lm_head, with labels aligned to t_{i+2} (the fixed top1_acc alignment).
    # ehead is constructed BEFORE the freeze so its deep-copied layers stay
    # trainable (deepcopy preserves requires_grad).
    ehead = EagleDraftHead(h, [model.model.layers[-1]], model.model.rotary_emb)
    model.requires_grad_(False)
    fused2 = torch.cat([f, model.model.embed_tokens(t_next)], dim=-1).unsqueeze(0)
    loss_e, reg_e, cls_e = eagle_loss(
        ehead, fused2, tgt_f.unsqueeze(0), tgt_t[1:], model, torch.device("cpu"))
    loss_e.backward()
    assert ehead.fc.weight.grad is not None
    assert ehead.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.lm_head.weight.grad is None, "frozen decode path must not store grads"

    # load_chunks re-blocking test — regression guard for the OOM-class bug
    # (a flat ~100K-token chunk must never reach the model as one sequence).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for idx, n in ((0, 3000), (1, 2000)):
            f = torch.randn(n, h, dtype=torch.float16)
            t = torch.randint(0, cfg.vocab_size, (n,))
            torch.save(f, Path(tmp) / f"chunk_{idx:06d}.features.pt")
            torch.save(t, Path(tmp) / f"chunk_{idx:06d}.tokens.pt")
        tr, va, total = load_chunks(Path(tmp), val_frac=0.1, block_len=1024)
        assert total == 3 * 1024, total                      # 5000 // 1024 -> 3 full blocks
        assert all(f.shape[0] == 1024 for f, _ in tr + va)   # never a giant sequence
        assert len(va) == 1 and len(tr) == 2                 # 10% of 3 blocks -> 1 val

    # results-log render test: a synthetic report must produce valid markdown
    # (this runs only at the end of a real GPU run, so a format bug would
    # crash AFTER training — test it here).
    import tempfile as _tmp

    fake = {"epochs": [
        {"epoch": 41, "train_mse": 11.8, "val_mse": 15.9, "top1_acc": 0.317,
         "greedy_agreement": 0.470, "target_top1_acc": 0.514},
        {"epoch": 42, "train_mse": 11.6, "val_mse": 15.8, "top1_acc": 0.318,
         "greedy_agreement": 0.474, "target_top1_acc": 0.514},
    ], "config": {"n_layers": 2, "loss": "eagle", "ce_weight": 0.1,
                  "model": "qwen2.5-3b", "lr": 0.0005, "seed": 42,
                  "resume": True},
        "head_params": 162544640,
        "tensorboard_dir": "data/draft-head/runs/eagle_head_20260811_090004",
        "data": {"total": 3014781, "train": 2713303, "val": 301478},
        "best": {"epoch": 42, "greedy_agreement": 0.474, "target_top1_acc": 0.514},
        "gate": {"verdict": "BORDERLINE — above baseline but thin"}}
    with _tmp.TemporaryDirectory() as td:
        fake_log = Path(td) / "draft-head-training.md"
        write_results_log(fake, 0, log_path=fake_log)
        md = fake_log.read_text(encoding="utf-8")
        assert "| 41 |" in md and "| 42 |" in md and "0.474" in md
        assert "Smooth L1 + 0.1×CE" in md and "BORDERLINE" in md

    print(f"self-test OK: pred {tuple(pred.shape)} loss {loss.item():.4f} | "
          f"target-logits {tuple(logits.shape)} | top1-vs-actual {agree:.2f} | "
          f"eagle-loss reg {reg_e.item():.4f} cls {cls_e.item():.4f} | "
          f"head params {sum(p.numel() for p in head.parameters()):,}")


# --------------------------------------------------------------------------
# Committable results log
# --------------------------------------------------------------------------
def write_results_log(report: dict, epochs_before: int,
                      log_path: Path | None = None) -> None:
    """Append a run section to the COMMITTABLE log results/draft-head-training.md.

    The heavy per-epoch JSON lives under data/ (gitignored — GBs of tensors).
    This markdown is the small, human-readable, tracked record so the training
    story ships to GitHub without the cache. Append-only; each resume appends
    only ITS OWN new epochs (report accumulates across runs). log_path is
    injectable for the self-test (defaults to results/draft-head-training.md)."""
    log = log_path or (REPO_ROOT / "results" / "draft-head-training.md")
    log.parent.mkdir(parents=True, exist_ok=True)
    es = report.get("epochs", [])
    new_es = es[epochs_before:]
    if not new_es:
        return
    cfg = report.get("config", {})
    run_tag = Path(report.get("tensorboard_dir", "")).name or "run"
    data = report.get("data", {})
    best = report.get("best") or {}
    gate = report.get("gate") or {}
    head = "FC + %d decoder layer%s (warm from target's top %d)" % (
        cfg.get("n_layers", 1), "s" if cfg.get("n_layers", 1) != 1 else "",
        cfg.get("n_layers", 1))
    loss = cfg.get("loss", "mse")
    if loss == "eagle":
        loss_desc = f"eagle (Smooth L1 + {cfg.get('ce_weight', 0.1)}×CE)"
    else:
        loss_desc = "mse (pure feature MSE)"
    resume = "resume" if cfg.get("resume") else "fresh"
    rows = []
    prev = None
    for e in new_es:
        d = f"{e['greedy_agreement'] - prev['greedy_agreement']:+.3f}" if prev else "—"
        rows.append(
            f"| {e['epoch']} | {e.get('train_mse', float('nan')):.2f} "
            f"| {e['val_mse']:.2f} | {e['top1_acc']:.3f} "
            f"| {e['greedy_agreement']:.3f} | {d} |")
        prev = e
    verdict = (gate.get("verdict") or "?").split(" — ")[0]
    section = (
        f"\n## {run_tag} · {resume} · n_layers {cfg.get('n_layers', 1)} · "
        f"{loss_desc}\n"
        f"- model {cfg.get('model', '?')} · lr {cfg.get('lr', '?')} · seed {cfg.get('seed', '?')} · "
        f"head = {head} ({report.get('head_params', '?'):,} params)\n"
        f"- data: {data.get('total', '?'):,} cached → {data.get('train', '?'):,} train / "
        f"{data.get('val', '?'):,} val · epochs {new_es[0]['epoch']}–{new_es[-1]['epoch']}\n"
        f"- target_top1_acc (corpus predictability ceiling): "
        f"{best.get('target_top1_acc', float('nan')):.3f}\n"
        f"- **best greedy_agreement {best.get('greedy_agreement', float('nan')):.3f} "
        f"@ epoch {best.get('epoch', '?')}** · gate: **{verdict}**\n"
        f"\n| epoch | train_mse | val_mse | top1_acc | greedy_agreement | Δ |\n"
        f"|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"appended run to {log}", flush=True)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="draft-head: train the EAGLE-1-style draft head (FC + N decoder layers)")
    ap.add_argument("--n-layers", type=int, default=1,
                    help="decoder layers in the head, warm-copied from the target's top N "
                         "(default 1; 2 = the Aug 11 capacity probe)")
    ap.add_argument("--cache", default="data/draft-head/wikitext2", help="feature cache dir (from collect_features.py)")
    ap.add_argument("--out", default="data/draft-head", help="output dir for head weights + report")
    ap.add_argument("--model", default="qwen2.5-3b", help="target alias in config/default.yaml")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate")
    ap.add_argument("--batch-blocks", type=int, default=4,
                    help="gradient accumulation over N blocks (effective batch ~N*1022 pairs)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out tail of the cache")
    ap.add_argument("--logdir", default="data/draft-head/runs",
                    help="TensorBoard log root (each run gets its own subdir; resume reuses it)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the best saved checkpoint + report (epoch numbering picks up)")
    ap.add_argument("--loss", choices=["mse", "eagle"], default="mse",
                    help="training objective: mse = pure feature MSE (original Phase-0 loss); "
                         "eagle = the paper's real loss — Smooth L1 on features + ce_weight × "
                         "cross-entropy on tokens (optimizes the argmax agreement we gate on). "
                         "The objective lever, Aug 11.")
    ap.add_argument("--ce-weight", type=float, default=0.1,
                    help="weight of the token cross-entropy term in --loss eagle "
                         "(the paper uses w_cls = 0.1; classification loss runs an order of "
                         "magnitude larger than the feature regression loss)")
    ap.add_argument("--self-test", action="store_true", help="tiny synthetic-model wiring test, then exit")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- model -----------------------------------------------------------
    t0 = time.time()
    cfg = load_config()
    model_id = resolve_model_id(cfg, "target", args.model)[1]
    print(f"loading target {args.model} ({model_id}) ...", flush=True)
    model, _ = load_model(model_id, device=args.device)
    print(f"model loaded in {time.time() - t0:.0f}s (hidden {model.config.hidden_size}, fp16)", flush=True)
    hidden = model.config.hidden_size

    # --- data ------------------------------------------------------------
    train_chunks, val_chunks, total_tokens = load_chunks(Path(args.cache), args.val_frac)
    n_train = sum(f.shape[0] for f, _ in train_chunks)
    n_val = sum(f.shape[0] for f, _ in val_chunks)
    print(f"cache: {total_tokens:,} tokens -> {n_train:,} train / {n_val:,} val", flush=True)

    # --- head + optimizer -------------------------------------------------
    head = EagleDraftHead(
        hidden,
        [model.model.layers[-i] for i in range(1, args.n_layers + 1)],
        model.model.rotary_emb,
    )
    head.to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"head: FC(2*{hidden}->{hidden}) + {args.n_layers} decoder layer(s) "
          f"(warm copies of the target's top {args.n_layers}) — {n_params:,} trainable params", flush=True)
    # Freeze the target AFTER the head deep-copied its layers. Everything
    # downstream of the head (top layer, final norm, LM head) stays frozen:
    # with --loss eagle the CE gradient flows through that shared machinery to
    # the head, and freezing means no gradients accumulate on the target's
    # params (they are not in the optimizer).
    for p in model.parameters():
        p.requires_grad_(False)

    # --- resume bookkeeping ------------------------------------------------
    report = {"epochs": [], "config": vars(args), "head_params": n_params,
              "data": {"total": total_tokens, "train": n_train, "val": n_val}}
    best = None
    start_epoch = 0
    if args.resume:
        fc_path, layer_path = out_dir / "head_fc.pt", out_dir / "head_layers.pt"
        legacy_path = out_dir / "head_layer.pt"   # pre-Aug-11 1-layer format
        have_layers = layer_path.exists() or (args.n_layers == 1 and legacy_path.exists())
        if not (fc_path.exists() and have_layers):
            raise SystemExit(f"--resume: no checkpoint in {out_dir} — run without --resume first")
        head.fc.load_state_dict(torch.load(fc_path, weights_only=True))
        if layer_path.exists():
            sd = torch.load(layer_path, weights_only=True)
            n_saved = len({k.split(".", 1)[0] for k in sd})
            if n_saved != args.n_layers:
                raise SystemExit(f"--resume: checkpoint has {n_saved} decoder layer(s) but "
                                 f"--n-layers {args.n_layers} — architecture mismatch; "
                                 f"train a different depth fresh")
            head.layers.load_state_dict(sd)
        else:  # legacy bare 1-layer state dict (guarded above)
            head.layers[0].load_state_dict(torch.load(legacy_path, weights_only=True))
        print(f"resumed weights from {fc_path.name} + "
              f"{layer_path.name if layer_path.exists() else legacy_path.name}", flush=True)
        report_path = out_dir / "train_report.json"
        if report_path.exists():
            try:
                prev = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prev = None  # corrupt report — continue from the checkpoint, fresh numbering
            epochs = prev.get("epochs") if isinstance(prev, dict) else None
            if isinstance(epochs, list) and epochs:
                start_epoch = epochs[-1]["epoch"]
                best = prev.get("best") or None  # seed FIRST (report aliases prev below)
                report = prev
                report["config"] = vars(args)
                report["data"] = {"total": total_tokens, "train": n_train, "val": n_val}
                report.pop("best", None)  # recomputed at the end (across both runs)
                if best:
                    print(f"resumed report: continuing from epoch {start_epoch} "
                          f"(previous best greedy_agreement {best['greedy_agreement']:.3f})",
                          flush=True)
                else:
                    print(f"resumed report: continuing from epoch {start_epoch}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    # --- tensorboard (live stats) -----------------------------------------
    tb_root = Path(args.logdir)
    tb_root.mkdir(parents=True, exist_ok=True)
    if args.resume and report.get("tensorboard_dir"):
        tb_dir = Path(report["tensorboard_dir"])   # keep one continuous curve set
    else:
        tb_dir = tb_root / f"eagle_head_{time.strftime('%Y%m%d_%H%M%S')}"
    report["tensorboard_dir"] = str(tb_dir)
    writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=5)
    print(f"tensorboard: {tb_dir}  (watch: uv run tensorboard --logdir {tb_root})", flush=True)

    # precompute the target's top-1 per val position ONCE (frozen; cheap)
    target_top1 = []
    with torch.no_grad():
        for feats, toks in val_chunks:
            _, _, tgt_f, _ = make_pairs(feats, toks)
            top = run_top_layer(model, tgt_f.unsqueeze(0).to(device), device)
            target_top1.append(model.lm_head(top).argmax(-1)[0].cpu())
    print(f"precomputed target top-1 for {n_val:,} val positions", flush=True)

    def evaluate() -> dict:
        head.eval()
        mse_sum = acc_sum = agree_sum = tgt_acc_sum = count = 0
        with torch.no_grad():
            for (feats, toks), tt1 in zip(val_chunks, target_top1):
                f, t_next, tgt_f, _ = make_pairs(feats, toks)
                e = model.model.embed_tokens(t_next.to(device))
                fused = torch.cat([f.float().to(device), e.float()], dim=-1).unsqueeze(0)
                pred = head(fused)
                mse_sum += F.mse_loss(pred.float(), tgt_f.float().unsqueeze(0).to(device)).item() * pred.shape[1]
                top = run_top_layer(model, pred.half(), device)
                htop1 = model.lm_head(top).argmax(-1)[0].cpu()   # P̂ over t_{i+2}
                n = htop1.numel()
                count += n
                # head logits at position i+1 predict the token TWO ahead (causal
                # LM), so the accuracy reference is toks[2:], NOT toks[1:]
                # (off-by-one, found Aug 10 reading the first run's report).
                # greedy_agreement vs tt1 is unaffected — both sides are
                # distributions over t_{i+2}.
                acc_sum += int((htop1[:-1] == toks[2:]).sum())
                agree_sum += int((htop1 == tt1).sum())
                tgt_acc_sum += int((tt1[:-1] == toks[2:]).sum())
        n_acc = max(1, count - len(val_chunks))   # Σ(n_b - 2) per block
        return {"val_mse": mse_sum / count,
                "top1_acc": acc_sum / n_acc,
                "greedy_agreement": agree_sum / count,
                "target_top1_acc": tgt_acc_sum / n_acc}

    tb_step = 0

    def train_epoch(epoch: int) -> float:
        nonlocal tb_step
        head.train()
        order = list(range(len(train_chunks)))
        random.shuffle(order)
        total_loss = reg_sum = cls_sum = 0.0
        opt.zero_grad()
        for bi, ci in enumerate(order):
            feats, toks = train_chunks[ci]
            f, t_next, tgt_f, _ = make_pairs(feats, toks)
            with torch.no_grad():  # frozen target — no gradients through the embedding
                e = model.model.embed_tokens(t_next.to(device))
            fused = torch.cat([f.float().to(device), e.float()], dim=-1).unsqueeze(0)
            tgt = tgt_f.float().unsqueeze(0).to(device)
            if args.loss == "eagle":
                # head logits at i+1 predict t_{i+2} (causal LM; same alignment
                # as the fixed top1_acc metric) — drop the last predicted
                # position and label against toks[2:].
                loss, reg_l, cls_l = eagle_loss(
                    head, fused, tgt, toks[2:], model, device, args.ce_weight)
                reg_sum += reg_l.item()
                cls_sum += cls_l.item()
            else:
                loss = F.mse_loss(head(fused).float(), tgt)
            (loss / args.batch_blocks).backward()
            total_loss += loss.item()
            tb_step += 1
            if bi % args.batch_blocks == args.batch_blocks - 1 or bi == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if (bi + 1) % 100 == 0:
                print(f"  epoch {epoch} block {bi + 1}/{len(order)} "
                      f"loss {total_loss / (bi + 1):.5f}", flush=True)
                writer.add_scalar("train/loss_step", total_loss / (bi + 1), tb_step)
                if args.loss == "eagle":
                    writer.add_scalar("train/reg", reg_sum / (bi + 1), tb_step)
                    writer.add_scalar("train/cls", cls_sum / (bi + 1), tb_step)
        return total_loss / len(order)

    # --- run --------------------------------------------------------------
    epochs_before = len(report["epochs"])   # results log: only THIS run's rows
    for epoch in range(start_epoch + 1, start_epoch + 1 + args.epochs):
        t_ep = time.time()
        train_mse = train_epoch(epoch)
        metrics = evaluate()
        metrics.update(epoch=epoch, train_mse=train_mse)
        report["epochs"].append(metrics)
        print(f"epoch {epoch}: train_mse {train_mse:.5f} | val_mse {metrics['val_mse']:.5f} | "
              f"top1_acc {metrics['top1_acc']:.3f} | greedy_agreement "
              f"{metrics['greedy_agreement']:.3f} ({time.time() - t_ep:.0f}s)", flush=True)
        if best is None or metrics["greedy_agreement"] > best["greedy_agreement"]:
            best = dict(metrics)
            torch.save(head.fc.state_dict(), out_dir / "head_fc.pt")
            torch.save(head.layers.state_dict(), out_dir / "head_layers.pt")
            print(f"  -> saved best checkpoint (epoch {epoch})", flush=True)
        writer.add_scalar("train/mse", train_mse, epoch)
        writer.add_scalar("val/mse", metrics["val_mse"], epoch)
        writer.add_scalar("val/top1_acc", metrics["top1_acc"], epoch)
        writer.add_scalar("val/greedy_agreement", metrics["greedy_agreement"], epoch)
        # incremental report write — crash-safe (M3 lesson): the on-disk
        # report always reflects every completed epoch, so a mid-resume crash
        # never loses epoch history (checkpoint + report stay in lockstep).
        (out_dir / "train_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    # --- gate + report -----------------------------------------------------
    if best is None:
        raise SystemExit("no epochs completed — nothing to report")
    agree = best["greedy_agreement"]
    if agree >= 0.50:
        verdict = "PASS — clearly above the ~0.35 vanilla baseline; proceed to scale-up + engine integration"
    elif agree >= 0.42:
        verdict = "BORDERLINE — above baseline but thin; consider more epochs/data before scaling"
    else:
        verdict = "STOP — not clearly above the ~0.35 vanilla baseline; reconsider corpus/head/token budget (plan §4 gate)"
    report.update({
        "best": best,
        "gate": {"vanilla_greedy_acceptance_baseline": 0.35, "thresholds": "PASS>=0.50, BORDERLINE 0.42-0.50",
                 "verdict": verdict},
        "note": "greedy_agreement = P(head top1 == target top1) on the wikitext holdout — the "
                "acceptance-relevant stat for greedy speculative decoding. top1_acc = "
                "head top1 vs the ACTUAL token two ahead (t_{i+2}); target_top1_acc = "
                "the target's own top1 accuracy, for calibrating the holdout's "
                "predictability (off-by-one in top1_acc fixed Aug 10).",
    })
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_results_log(report, epochs_before)
    writer.close()
    print(f"\nbest: top1_acc {best['top1_acc']:.3f} | greedy_agreement {agree:.3f} (epoch {best['epoch']})")
    print(f"GATE: {verdict}")
    print(f"wrote {out_dir / 'head_fc.pt'}, {out_dir / 'head_layers.pt'}, train_report.json")
    print(f"tensorboard logs: {tb_dir} (watch: uv run tensorboard --logdir {tb_root})")


if __name__ == "__main__":
    main()
