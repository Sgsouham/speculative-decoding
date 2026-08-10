"""train_head.py — M4 Phase 0: train the EAGLE-1-style draft head.

Consumes the feature cache produced by m4/collect_features.py
(data/m4/wikitext2/): fp16 second-to-top-layer features + token ids.

Head (arXiv 2401.15077 §3.2):
    FC(2h -> h)  +  one transformer decoder layer, deep-copied from the
    target's TOP decoder layer (the layer right after the cached feature
    layer). The target's embedding / top layer / norm / LM head are FROZEN
    and reused. Per position i:  [f_i ; embed(t_{i+1})] -> f̂_{i+1}, trained
    with MSE against the cached f_{i+1}. FC is identity-initialized on the
    feature half ([f_i ; e] -> f_i), a principled warm start.

Gate metric (plan §4 step 4a): GREEDY AGREEMENT = the fraction of positions
where the head's top-1 token equals the target's own top-1 token, measured
on a held-out tail of the cache. This is the acceptance-relevant statistic
for greedy speculative decoding — in greedy, a draft token is accepted iff
it is the target's argmax. Vanilla baseline: ~0.35 (M3, real text). If the
head is not clearly above that, STOP and reconsider (plan §4 gate).

Training precision: head params fp32 (small, ~50M — fp32 costs nothing and
avoids fp16 instability), cache features fp16 upcast per batch, MSE in fp32.
The engine casts the head to fp16 at deploy.

Outputs (--out, default data/m4/):
    head_fc.pt        FC state dict (fp32, best epoch)
    head_layer.pt     decoder-layer state dict (fp32, best epoch)
    train_report.json per-epoch val metrics + gate verdict

Usage (WSL2):
    HF_HOME=/mnt/d/projects/hf-cache uv run python m4/train_head.py --epochs 5
    HF_HOME=/mnt/d/projects/hf-cache uv run python m4/train_head.py --resume --epochs 5
    HF_HOME=/mnt/d/projects/hf-cache uv run python m4/train_head.py --self-test

--resume: continue from the BEST saved checkpoint (head_fc.pt / head_layer.pt)
and the epoch numbering in train_report.json, instead of re-training from the
identity init. Same LR unless --lr is passed; the best-checkpoint tracking
only overwrites the saved weights when greedy_agreement improves, so a resume
that doesn't beat the previous best leaves the checkpoint untouched.

Live stats: per-step train loss + per-epoch val metrics are written to
TensorBoard under --logdir (default data/m4/runs/<run-tag>/, flush every 5s).
A resume reuses the previous run's tag so the curves stay continuous. Watch:

    uv run tensorboard --logdir data/m4/runs --port 6006

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

from src.config import load_config, resolve_model_id  # noqa: E402
from src.models import load_model  # noqa: E402
from m4.collect_features import _chunk_idx  # noqa: E402


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
        raise SystemExit(f"no chunks in {cache_dir} — run m4/collect_features.py first")
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
    """FC(2h -> h) + one decoder layer (copy of the target's TOP layer).

    The target's embedding / top layer / norm / LM head are frozen and used
    externally by the training loop. FC is identity-initialized on the
    feature half so the head starts as the identity feature map.
    """

    def __init__(self, hidden: int, decoder_layer: nn.Module, rotary_emb: nn.Module):
        super().__init__()
        self.fc = nn.Linear(2 * hidden, hidden)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:, :hidden] = torch.eye(hidden)
            self.fc.bias.zero_()
        self.layer = copy.deepcopy(decoder_layer).float()  # fp32 training
        self.rotary_emb = rotary_emb
        self.hidden = hidden

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """fused [B, S, 2h] -> predicted next features [B, S, h] (f̂_{i+1})."""
        x = self.fc(fused)
        b, s, _ = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary_emb(x, pos)
        mask = causal_mask(s, x.device, x.dtype)
        out = self.layer(
            x,
            attention_mask=mask,
            position_ids=pos,
            position_embeddings=(cos, sin),
            use_cache=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out


@torch.no_grad()
def run_top_layer(model, x: torch.Tensor, device) -> torch.Tensor:
    """Map cached features (layer -2 space, [B, S, h]) through the frozen
    target TOP decoder layer + final norm — the head of the LM path. x must
    be the model's dtype (fp16 in production, matching the engine)."""
    b, s, _ = x.shape
    pos = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
    cos, sin = model.model.rotary_emb(x, pos)
    mask = causal_mask(s, device, x.dtype)
    out = model.model.layers[-1](
        x, attention_mask=mask, position_ids=pos,
        position_embeddings=(cos, sin), use_cache=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return model.model.norm(out)


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
    head = EagleDraftHead(h, model.model.layers[-1], model.model.rotary_emb)
    feats = torch.randn(48, h)
    toks = torch.randint(0, cfg.vocab_size, (48,))
    f, t_next, tgt_f, tgt_t = make_pairs(feats, toks)
    fused = torch.cat([f, model.model.embed_tokens(t_next)], dim=-1).unsqueeze(0)
    pred = head(fused)
    assert pred.shape == (1, f.shape[0], h), pred.shape
    loss = F.mse_loss(pred, tgt_f.unsqueeze(0))
    loss.backward()
    assert head.fc.weight.grad is not None
    assert head.layer.self_attn.q_proj.weight.grad is not None
    top = run_top_layer(model, tgt_f.unsqueeze(0), torch.device("cpu"))
    logits = model.lm_head(top)
    assert logits.shape == (1, tgt_f.shape[0], cfg.vocab_size), logits.shape
    agree = float((logits.argmax(-1)[0] == tgt_t).float().mean())

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

    print(f"self-test OK: pred {tuple(pred.shape)} loss {loss.item():.4f} | "
          f"target-logits {tuple(logits.shape)} | top1-vs-actual {agree:.2f} | "
          f"head params {sum(p.numel() for p in head.parameters()):,}")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="M4 Phase 0: train the EAGLE-1 draft head (FC + 1 decoder layer)")
    ap.add_argument("--cache", default="data/m4/wikitext2", help="feature cache dir (from collect_features.py)")
    ap.add_argument("--out", default="data/m4", help="output dir for head weights + report")
    ap.add_argument("--model", default="qwen2.5-3b", help="target alias in config/default.yaml")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate")
    ap.add_argument("--batch-blocks", type=int, default=4,
                    help="gradient accumulation over N blocks (effective batch ~N*1022 pairs)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out tail of the cache")
    ap.add_argument("--logdir", default="data/m4/runs",
                    help="TensorBoard log root (each run gets its own subdir; resume reuses it)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the best saved checkpoint + report (epoch numbering picks up)")
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
    head = EagleDraftHead(hidden, model.model.layers[-1], model.model.rotary_emb)
    head.to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"head: FC(2*{hidden}->{hidden}) + decoder layer — {n_params:,} trainable params", flush=True)

    # --- resume bookkeeping ------------------------------------------------
    report = {"epochs": [], "config": vars(args), "head_params": n_params}
    best = None
    start_epoch = 0
    if args.resume:
        fc_path, layer_path = out_dir / "head_fc.pt", out_dir / "head_layer.pt"
        if not (fc_path.exists() and layer_path.exists()):
            raise SystemExit(f"--resume: no checkpoint in {out_dir} — run without --resume first")
        head.fc.load_state_dict(torch.load(fc_path, weights_only=True))
        head.layer.load_state_dict(torch.load(layer_path, weights_only=True))
        print(f"resumed weights from {fc_path.name} + {layer_path.name}", flush=True)
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
        total_loss = 0.0
        opt.zero_grad()
        for bi, ci in enumerate(order):
            feats, toks = train_chunks[ci]
            f, t_next, tgt_f, _ = make_pairs(feats, toks)
            with torch.no_grad():  # frozen target — no gradients through the embedding
                e = model.model.embed_tokens(t_next.to(device))
            fused = torch.cat([f.float().to(device), e.float()], dim=-1).unsqueeze(0)
            tgt = tgt_f.float().unsqueeze(0).to(device)
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
        return total_loss / len(order)

    # --- run --------------------------------------------------------------
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
            torch.save(head.layer.state_dict(), out_dir / "head_layer.pt")
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
    writer.close()
    print(f"\nbest: top1_acc {best['top1_acc']:.3f} | greedy_agreement {agree:.3f} (epoch {best['epoch']})")
    print(f"GATE: {verdict}")
    print(f"wrote {out_dir / 'head_fc.pt'}, {out_dir / 'head_layer.pt'}, train_report.json")
    print(f"tensorboard logs: {tb_dir} (watch: uv run tensorboard --logdir {tb_root})")


if __name__ == "__main__":
    main()
