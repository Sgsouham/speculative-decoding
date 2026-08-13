"""collect_eagle_features.py — the draft-head feature-cache pipeline.

Runs the target model (Qwen2.5-3B) over the WikiText-2 raw train corpus and
caches, for every token position, the **second-to-top-layer hidden state** —
the "feature" the EAGLE-style draft head learns to predict (arXiv 2401.15077).
The corpus is *existing text*, so this is prefill-bound (fast), not generation.

Cache layout (data/draft-head/wikitext2/):
    manifest.json                 — model id, dims, dtype, totals, chunk bookkeeping
    chunk_000000.features.npy     — fp16 [n, hidden]   second-to-top-layer features
    chunk_000000.tokens.npy       — int64 [n]          token ids (same positions)

Pair construction happens at TRAINING time, not here: example i is
(features[i], tokens[i+1]) -> features[i+1]. The LAST position of every block
is dropped here so every cached pair is self-contained within one block's
causal context (cost: 1 token per 1024, ~0.1%).

Design notes (same discipline as the benchmark harness):
- Blocks are exactly max_seq_len tokens → no padding, no masking.
- Chunked writes + resume: finished chunks are never rewritten; a restart
  skips existing chunk indices and continues. Each chunk is written to a tmp
  path and renamed, so a killed run never leaves a corrupt chunk.
- --smoke N processes only the first N blocks and prints a validation summary
  (shapes, dtype, finiteness) — the pipeline self-test before a real run.
- --max-tokens N caps the cache (the Phase-0 gate runs 500K; the full train
  split is ~2.1M). A resumed run with the same --max-tokens stops at the same
  total (already-cached tokens count against the budget).

Usage (WSL2):
    HF_HOME=/mnt/d/projects/hf-cache uv run python src/collect_eagle_features.py --smoke 3
    HF_HOME=/mnt/d/projects/hf-cache uv run python src/collect_eagle_features.py --max-tokens 500000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import chunk_idx as _chunk_idx  # noqa: E402
from src.config import load_config, resolve_model_id  # noqa: E402
from src.models import ModelHandle, load_model  # noqa: E402

CORPUS_REPO = "Salesforce/wikitext"
CORPUS_NAME = "wikitext-2-raw-v1"          # raw = keeps wiki markup; ~2.1M tokens
CORPUS_FILENAME = "train-00000-of-00001.parquet"


def get_corpus_text(cache_dir: Path) -> tuple[str, dict]:
    """Fetch the WikiText-2 raw train split once, cache it as plain text.

    The old Salesforce S3 zip is dead; HF serves the corpus as one parquet
    file. Returns (text, stats) where stats has row/char counts. The plain
    text cache makes re-runs and resume network-free.
    """
    txt_path = cache_dir / "train.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
        return text, {"rows": text.count("\n") + 1, "chars": len(text), "source": "cached txt"}

    parquet_path = cache_dir / f"{CORPUS_NAME}-train.parquet"
    if not parquet_path.exists():
        from huggingface_hub import hf_hub_download

        print(f"downloading {CORPUS_REPO} {CORPUS_FILENAME} ...", flush=True)
        downloaded = hf_hub_download(
            repo_id=CORPUS_REPO,
            filename=f"{CORPUS_NAME}/{CORPUS_FILENAME}",
            repo_type="dataset",  # Salesforce/wikitext is a DATASET repo (401 without this)
            cache_dir=str(cache_dir),
        )
        shutil.copyfile(downloaded, parquet_path)
        print(f"parquet -> {parquet_path}", flush=True)

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    texts = table.column("text").to_pylist()
    text = "\n".join(texts)
    txt_path.write_text(text, encoding="utf-8")
    return text, {"rows": len(texts), "chars": len(text), "source": "parquet"}


def load_target_handle(model_alias: str) -> tuple[ModelHandle, object, int]:
    """Load the target model; returns (handle, tokenizer, hidden_size)."""
    cfg = load_config()
    model_id = resolve_model_id(cfg, "target", model_alias)[1]
    model, tokenizer = load_model(model_id)
    handle = ModelHandle(model, tokenizer)
    hidden_size = model.config.hidden_size
    return handle, tokenizer, hidden_size


def main() -> None:
    dh = load_config().get("draft_head", {})  # config/default.yaml → CLI defaults
    ap = argparse.ArgumentParser(description="draft-head: EAGLE-style feature cache (WikiText-2 -> hidden states)")
    ap.add_argument("--model", default=dh.get("model", "qwen2.5-3b"),
                    help="target alias in config/default.yaml (default: qwen2.5-3b)")
    ap.add_argument("--out", default=dh.get("cache", "data/draft-head/wikitext2"),
                    help="cache dir (default: data/draft-head/wikitext2)")
    ap.add_argument("--max-seq-len", type=int, default=1024, help="tokens per block (no padding needed)")
    ap.add_argument("--min-seq-len", type=int, default=64, help="drop corpus tail shorter than this")
    ap.add_argument("--chunk-tokens", type=int, default=100_000, help="approx tokens per chunk file")
    ap.add_argument("--max-tokens", type=int, default=None, help="stop after caching ~N tokens (Phase-0 gate: 500000)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N", help="process only the first N blocks, validate, exit")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # smoke runs write to a scratch subdir so they never pollute the real cache
    chunk_dir = out_dir / "smoke" if args.smoke else out_dir
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # --- corpus ---------------------------------------------------------
    text, corpus_stats = get_corpus_text(out_dir)
    print(f"corpus: {corpus_stats['rows']:,} rows, {corpus_stats['chars']:,} chars ({corpus_stats['source']})",
          flush=True)

    # --- model ----------------------------------------------------------
    print(f"loading target {args.model} ...", flush=True)
    t0 = time.time()
    handle, tokenizer, hidden_size = load_target_handle(args.model)
    print(f"model loaded in {time.time() - t0:.0f}s (hidden {hidden_size}, {handle.dtype})", flush=True)
    device = handle.device

    # --- tokenize + block ------------------------------------------------
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    print(f"tokenized: {ids.numel():,} tokens", flush=True)
    n_blocks = ids.numel() // args.max_seq_len
    blocks = ids[: n_blocks * args.max_seq_len].view(n_blocks, args.max_seq_len)
    if args.smoke:
        blocks = blocks[: args.smoke]
        if blocks.shape[0] == 0:
            raise SystemExit("--smoke: corpus has no full blocks — nothing to validate")

    # --- resume bookkeeping ----------------------------------------------
    done_chunks = sorted(_chunk_idx(p) for p in chunk_dir.glob("chunk_*.features.pt"))
    cached_tokens = sum(
        int(torch.load(p, weights_only=True).numel()) for p in chunk_dir.glob("chunk_*.tokens.pt")
    )
    if done_chunks:
        print(f"resume: {len(done_chunks)} chunk(s), {cached_tokens:,} tokens already cached", flush=True)
    max_tokens = args.max_tokens
    if max_tokens is not None and cached_tokens >= max_tokens:
        print(f"already at/over --max-tokens ({cached_tokens:,} >= {max_tokens:,}) — nothing to do", flush=True)
        return
    if max_tokens is not None:
        max_tokens -= cached_tokens

    next_chunk_idx = done_chunks[-1] + 1 if done_chunks else 0
    feature_buf: list[torch.Tensor] = []
    token_buf: list[torch.Tensor] = []
    buf_tokens = 0
    total = 0
    t_start = time.time()

    def flush_chunk() -> None:
        nonlocal feature_buf, token_buf, buf_tokens, next_chunk_idx
        feats = torch.cat(feature_buf, dim=0)
        toks = torch.cat(token_buf, dim=0)
        feats_path = chunk_dir / f"chunk_{next_chunk_idx:06d}.features.pt"
        toks_path = chunk_dir / f"chunk_{next_chunk_idx:06d}.tokens.pt"
        tmp_f, tmp_t = feats_path.with_suffix(".pt.tmp"), toks_path.with_suffix(".pt.tmp")
        torch.save(feats, tmp_f)   # torch format (fp16 features / int64 tokens)
        torch.save(toks, tmp_t)
        tmp_f.replace(feats_path)
        tmp_t.replace(toks_path)
        feature_buf, token_buf, buf_tokens = [], [], 0
        next_chunk_idx += 1

    def write_manifest() -> None:
        payload = {
            "pipeline": "src/collect_eagle_features.py",
            "corpus": {"repo": CORPUS_REPO, "config": CORPUS_NAME, **corpus_stats},
            "model": {"alias": args.model, "hidden_size": hidden_size},
            "feature_layer": -2, "dtype": str(handle.dtype),
            "max_seq_len": args.max_seq_len, "min_seq_len": args.min_seq_len,
            "chunk_tokens": args.chunk_tokens,
            "total_tokens": total + cached_tokens,
            "n_chunks": len(list(chunk_dir.glob("chunk_*.features.pt"))),
            "pairs_dropped_per_block": 1,  # last position of each block (no next token)
        }
        (chunk_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- main loop --------------------------------------------------------
    print(f"caching {blocks.shape[0]} block(s) x {args.max_seq_len} tokens ...", flush=True)
    for bi, block in enumerate(blocks):
        # block is 1-D [seq] — the model needs a batch dim ([1, seq]); a 1-D
        # input makes transformers treat the embedding dim as the sequence
        # length (position_ids = arange(embedding_dim)) and the attention
        # reshape silently mis-splits, blowing up in rotary (Aug 10 bug).
        feats = handle.features(block.unsqueeze(0).to(device))  # [1, seq, hidden]
        feats = feats[0, :-1].float().cpu()             # [seq-1, hidden]; drop last pos
        toks = block[:-1].clone().cpu()                 # same positions
        feature_buf.append(feats.half())                # fp16 on disk (2 bytes/val)
        token_buf.append(toks)
        n = feats.shape[0]
        buf_tokens += n
        total += n
        if bi % 20 == 0 or buf_tokens >= args.chunk_tokens:
            if buf_tokens >= args.chunk_tokens or bi == blocks.shape[0] - 1:
                flush_chunk()
                write_manifest()
            rate = total / (time.time() - t_start)
            remaining = ""
            if max_tokens:
                remaining = f" | {total + cached_tokens:,}/{args.max_tokens:,} ({rate:,.0f} tok/s)"
            print(f"  block {bi + 1}/{blocks.shape[0]} | cached {total + cached_tokens:,} tokens{remaining}",
                  flush=True)
        if max_tokens and total >= max_tokens:
            print(f"--max-tokens reached ({total + cached_tokens:,} cached)", flush=True)
            break

    if buf_tokens:
        flush_chunk()
        write_manifest()

    # --- smoke validation -------------------------------------------------
    if args.smoke:
        chunks = sorted(chunk_dir.glob("chunk_*.features.pt"))
        f = torch.load(chunks[0], weights_only=True)
        t_path = chunks[0].with_name(chunks[0].name.replace("features", "tokens"))
        t = torch.load(t_path, weights_only=True)
        finite = bool(torch.isfinite(f).all())
        stats = (float(f.mean()), float(f.std()), float(f.abs().max()))
        print(f"\nsmoke OK: {chunks[0].name} features {tuple(f.shape)} {f.dtype} | "
              f"tokens {tuple(t.shape)} {t.dtype} | finite={finite} | "
              f"mean={stats[0]:.4f} std={stats[1]:.4f} absmax={stats[2]:.2f}")
        print(f"  (first 5 tokens: {t[:5].tolist()})")
    else:
        write_manifest()
        print(f"\ndone: {total + cached_tokens:,} tokens cached in {out_dir} "
              f"({time.time() - t_start:.0f}s, {total / (time.time() - t_start):,.0f} tok/s)")


if __name__ == "__main__":
    main()
