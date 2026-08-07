"""Repo 02 — speculative-decoding (vanilla, from scratch)."""

import os

# Pin transformers' hub-kernels behavior OFF. transformers 5.x reads
# USE_HUB_KERNELS at import time (default YES: may fetch compiled kernels from
# the HF hub at runtime). Our models (Qwen2.5/Qwen3) don't consume hub kernels,
# so this is behavior-neutral — but it keeps all runs network-independent and
# reproducible. setdefault: an explicitly-set env var still wins.
os.environ.setdefault("USE_HUB_KERNELS", "0")

__version__ = "0.1.0"
