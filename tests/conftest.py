"""Test-suite env pinning — runs before any test imports transformers.

transformers 5.x reads USE_HUB_KERNELS (runtime hub-kernel fetch, default YES)
at import time; pin it OFF so tests never depend on the network or on compiled
kernels fetched from the HF hub. Mirrors src/__init__.py for test modules that
import transformers directly. setdefault: an explicit env var still wins.
"""
import os

os.environ.setdefault("USE_HUB_KERNELS", "0")
