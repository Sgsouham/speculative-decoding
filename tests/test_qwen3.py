"""Qwen3 loader genericity — isolated from the Qwen2.5 pair fixture so the
Qwen3 model is the only resident model (keeps peak VRAM ~1.5 GB; the GPU is
shared with the Windows host)."""
import pytest
import torch

from src.config import load_config
from src.models import load_model

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (run inside WSL2)"
)

PROMPT = "The capital of France is"


@requires_cuda
def test_qwen3_draft_loader():
    """A Qwen3 model from the switchable catalog loads + decodes (fp16/eval)."""
    cfg = load_config()
    qwen3_id = cfg["models"]["drafts"]["qwen3-0.6b"]
    model, tok = load_model(qwen3_id)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape[1] > ids.shape[1]
    del model
    torch.cuda.empty_cache()
