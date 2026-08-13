"""Model plumbing tests. GPU required (run inside WSL2 via `uv run pytest`)."""
import pytest
import torch
from transformers import AutoTokenizer

from src.config import load_config, resolve_model_id
from src.models import load_pair

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (run inside WSL2)"
)

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def pair():
    cfg = load_config()
    draft, target, tok = load_pair(cfg)
    yield draft, target, tok
    del draft, target
    torch.cuda.empty_cache()


@requires_cuda
def test_pair_loads_fp16_eval(pair):
    draft, target, _ = pair
    for handle in (draft, target):
        assert handle.dtype == torch.float16
        assert not handle.model.training
        assert str(handle.device).startswith("cuda")


@requires_cuda
def test_shared_tokenizer_same_vocab(pair):
    _, _, tok = pair
    cfg = load_config()
    # Each model's own tokenizer must produce identical ids (shared vocab).
    for kind, alias in (("draft", cfg["model"]["draft"]), ("target", cfg["model"]["target"])):
        repo_id = resolve_model_id(cfg, kind, alias)[1]
        own = AutoTokenizer.from_pretrained(repo_id)
        assert own.vocab_size == tok.vocab_size
        assert own(PROMPT).input_ids == tok(PROMPT).input_ids


@requires_cuda
def test_standalone_greedy_decode(pair):
    draft, target, tok = pair
    for handle in (draft, target):
        handle.reset()
        ids = tok(PROMPT, return_tensors="pt").input_ids.to(handle.device)
        out = handle.generate(ids, max_new_tokens=16)
        assert out.shape[1] > ids.shape[1]  # produced new tokens
        text = tok.decode(out[0], skip_special_tokens=True)
        assert len(text.strip()) > 0


@requires_cuda
def test_kv_cache_equivalence(pair):
    """Cached step-by-step forward reproduces one-shot forward (cache correctness).

    Token identity (argmax) is what decoding depends on; logits themselves are
    fp16-honest — stepwise vs one-shot kernel ordering drifts by ~1 ULP (§8).
    """
    draft, target, tok = pair
    for handle in (draft, target):
        handle.reset()
        ids = tok(PROMPT, return_tensors="pt").input_ids.to(handle.device)
        # one-shot reference
        ref = handle.model(ids).logits  # [1, seq, vocab]
        # cached walk, one token at a time
        handle.reset()
        got = []
        for i in range(ids.shape[1]):
            out = handle.forward(ids[:, i : i + 1])
            got.append(out.logits[:, -1, :])
        got = torch.stack(got, dim=1)
        # real bugs flip argmax; fp16 noise of ~1 ULP does not
        assert torch.equal(got.argmax(-1), ref.argmax(-1))
        # fp16-honest tolerance: ~4 ULP at max logit magnitude (ULP(64)=0.0625);
        # a broken cache diverges by O(1)+ logits or flips argmax, well above this
        max_ulp = torch.finfo(torch.float16).eps * ref.abs().max()
        torch.testing.assert_close(got, ref, atol=4 * max_ulp.item(), rtol=0.1)


@requires_cuda
def test_assistant_generate_matches_greedy(pair):
    """§6.3 oracle: generate(assistant_model=draft) works on transformers 5.14.1
    and reproduces target-only greedy output token-identically."""
    draft, target, tok = pair
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(target.device)
    ref = target.generate(ids, max_new_tokens=32)
    got = target.generate(ids, max_new_tokens=32, assistant_model=draft.model)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), "assisted greedy diverged from target-only greedy"

