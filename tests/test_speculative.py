"""Core speculative decoding loop tests. GPU required (run inside WSL2)."""
import pytest
import torch

from src.config import load_config
from src.models import load_pair
from src.speculative import _residual_distribution, autoregressive_decode, speculative_decode

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (run inside WSL2)"
)

PROMPTS = [
    "The capital of France is",
    "In a distant future, humanity",
    "def compute_fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "The theory of evolution by natural selection",
]


@pytest.fixture(scope="module")
def pair():
    cfg = load_config()
    draft, target, tok = load_pair(cfg)
    yield draft, target, tok
    del draft, target
    torch.cuda.empty_cache()


def _ids(tok, prompt, device):
    return tok(prompt, return_tensors="pt").input_ids.to(device)


@requires_cuda
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_target_only(pair, prompt):
    """§6.1 — the core-loop exit criterion: speculative == target-only greedy, token-identical."""
    draft, target, tok = pair
    ids = _ids(tok, prompt, target.device)
    sd, stats = speculative_decode(draft, target, ids, max_new_tokens=48, draft_length=4)
    ar, _ = autoregressive_decode(target, ids, max_new_tokens=48)
    assert torch.equal(sd, ar)
    assert stats["proposed"] >= stats["accepted"] > 0


@requires_cuda
@pytest.mark.parametrize("prompt", PROMPTS[:2])
def test_greedy_matches_hf_generate(pair, prompt):
    """hand-rolled SD == plain HF greedy generate."""
    draft, target, tok = pair
    ids = _ids(tok, prompt, target.device)
    sd, _ = speculative_decode(draft, target, ids, max_new_tokens=48, draft_length=4)
    ref = target.generate(ids, max_new_tokens=48)
    assert torch.equal(sd, ref)


@requires_cuda
@pytest.mark.parametrize("prompt", PROMPTS[:2])
def test_greedy_matches_hf_assisted(pair, prompt):
    """§6.3 three-way agreement: hand-rolled == HF-speculative == HF-greedy."""
    draft, target, tok = pair
    ids = _ids(tok, prompt, target.device)
    sd, _ = speculative_decode(draft, target, ids, max_new_tokens=48, draft_length=4)
    ref = target.generate(ids, max_new_tokens=48, assistant_model=draft.model)
    assert torch.equal(sd, ref)


@requires_cuda
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_greedy_all_draft_lengths(pair, k):
    draft, target, tok = pair
    ids = _ids(tok, PROMPTS[0], target.device)
    sd, _ = speculative_decode(draft, target, ids, max_new_tokens=24, draft_length=k)
    ar, _ = autoregressive_decode(target, ids, max_new_tokens=24)
    assert torch.equal(sd, ar)


@requires_cuda
def test_sampled_runs_and_is_reproducible(pair):
    draft, target, tok = pair
    ids = _ids(tok, PROMPTS[0], target.device)
    a, _ = speculative_decode(draft, target, ids, max_new_tokens=32, draft_length=4, temperature=0.7, seed=42)
    b, _ = speculative_decode(draft, target, ids, max_new_tokens=32, draft_length=4, temperature=0.7, seed=42)
    assert torch.equal(a, b)  # seeded -> reproducible
    assert a.shape[1] == ids.shape[1] + 32


@requires_cuda
def test_residual_distribution_math():
    """§6.2 — the correctness-critical max(0, p_t - p_d)_+ line, unit-tested."""
    p_d = torch.tensor([0.5, 0.3, 0.2])
    p_t = torch.tensor([0.4, 0.35, 0.25])
    q = _residual_distribution(p_t, p_d)
    expected = torch.tensor([0.0, 0.5, 0.5])  # (0.05, 0.05) normalized
    torch.testing.assert_close(q, expected, atol=1e-6, rtol=1e-6)
    assert torch.isclose(q.sum(), torch.tensor(1.0))


@requires_cuda
def test_cache_crop_continuity(pair):
    """Rollback semantics: after cropping the cache mid-decode and continuing,
    output must match a fresh decode (the SD rejection path's cache handling)."""
    draft, target, tok = pair
    ids = _ids(tok, PROMPTS[0], target.device)
    target.reset()
    target.forward(ids)
    seq = []
    for _ in range(4):
        t = target.pending_logits.argmax(-1)
        target.forward(t)
        seq.append(t)
    # roll back to prompt + first 2 tokens, re-append the 3rd, continue 3 more
    target.crop_cache(ids.shape[1] + 2)
    target.forward(seq[2])
    got = []
    for _ in range(3):
        t = target.pending_logits.argmax(-1)
        target.forward(t)
        got.append(t)
    # crop kept the right prefix: the next prediction is exactly seq[3]
    assert torch.equal(got[0], seq[3])
    # and the continuation matches a fresh HF greedy decode from the full prefix
    ref = target.generate(torch.cat([ids, *seq], dim=1), max_new_tokens=2)
    assert torch.equal(torch.cat(got[1:], dim=1), ref[:, -2:])
