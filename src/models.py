"""Model plumbing for speculative decoding.

- fp16/eval loaders via AutoModelForCausalLM (works for Qwen2.5 + Qwen3).
- Shared tokenizer across draft/target (same 151,669-token Qwen vocab).
- ModelHandle wraps a model with its OWN KV-cache state + reset() semantics,
  per plan §4: draft and target each keep their own cache.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config, resolve_model_id

DEFAULT_DTYPE = torch.float16


class ModelHandle:
    """A model plus its own KV-cache state.

    forward() caches past_key_values on the handle and records pending_logits
    (the last row of the output — the prediction of the next token); reset()
    clears both. The cached path is what the speculative loop uses: draft runs
    k autoregressive steps, the target runs one parallel verification pass over
    the draft block.

    Contract: forward() and generate() are per-sequence paths — call reset()
    between sequences. generate() resets automatically (HF manages its own
    cache internally); forward() callers are responsible for their own reset.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reset()

    # -- cache semantics -------------------------------------------------
    def reset(self) -> None:
        """Clear this model's KV cache + pending logits (per-model semantics)."""
        self.past_key_values = None
        self.pending_logits = None
        self.last_hidden_states = None

    def crop_cache(self, seq_len: int) -> None:
        """Truncate this handle's KV cache to the first seq_len positions.

        Used by the speculative loop to roll the caches back to the accepted
        prefix after draft tokens are rejected. Note: transformers 5.x `crop`
        mutates the cache in place and returns None; 4.x returns a new cache —
        handle both.
        """
        if self.past_key_values is None:
            return
        result = self.past_key_values.crop(seq_len)
        if result is not None:
            self.past_key_values = result

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    # -- forward ---------------------------------------------------------
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, use_cache: bool = True,
                capture_hidden: bool = False):
        """Forward with this handle's cached past_key_values; updates the cache.

        pending_logits is updated to the last row of the output (the prediction
        of the token following the current sequence). capture_hidden=True also
        requests hidden states and stashes them on last_hidden_states        (None otherwise) — the EAGLE engine's seed-feature seam (plan §3). Default
        False keeps the decode/benchmark paths byte-identical.
        """
        out = self.model(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=use_cache,
            output_hidden_states=capture_hidden,
        )
        self.past_key_values = out.past_key_values
        if out.logits is not None:
            self.pending_logits = out.logits[:, -1:, :]
        self.last_hidden_states = out.hidden_states if capture_hidden else None
        return out

    # -- EAGLE feature capture (draft-head data pipeline) -------------------------
    @torch.no_grad()
    def features(self, input_ids: torch.Tensor, layer_index: int = -2) -> torch.Tensor:
        """Hidden states for ALL positions of `input_ids` (a full-prefill block).

        Returns `hidden_states[layer_index]` — default -2, the second-to-top
        layer (the layer before the LM head): the "feature" the EAGLE draft
        head learns to predict (arXiv 2401.15077). Pure prefill with
        use_cache=False — deliberately does NOT touch this handle's KV cache
        or pending_logits, so feature collection can never disturb the
        decode/benchmark paths.
        """
        out = self.model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        return out.hidden_states[layer_index]

    # -- standalone decode (baseline / oracle path) -----------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        assistant_model=None,
        **gen_kwargs,
    ):
        """HF generate — FORCED greedy.

        Qwen instruct models bake sampling defaults into generation_config
        (do_sample=True, temperature 0.7, top_k 20, top_p 0.8, repetition
        penalty 1.1); the oracle/baseline must be pure greedy over raw logits
        to match the hand-rolled engine. Pass assistant_model=<draft .model>
        for the §6.3 oracle.
        """
        self.reset()  # generate manages its own cache; never continue a stale one
        greedy = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
        }
        greedy.update(gen_kwargs)  # explicit caller kwargs still win
        if assistant_model is not None:
            greedy["assistant_model"] = assistant_model
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **greedy,
        )


def _normalize_generation(model) -> None:
    """Force greedy decoding defaults on the model's generation_config.

    Qwen instruct repos bake sampling defaults (do_sample=True, temperature
    0.7, top_k 20, top_p 0.8, repetition_penalty 1.1) — wrong for equivalence
    testing/benchmarking, where we decode over raw logits.
    """
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = 1.0
    gc.top_k = None
    gc.top_p = None
    gc.repetition_penalty = None


def load_model(repo_id: str, dtype=DEFAULT_DTYPE, device: str = "cuda"):
    """Load one causal LM in fp16/eval. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, dtype=dtype, low_cpu_mem_usage=True  # `dtype` not torch_dtype (5.14.1 deprecation)
    )
    _normalize_generation(model)
    model.to(device).eval()
    return model, tokenizer


def load_pair(cfg: dict | None = None, dtype=DEFAULT_DTYPE, device: str = "cuda"):
    """Load the active draft + target with a SHARED tokenizer.

    The Qwen2.5/Qwen3 families share one vocab, so a single tokenizer serves
    both. Returns (draft_handle, target_handle, tokenizer).
    """
    cfg = cfg or load_config()
    target_id = resolve_model_id(cfg, "target")[1]
    draft_id = resolve_model_id(cfg, "draft")[1]

    tokenizer = AutoTokenizer.from_pretrained(target_id)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_id, dtype=dtype, low_cpu_mem_usage=True  # `dtype` not torch_dtype (5.14.1 deprecation)
    )
    _normalize_generation(draft_model)
    draft_model.to(device).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        target_id, dtype=dtype, low_cpu_mem_usage=True  # `dtype` not torch_dtype (5.14.1 deprecation)
    )
    _normalize_generation(target_model)
    target_model.to(device).eval()

    return ModelHandle(draft_model, tokenizer), ModelHandle(target_model, tokenizer), tokenizer
