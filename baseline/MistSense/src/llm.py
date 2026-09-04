"""LLM loader for MistSense Stage 2/3 (§3.2, §3.3, §4.6).

[DEVIATION] Paper §4.6 uses Llama2-7B (gated, requires HF auth, ~13GB).
This repro substitutes OPT-2.7b extracted from the already-cached
`Salesforce/blip2-opt-2.7b`. OPT-2.7b shares BLIP-2 alignment with the
paper's BLIP-2 Q-Former and avoids gated downloads.

  - paper Llama2-7B: d_hidden = 4096
  - our OPT-2.7b:    d_hidden = 2560

`llm_hidden` in config must match (we override to 2560 in egoops.yaml).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def build_llm(
    pretrained: str = "Salesforce/blip2-opt-2.7b",
    from_blip2: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[nn.Module, "AutoTokenizer", int]:
    """Load OPT-2.7b language model.

    If `from_blip2=True`, extract `.language_model` from `Blip2ForConditionalGeneration`
    (avoids re-downloading OPT-2.7b weights). Otherwise load `facebook/opt-2.7b` directly.

    Returns (llm, tokenizer, hidden_size).
    """
    from transformers import AutoTokenizer

    if from_blip2:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        blip2 = Blip2ForConditionalGeneration.from_pretrained(pretrained, torch_dtype=dtype)
        llm = blip2.language_model
        proc = Blip2Processor.from_pretrained(pretrained)
        tokenizer = proc.tokenizer
        # Free blip2 wrapper; we only keep llm
        blip2.language_model = nn.Identity()
        del blip2
    else:
        from transformers import AutoModelForCausalLM

        llm = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # OPT tokenizer has no pad token by default; reuse eos for batched padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.config.pad_token_id = tokenizer.pad_token_id

    llm.to(device)
    return llm, tokenizer, int(llm.config.hidden_size)


def lora_wrap(llm: nn.Module, r: int, alpha: int, dropout: float, target_modules: tuple[str, ...]) -> nn.Module:
    """§3.3 Stage 3 — wrap LLM with PEFT LoRA.

    Returns the wrapped model (a `PeftModel`); existing references to `llm`
    weights remain valid (LoRA adapters are inserted in-place around them).
    """
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(llm, cfg)


def build_llm_inputs(
    prefix_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embed_layer: nn.Module,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (inputs_embeds, attention_mask, labels) for causal LM forward
    with a visual soft-prefix (§3.2).

      prefix_embeds:  (B, t_q, d_llm)  — output of `model.project_to_llm`
      input_ids:      (B, L)           — tokenized explanation
      attention_mask: (B, L)           — 1 = real token, 0 = pad
      embed_layer:    nn.Embedding from `llm.get_input_embeddings()`

    Returns full-length tensors over (t_q + L) positions:
      inputs_embeds:  (B, t_q+L, d_llm)
      attention_mask: (B, t_q+L)
      labels:         (B, t_q+L) — prefix positions = ignore_index;
                                   pad positions = ignore_index.
    """
    B, t_q, d = prefix_embeds.shape
    tok_embeds = embed_layer(input_ids).to(prefix_embeds.dtype)        # (B, L, d_llm)
    inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)      # (B, t_q+L, d_llm)
    prefix_mask = torch.ones(B, t_q, dtype=attention_mask.dtype, device=attention_mask.device)
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    prefix_labels = torch.full((B, t_q), ignore_index, dtype=input_ids.dtype, device=input_ids.device)
    masked_ids = input_ids.masked_fill(attention_mask == 0, ignore_index)
    labels = torch.cat([prefix_labels, masked_ids], dim=1)
    return inputs_embeds, full_mask, labels


@torch.no_grad()
def generate_explanation(
    llm: nn.Module,
    tokenizer,
    prefix_embeds: torch.Tensor,
    max_new_tokens: int = 32,
    bos_token_id: Optional[int] = None,
) -> list[str]:
    """§3.2 inference — given a visual soft-prefix, generate explanation text.

    Adds a single BOS embedding after the prefix so HF `generate` has a token
    position to extend from. Returns decoded strings (one per batch item).
    """
    B = prefix_embeds.size(0)
    bos = bos_token_id if bos_token_id is not None else tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    bos_ids = torch.full((B, 1), bos, dtype=torch.long, device=prefix_embeds.device)
    bos_embed = llm.get_input_embeddings()(bos_ids).to(prefix_embeds.dtype)
    inputs_embeds = torch.cat([prefix_embeds, bos_embed], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)
    out = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)
