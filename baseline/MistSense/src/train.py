"""MistSense — three-stage training loop (§3.3).

Stage 1: weighted BCE for mistake detection.
  - Freeze: ViT-L (visual encoder), base BLIP-2 Q-Former, LLM, llm_proj
  - Train:  Video Q-Former, hand encoder, Hand Q-Former, mistake head
Stage 2: CE on explanation tokens.
  - Freeze everything (Stage-1 modules + LLM).
  - Train: llm_proj only.
Stage 3: LoRA-finetune the LLM with CE.
  - Freeze llm_proj.
  - Train: LoRA adapters on the LLM.

§4.6 hyperparameters: AdamW, lr=3e-5 (main) / 2.5e-2 (hand encoder),
wd=5e-3 (main) / 0 (hand encoder), cosine schedule, linear warmup
from 1e-6, 12 epochs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .data import MistakeClipDataset
from .llm import build_llm, build_llm_inputs, lora_wrap
from .loss import WeightedBCELoss, explanation_ce_loss
from .model import MistSense, ModelConfig, build_mistsense
from .utils import Config, count_trainable, freeze, unfreeze


# ---------------------------------------------------------------------------
# Param groups: §4.6 — hand encoder gets its own lr/wd
# ---------------------------------------------------------------------------
def build_optimizer(model: MistSense, cfg: Config) -> torch.optim.Optimizer:
    tr = cfg.training
    hand_params, main_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("hand_encoder"):
            hand_params.append(p)
        else:
            main_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": main_params, "lr": tr["lr_main"], "weight_decay": tr["weight_decay_main"]},
            {"params": hand_params, "lr": tr["lr_hand"], "weight_decay": tr["weight_decay_hand"]},
        ],
        betas=tuple(tr["betas"]),
        eps=tr["eps"],
    )


# ---------------------------------------------------------------------------
# §4.6 — cosine schedule with linear warmup starting at 1e-6
# ---------------------------------------------------------------------------
class CosineWarmupScheduler:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, warmup_start_lr: float):
        self.optimizer = optimizer
        self.total = total_steps
        self.warmup = warmup_steps
        self.start_lr = warmup_start_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i]
            if self.step_num < self.warmup:
                # linear warmup from start_lr -> base
                frac = self.step_num / max(1, self.warmup)
                lr = self.start_lr + frac * (base - self.start_lr)
            else:
                progress = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
                lr = 0.5 * base * (1.0 + math.cos(math.pi * progress))
            group["lr"] = lr


# ---------------------------------------------------------------------------
# Stage freeze logic (§3.3)
# ---------------------------------------------------------------------------
def apply_stage(model: MistSense, stage: int) -> None:
    if stage == 1:
        # train all trainable modules in this scaffold; freeze llm_proj
        unfreeze(model)
        freeze(model.llm_proj)
    elif stage == 2:
        freeze(model)
        unfreeze(model.llm_proj)
    elif stage == 3:
        freeze(model)
        # In Stage 3 the trainer is expected to wrap the LLM with LoRA;
        # LoRA params will be marked trainable by `peft`.
    else:
        raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
# Main training entry point (Stage 1; Stage 2/3 follow same pattern using
# explanation_ce_loss + LoRA wrapped LLM — left as TODO since LLM is loaded
# externally)
# ---------------------------------------------------------------------------
def train_stage1(
    cfg: Config,
    data_root: str,
    train_annot: Optional[str] = None,
    val_annot: Optional[str] = None,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> MistSense:
    mcfg = ModelConfig(
        **{k: cfg.model[k] for k in cfg.model if k in ModelConfig.__dataclass_fields__}
    )
    model = build_mistsense(mcfg).to(device)
    apply_stage(model, stage=1)

    ds = MistakeClipDataset(
        data_root=data_root,
        segment_length=mcfg.segment_length,
        num_queries=mcfg.num_queries,
        annot_file=train_annot,
    )
    val_ds = (
        MistakeClipDataset(
            data_root=data_root,
            segment_length=mcfg.segment_length,
            num_queries=mcfg.num_queries,
            annot_file=val_annot,
        )
        if val_annot
        else None
    )
    loader = DataLoader(
        ds, batch_size=cfg.data["batch_size"], shuffle=True, num_workers=cfg.data["num_workers"], drop_last=True
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.data["batch_size"], shuffle=False, num_workers=cfg.data["num_workers"])
        if val_ds is not None
        else None
    )

    optim = build_optimizer(model, cfg)
    total_steps = cfg.training["epochs"] * max(1, len(loader))
    sched = CosineWarmupScheduler(
        optim,
        total_steps=total_steps,
        warmup_steps=int(cfg.training["warmup_ratio"] * total_steps),
        warmup_start_lr=cfg.training["warmup_start_lr"],
    )
    loss_fn = WeightedBCELoss(w0=cfg.training["bce_w0"], w1=cfg.training["bce_w1"])

    from .evaluate import detection_metrics

    best_f1m = -1.0
    best_epoch = -1
    for epoch in range(cfg.training["epochs"]):
        model.train()
        running, n = 0.0, 0
        for batch in loader:
            visual = batch["visual"].to(device)
            hand = batch["hand"].to(device)
            target = batch["target"].to(device)
            out = model(visual, hand)
            loss = loss_fn(out["logits"], target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), cfg.training["grad_clip_norm"]
            )
            optim.step()
            sched.step()
            running += loss.item() * visual.size(0)
            n += visual.size(0)
        train_loss = running / max(1, n)

        msg = f"epoch {epoch + 1}/{cfg.training['epochs']} | train_loss={train_loss:.4f}"
        if val_loader is not None:
            model.eval()
            v_logits, v_target = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch["visual"].to(device), batch["hand"].to(device))
                    v_logits.append(out["logits"].flatten().cpu())
                    v_target.append(batch["target"].flatten())
            v_logits = torch.cat(v_logits)
            v_target = torch.cat(v_target)
            m = detection_metrics(v_logits, v_target)
            msg += f" | val_f1={m['f1_overall']:.4f} f1_mistake={m['f1_mistake']:.4f}"
            if save_path is not None and m["f1_mistake"] > best_f1m:
                best_f1m = m["f1_mistake"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_path)
                msg += f" | best (saved)"
        print(msg, flush=True)

    if save_path is not None:
        print(f"best epoch: {best_epoch} (f1_mistake={best_f1m:.4f}) saved -> {save_path}")
    else:
        print("no save_path; not persisting model.")
    return model


# ---------------------------------------------------------------------------
# Stage 2 / Stage 3 — explanation training (§3.3)
# ---------------------------------------------------------------------------
def _explanation_collate(batch: list[dict], tokenizer, max_len: int) -> dict:
    """Filter empty explanations (no_mistake clips) and tokenize."""
    keep = [b for b in batch if b["explanation"].strip()]
    if not keep:
        return {}  # signal "skip batch"
    visual = torch.stack([b["visual"] for b in keep])
    hand = torch.stack([b["hand"] for b in keep])
    target = torch.stack([b["target"] for b in keep])
    texts = [b["explanation"] for b in keep]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    return {
        "visual": visual,
        "hand": hand,
        "target": target,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "clip_ids": [b["clip_id"] for b in keep],
        "explanations": texts,
    }


def _train_explanation_stage(
    cfg: Config,
    stage: int,
    data_root: str,
    train_annot: Optional[str],
    val_annot: Optional[str],
    stage1_ckpt: Optional[str],
    save_path: Optional[str],
    device: str = "cuda",
    max_text_len: int = 64,
    stage2_ckpt: Optional[str] = None,
) -> tuple[MistSense, torch.nn.Module]:
    """Stage 2 (llm_proj only) or Stage 3 (LoRA on LLM).

    Both stages share the forward path:
      visual+hand -> Q-Former -> project_to_llm -> prefix -> LLM(CE)
    Differ only in `apply_stage` and which params are trainable.
    """
    assert stage in (2, 3)
    mcfg = ModelConfig(**{k: cfg.model[k] for k in cfg.model if k in ModelConfig.__dataclass_fields__})
    model = build_mistsense(mcfg).to(device)
    if stage1_ckpt:
        sd = torch.load(stage1_ckpt, map_location=device)
        # Stage 1 ckpt may carry llm_proj with the *old* llm_hidden (e.g. 4096
        # when targeted at Llama2-7B). llm_proj is randomly initialized in
        # Stage 1 (frozen) so we strip it and let the current llm_hidden win.
        sd = {k: v for k, v in sd.items() if not k.startswith("llm_proj.")}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"loaded stage1: miss={len(miss)} unexp={len(unexp)}")
    if stage2_ckpt:
        sd2 = torch.load(stage2_ckpt, map_location=device)
        proj_sd = sd2["llm_proj"] if isinstance(sd2, dict) and "llm_proj" in sd2 else sd2
        model.llm_proj.load_state_dict(proj_sd)
        print(f"loaded stage2 llm_proj from {stage2_ckpt}")

    # LLM
    llm, tokenizer, llm_hidden = build_llm(
        pretrained=cfg.model.get("llm_pretrained", "Salesforce/blip2-opt-2.7b"),
        from_blip2=bool(cfg.model.get("llm_from_blip2", True)),
        dtype=torch.float16,
        device=device,
    )
    assert llm_hidden == mcfg.llm_hidden, (
        f"config llm_hidden={mcfg.llm_hidden} but LLM exposes {llm_hidden}; "
        f"update configs/egoops.yaml model.llm_hidden"
    )

    # Stage freeze logic
    apply_stage(model, stage=stage)
    if stage == 2:
        for p in llm.parameters():
            p.requires_grad = False
    elif stage == 3:
        # Train LoRA only; freeze base LLM weights inside the wrapper
        for p in llm.parameters():
            p.requires_grad = False
        llm = lora_wrap(
            llm,
            r=mcfg.lora_r,
            alpha=mcfg.lora_alpha,
            dropout=mcfg.lora_dropout,
            target_modules=mcfg.lora_target_modules,
        )
        # llm_proj stays frozen in stage 3 per paper §3.3
        freeze(model.llm_proj)
        llm.print_trainable_parameters()

    # Datasets: only clips with non-empty explanations
    ds = MistakeClipDataset(
        data_root=data_root,
        segment_length=mcfg.segment_length,
        num_queries=mcfg.num_queries,
        annot_file=train_annot,
    )
    val_ds = (
        MistakeClipDataset(
            data_root=data_root,
            segment_length=mcfg.segment_length,
            num_queries=mcfg.num_queries,
            annot_file=val_annot,
        )
        if val_annot
        else None
    )
    coll = lambda b: _explanation_collate(b, tokenizer, max_text_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.data["batch_size"],
        shuffle=True,
        num_workers=cfg.data["num_workers"],
        collate_fn=coll,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.data["batch_size"], shuffle=False,
                   num_workers=cfg.data["num_workers"], collate_fn=coll)
        if val_ds is not None else None
    )

    # Optimizer: gather all trainable params (mistake-detection backbone is
    # frozen in stages 2/3; only llm_proj OR LoRA is trainable)
    trainable = [p for p in model.parameters() if p.requires_grad] + \
                [p for p in llm.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"stage {stage}: trainable params = {n_trainable:,}")
    optim = torch.optim.AdamW(
        trainable,
        lr=cfg.training["lr_main"],
        weight_decay=cfg.training["weight_decay_main"],
        betas=tuple(cfg.training["betas"]),
        eps=cfg.training["eps"],
    )
    total_steps = cfg.training["epochs"] * max(1, len(loader))
    sched = CosineWarmupScheduler(
        optim,
        total_steps=total_steps,
        warmup_steps=int(cfg.training["warmup_ratio"] * total_steps),
        warmup_start_lr=cfg.training["warmup_start_lr"],
    )

    best_loss = float("inf")
    best_epoch = -1
    embed_layer = llm.get_input_embeddings()
    for epoch in range(cfg.training["epochs"]):
        model.train()
        llm.train()
        running, n = 0.0, 0
        for batch in loader:
            if not batch:  # all-empty after filtering
                continue
            visual = batch["visual"].to(device)
            hand = batch["hand"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            f_rgb = model.encode_video(visual)
            prefix = model.project_to_llm(f_rgb).to(torch.float16)
            inputs_embeds, full_mask, labels = build_llm_inputs(
                prefix, input_ids, attn, embed_layer
            )
            out = llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
            loss = out.loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.training["grad_clip_norm"])
            optim.step()
            sched.step()
            running += loss.item() * input_ids.size(0)
            n += input_ids.size(0)
        train_loss = running / max(1, n)

        msg = f"[stage{stage}] epoch {epoch + 1}/{cfg.training['epochs']} | train_ce={train_loss:.4f}"
        if val_loader is not None:
            model.eval()
            llm.eval()
            v_sum, v_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if not batch:
                        continue
                    visual = batch["visual"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attn = batch["attention_mask"].to(device)
                    f_rgb = model.encode_video(visual)
                    prefix = model.project_to_llm(f_rgb).to(torch.float16)
                    inputs_embeds, full_mask, labels = build_llm_inputs(
                        prefix, input_ids, attn, embed_layer
                    )
                    out = llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
                    v_sum += out.loss.item() * input_ids.size(0)
                    v_n += input_ids.size(0)
            val_loss = v_sum / max(1, v_n)
            msg += f" | val_ce={val_loss:.4f}"
            if save_path is not None and val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                _save_explanation_ckpt(model, llm, stage, save_path)
                msg += " | best (saved)"
        print(msg, flush=True)

    if save_path and best_epoch < 0:
        # No val_loader; save final ckpt
        _save_explanation_ckpt(model, llm, stage, save_path)
    print(f"stage {stage} done. best epoch: {best_epoch} val_ce={best_loss:.4f}")
    return model, llm


def _save_explanation_ckpt(model: MistSense, llm: torch.nn.Module, stage: int, save_path: str) -> None:
    """Stage 2 saves llm_proj only; Stage 3 saves LoRA adapters separately."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"llm_proj": model.llm_proj.state_dict(), "stage": stage}, str(p))
    if stage == 3:
        # peft model has .save_pretrained
        lora_dir = p.parent / (p.stem + "_lora")
        llm.save_pretrained(str(lora_dir))


def train_stage2(cfg: Config, **kw) -> tuple[MistSense, torch.nn.Module]:
    return _train_explanation_stage(cfg, stage=2, **kw)


def train_stage3(cfg: Config, **kw) -> tuple[MistSense, torch.nn.Module]:
    return _train_explanation_stage(cfg, stage=3, **kw)


def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--override", default="configs/egoops.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_annot", default=None)
    p.add_argument("--val_annot", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--stage1_ckpt", default=None,
                   help="Stage 1 checkpoint for stages 2/3 (frozen backbone)")
    p.add_argument("--stage2_ckpt", default=None,
                   help="Stage 2 checkpoint with llm_proj weights (used in stage 3)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = Config.load(args.base, args.override)
    if args.stage == 1:
        train_stage1(
            cfg,
            data_root=args.data_root,
            train_annot=args.train_annot,
            val_annot=args.val_annot,
            device=args.device,
            save_path=args.save,
        )
    else:
        fn = train_stage2 if args.stage == 2 else train_stage3
        kw = dict(
            data_root=args.data_root,
            train_annot=args.train_annot,
            val_annot=args.val_annot,
            stage1_ckpt=args.stage1_ckpt,
            save_path=args.save,
            device=args.device,
        )
        if args.stage == 3:
            kw["stage2_ckpt"] = args.stage2_ckpt
        fn(cfg, **kw)


if __name__ == "__main__":
    _cli()
