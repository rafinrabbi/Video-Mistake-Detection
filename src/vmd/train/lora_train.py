"""LoRA fine-tuning for Qwen2.5-VL-7B on procedural mistake detection.

Operates on CaptainCook4D via ``vmd.data.get_dataset``. Clips are exposed
through the legacy ``EgoOopsClip`` dataclass (kept for naming continuity).
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from vmd.vlm.lora import get_lora_config, save_adapter
from vmd.vlm.qwen25vl import _build_prompt
from vmd.data import get_dataset
from vmd.graph.schema import ProceduralGraph
from vmd.data.step_tracking import gold_step_span, graph_index_of_span
from vmd.data.sampler import online_clips, _load_frames

import wandb
from PIL import Image
from tqdm import tqdm


class MistakeDataset(Dataset):
    """Lazy + class-balanced window index.

    The index pass (online_clips with load_frames=False) is cheap — it stores
    only window time-bounds + label, never pixels. Frames are decoded per item
    in __getitem__. Negatives are subsampled to ``neg_per_pos`` and the total
    is capped at ``max_windows`` so CC4D's ~90k windows stay tractable in RAM
    and compute. Pass ``max_windows=0`` / ``neg_per_pos<=0`` to keep all
    windows.
    """

    def __init__(
        self,
        dataset,
        graphs: dict,
        clip_len_sec: float = 4.0,
        stride_sec: float = 2.0,
        fps: float = 1.0,
        neg_per_pos: float = 1.0,
        max_windows: int = 0,
        seed: int = 0,
    ) -> None:
        self.fps = fps
        pos, neg = [], []
        for clip in dataset.clips:
            graph = graphs.get(clip.task_id)
            if graph is None:
                continue
            for i, window in enumerate(online_clips(
                clip.video_path, clip.clip_id, clip.task_id,
                clip.duration_sec, clip.annotations,
                clip_len_sec, stride_sec, fps, load_frames=False,
            )):
                # Teacher-force on the benchmark's gold step segmentation.
                # Previously this was graph.steps[min(i, n-1)] -- the window
                # index -- which pinned 93% of training windows to the final
                # step and left the model almost no positional signal to learn
                # from. Fall back to the old rule only if a recording has no
                # gold spans.
                gi = None
                if clip.step_spans:
                    mid = (window.start_sec + window.end_sec) / 2.0
                    gi = graph_index_of_span(
                        graph.steps, gold_step_span(clip.step_spans, mid))
                step = graph.steps[gi] if gi is not None else \
                    graph.steps[min(i, len(graph.steps) - 1)]
                next_steps = graph.valid_next_steps(step.id)
                ctx = {
                    "task_name": clip.task_name,
                    "current_step_id": step.id,
                    "current_step_desc": step.description,
                    "valid_next_steps": str([s.description for s in next_steps]),
                    "required_objects": str(step.required_object_ids),
                    "expected_states": str(step.produced_state_ids),
                    "valid_variations": str(step.valid_variations),
                }
                target = json.dumps({
                    "deviation": bool(window.true_label),
                    "mistake_class": window.true_class,
                    "confidence": 0.9,
                    "uncertainty_reason": "none",
                    "rationale": "Ground truth label.",
                    "observed_action": "",
                    "observed_objects": [],
                    "step_id": step.id,
                    "expected_vs_observed": "",
                })
                rec = {
                    "video_path": str(clip.video_path),
                    "start_sec": window.start_sec,
                    "end_sec": window.end_sec,
                    "ctx": ctx,
                    "target": target,
                }
                (pos if window.true_label else neg).append(rec)

        rng = random.Random(seed)
        rng.shuffle(neg)
        if neg_per_pos > 0:
            neg = neg[: int(len(pos) * neg_per_pos)]
        samples = pos + neg
        rng.shuffle(samples)
        if max_windows and len(samples) > max_windows:
            samples = samples[:max_windows]
        self.samples = samples
        self.n_pos = sum(1 for s in samples if json.loads(s["target"])["deviation"])
        print(f"MistakeDataset: {len(samples)} windows "
              f"({self.n_pos} pos / {len(samples)-self.n_pos} neg) "
              f"from {len(pos)} pos + {len(neg)} neg candidates")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        frames = _load_frames(Path(s["video_path"]), s["start_sec"], s["end_sec"], self.fps)
        return {"frames": torch.from_numpy(frames), "ctx": s["ctx"], "target": s["target"]}


def train(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    dataset_name: str = "captaincook4d",
    raw_dir: str = "data/raw/captaincook4d",
    graphs_dir: str = "data/graphs",
    output_dir: str = "outputs/lora_adapters",
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    warmup_ratio: float = 0.05,
    lora_r: int = 32,
    lora_alpha: int = 64,
    clip_len_sec: float = 4.0,
    stride_sec: float = 2.0,
    fps: float = 1.0,
    neg_per_pos: float = 1.0,
    max_windows: int = 0,
    num_workers: int = 4,
    wandb_project: str = "vmd-mistake",
    use_wandb: bool = True,
    prompt_variant: str = "graph",
    checkpoint_every: int = 50,
    resume: bool = False,
    seed: int = 42,
) -> None:
    from peft import (get_peft_model, get_peft_model_state_dict,
                      set_peft_model_state_dict)

    raw_path = Path(raw_dir)
    graphs_path = Path(graphs_dir)
    out_path = Path(output_dir)

    graphs = {
        p.stem: ProceduralGraph.model_validate_json(p.read_text())
        for p in graphs_path.glob("*.json")
    }

    train_ds = get_dataset(dataset_name, raw_path, "train")
    dataset = MistakeDataset(
        train_ds, graphs,
        clip_len_sec=clip_len_sec, stride_sec=stride_sec, fps=fps,
        neg_per_pos=neg_per_pos, max_windows=max_windows,
    )
    # Epoch order comes from a seeded permutation rather than DataLoader(shuffle=True),
    # so a resumed epoch can replay the same order and skip the completed prefix by
    # index -- without decoding the frames of batches it is going to discard.
    def epoch_loader(epoch: int, skip_batches: int = 0):
        g = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(dataset), generator=g).tolist()
        order = order[skip_batches * batch_size:]
        return DataLoader(Subset(dataset, order), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)

    n_batches = (len(dataset) + batch_size - 1) // batch_size

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = get_peft_model(model, get_lora_config(r=lora_r, lora_alpha=lora_alpha))
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * n_batches // grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps
    )

    if use_wandb:
        wandb.init(project=wandb_project, config={
            "lr": lr, "epochs": epochs, "dataset": dataset_name,
            "n_windows": len(dataset), "n_pos": dataset.n_pos,
            "neg_per_pos": neg_per_pos, "stride_sec": stride_sec,
            "prompt_variant": prompt_variant,
        })

    ckpt_path = out_path / "training_state.pt"
    start_epoch, start_batch, step = 0, 0, 0
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_batch, step = state["epoch"], state["batch_idx"] + 1, state["step"]
        print(f"[resume] epoch {start_epoch+1}, batch {start_batch}, optimiser step {step}",
              flush=True)

    def save_state(epoch: int, batch_idx: int, step: int) -> None:
        # Written to a temp file and renamed: a checkpoint interrupted mid-write
        # must not replace a good one with a truncated file.
        out_path.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({
            "adapter": get_peft_model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "batch_idx": batch_idx, "step": step,
        }, tmp)
        tmp.replace(ckpt_path)

    model.train()
    for epoch in range(start_epoch, epochs):
        skip = start_batch if epoch == start_epoch else 0
        loader = epoch_loader(epoch, skip)
        for offset, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            batch_idx = skip + offset
            frames = batch["frames"]     # (B, T, H, W, C)
            ctx_list = batch["ctx"]
            targets = batch["target"]

            loss_accum = torch.tensor(0.0, device="cuda")
            for b in range(len(targets)):
                imgs = [Image.fromarray(f.numpy()) for f in frames[b]]
                prompt = _build_prompt(
                    {k: v[b] for k, v in ctx_list.items()}, "", 3,
                    variant=prompt_variant,
                )
                messages = [{"role": "user", "content":
                    [{"type": "image", "image": img} for img in imgs] +
                    [{"type": "text", "text": prompt}]
                }]
                text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                target_full = text_in + targets[b] + processor.tokenizer.eos_token

                inputs = processor(text=[target_full], images=imgs, return_tensors="pt").to("cuda")
                # Completion-only loss: mask the prompt + image tokens with -100 so
                # CE is computed ONLY over the target JSON. Including the prompt /
                # image-placeholder tokens drowns the target signal (loss plateaus).
                prompt_inputs = processor(text=[text_in], images=imgs, return_tensors="pt")
                prompt_len = prompt_inputs["input_ids"].shape[1]
                labels = inputs["input_ids"].clone()
                labels[:, :prompt_len] = -100
                out = model(**inputs, labels=labels)
                loss_accum = loss_accum + out.loss / grad_accum

            loss_accum.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                loss_val = loss_accum.item() * grad_accum
                if use_wandb:
                    wandb.log({"loss": loss_val, "step": step})
                if step % 25 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"[e{epoch+1} step {step}/{total_steps}] loss={loss_val:.4f} lr={lr_now:.2e}",
                          flush=True)
                # Checkpoint only on a grad-accum boundary: the optimiser has just
                # stepped and no gradient is half-accumulated, so resuming here is
                # a clean restart rather than a dropped partial batch.
                if checkpoint_every and step % checkpoint_every == 0:
                    save_state(epoch, batch_idx, step)

    save_adapter(model, out_path)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB "
              f"allocated, {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB reserved",
              flush=True)
    # The run finished: drop the resume state so a later --resume does not
    # silently restart a completed job mid-way.
    ckpt_path.unlink(missing_ok=True)
    print(f"Adapters saved to {out_path}")


if __name__ == "__main__":
    train()
