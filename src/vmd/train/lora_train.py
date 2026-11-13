"""LoRA fine-tuning for Qwen2.5-VL-7B on procedural mistake detection.

Operates on CaptainCook4D via ``vmd.data.get_dataset``. Clips are exposed
through the legacy ``EgoOopsClip`` dataclass (kept for naming continuity).

Two prompt families are trained here:

* legacy ``graph`` / ``nograph``: gold-step context, verbose JSON target
  (R4/R6/R11 lineage; kept byte-compatible);
* GraphTrack ``belief`` / ``belief_nograph``: lettered step menu, a tracked
  state block, and the compact target ``{"step":L,"deviation":b,"mistake_class":c}``.
  The step letter is supervised from the gold segmentation, and the tracked
  state shown in the prompt is CORRUPTED at a configurable rate so training
  matches the noisy state the causal filter delivers at test time. The letter
  target is always gold, so the model cannot solve it by copying the prompt.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from vmd.vlm.lora import get_lora_config, save_adapter
from vmd.vlm.qwen25vl import _build_prompt
from vmd.vlm.step_scoring import STEP_PREFIX, build_menu, compact_target
from vmd.data import get_dataset
from vmd.graph.schema import ProceduralGraph
from vmd.data.step_tracking import (gold_span_index_at, gold_step_span, graph_index_of_span,
                                    match_spans_to_graph)
from vmd.data.sampler import online_clips, _load_frames
from vmd.tracking.belief import MENU_LETTERS, TRANSIT_LABEL, build_step_space

import wandb
from PIL import Image
from tqdm import tqdm

BELIEF_VARIANTS = ("belief", "belief_nograph", "stepreco")
STEP_ONLY_VARIANTS = ("stepreco",)
CONTEXT_MODES = ("gold", "corrupt")


def _ready(j: int, completed: set, pred: List[List[int]]) -> bool:
    return all(p in completed for p in pred[j])


def belief_context(space, gold_idx: Optional[int], completed: List[int], rng: random.Random,
                   corrupt_rate: float, prev_gold: Optional[int]) -> Dict[str, str]:
    """Tracked-state block for one training window.

    With probability ``corrupt_rate`` the shown step is wrong in the ways the
    filter is wrong at test time: a DAG neighbour or ready sibling (60%), the
    previous step -- a lagging tracker (25%), or any step (15%). The completed
    set loses or gains one item with probability 0.2. Confidence is 0.9 when
    clean and 0.6 when corrupted, mimicking a less certain posterior.
    """
    n = space.n
    shown = gold_idx if gold_idx is not None else n          # transit
    p_shown = 0.9
    comp = set(completed)
    if rng.random() < corrupt_rate:
        p_shown = 0.6
        u = rng.random()
        if u < 0.6 and gold_idx is not None:
            nb = set(space.succ[gold_idx]) | set(space.pred[gold_idx]) | \
                 {j for j in range(n) if j not in comp and j != gold_idx and _ready(j, comp, space.pred)}
            nb.discard(gold_idx)
            shown = rng.choice(sorted(nb)) if nb else rng.randrange(n)
        elif u < 0.85 and prev_gold is not None:
            shown = prev_gold
        else:
            shown = rng.randrange(n + 1)                      # any step or transit
    if rng.random() < 0.2 and n > 1:
        if comp and rng.random() < 0.5:
            comp.discard(rng.choice(sorted(comp)))
        else:
            extra = [j for j in range(n) if j not in comp]
            if extra:
                comp.add(rng.choice(extra))
    ready_next = [j for j in range(n) if j not in comp and _ready(j, comp, space.pred)]
    if shown < n:
        ready_next = [j for j in ready_next if j != shown]
    letter = TRANSIT_LABEL if shown >= n else MENU_LETTERS[shown]
    desc = "between steps / no recipe step in progress" if shown >= n else space.descriptions[shown]
    return {
        "step_menu": build_menu(space.descriptions),
        "tracked_letter": letter,
        "tracked_desc": desc,
        "tracked_p": f"{p_shown:.2f}",
        "completed_letters": ", ".join(MENU_LETTERS[i] for i in sorted(comp)) if comp else "none",
        "next_letters": ", ".join(MENU_LETTERS[i] for i in ready_next) if ready_next else "none",
    }


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
        prompt_variant: str = "graph",
        context_mode: str = "gold",
        corrupt_rate: float = 0.35,
        context_sec: float = 0.0,
        context_fps: float = 1.0,
    ) -> None:
        self.fps = fps
        self.context_sec = context_sec
        self.context_fps = context_fps
        self.clip_len_sec = clip_len_sec
        self.prompt_variant = prompt_variant
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode {context_mode!r} not in {CONTEXT_MODES}")
        belief = prompt_variant in BELIEF_VARIANTS
        rng_ctx = random.Random(seed + 1)
        rate = corrupt_rate if context_mode == "corrupt" else 0.0
        pos, neg = [], []
        n_step_targets = 0
        for clip in dataset.clips:
            graph = graphs.get(clip.task_id)
            if graph is None:
                continue
            space = build_step_space(graph) if belief else None
            span_to_pos = match_spans_to_graph(clip.step_spans, graph.steps,
                                               order=space.graph_pos if belief else None) if belief else {}
            prev_gold = None
            for i, window in enumerate(online_clips(
                clip.video_path, clip.clip_id, clip.task_id,
                clip.duration_sec, clip.annotations,
                clip_len_sec, stride_sec, fps, load_frames=False,
            )):
                mid = (window.start_sec + window.end_sec) / 2.0
                if belief:
                    si = gold_span_index_at(clip.step_spans, mid, strict=True)
                    gold_idx = space.menu_index_of_graph_pos(span_to_pos[si]) \
                        if (si is not None and si in span_to_pos) else None
                    completed = sorted({space.menu_index_of_graph_pos(span_to_pos[k])
                                        for k, sp in enumerate(clip.step_spans)
                                        if k in span_to_pos and 0 <= sp.end_sec <= window.start_sec
                                        and space.menu_index_of_graph_pos(span_to_pos[k]) is not None})
                    letter = TRANSIT_LABEL if gold_idx is None else MENU_LETTERS[gold_idx]
                    if prompt_variant in STEP_ONLY_VARIANTS:
                        ctx = {"task_name": clip.task_name, "step_menu": build_menu(space.descriptions),
                               "context_sec": str(int(context_sec) if context_sec else int(clip_len_sec))}
                        target = json.dumps({"step": letter}, separators=(",", ":"))
                    else:
                        ctx = {"task_name": clip.task_name,
                               **belief_context(space, gold_idx, completed, rng_ctx, rate, prev_gold)}
                        target = compact_target(letter, bool(window.true_label), window.true_class)
                    n_step_targets += gold_idx is not None
                    if gold_idx is not None:
                        prev_gold = gold_idx
                else:
                    # Teacher-force on the benchmark's gold step segmentation.
                    # Previously this was graph.steps[min(i, n-1)] -- the window
                    # index -- which pinned 93% of training windows to the final
                    # step and left the model almost no positional signal to learn
                    # from. Fall back to the old rule only if a recording has no
                    # gold spans.
                    gi = None
                    if clip.step_spans:
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
        self.n_pos = sum(1 for s in samples if json.loads(s["target"]).get("deviation", False))
        print(f"MistakeDataset[{prompt_variant}/{context_mode}@{rate}]: {len(samples)} windows "
              f"({self.n_pos} pos / {len(samples)-self.n_pos} neg) "
              f"from {len(pos)} pos + {len(neg)} neg candidates"
              + (f"; {n_step_targets} windows with a gold step letter" if belief else ""))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        if self.context_sec and self.context_sec > self.clip_len_sec:
            frames = _load_frames(Path(s["video_path"]), max(0.0, s["end_sec"] - self.context_sec),
                                  s["end_sec"], self.context_fps)
        else:
            frames = _load_frames(Path(s["video_path"]), s["start_sec"], s["end_sec"], self.fps)
        return {"frames": torch.from_numpy(frames), "ctx": s["ctx"], "target": s["target"]}


def completion_loss(model, inputs, prompt_len: int, letter_weight: float,
                    letter_prefix_ids: Optional[List[int]]) -> torch.Tensor:
    """Completion-only CE with an optional up-weight on the step-letter token.

    The prompt + image tokens are masked; including them drowns the target
    signal (loss plateaus). The letter token is located as the first completion
    token preceded by the ``{"step":"`` prefix ids.
    """
    out = model(**inputs)
    logits = out.logits[:, :-1, :].float()
    ids = inputs["input_ids"]
    labels = ids[:, 1:].clone()
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, : max(prompt_len - 1, 0)] = 0.0
    if letter_weight != 1.0 and letter_prefix_ids:
        seq = ids[0].tolist(); p = letter_prefix_ids
        for k in range(prompt_len + len(p), len(seq)):
            if seq[k - len(p):k] == p:
                weights[0, k - 1] = letter_weight   # labels are shifted by one
                break
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).view_as(labels)
    return (ce * weights).sum() / weights.sum().clamp_min(1.0)


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
    context_mode: str = "gold",
    corrupt_rate: float = 0.35,
    step_loss_weight: float = 2.0,
    recipe_holdout: Optional[str] = None,
    context_sec: float = 0.0,
    context_fps: float = 1.0,
    gradient_checkpointing: bool = False,
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

    ds_kwargs = {"recipe_holdout": Path(recipe_holdout)} if recipe_holdout else {}
    train_ds = get_dataset(dataset_name, raw_path, "train", **ds_kwargs)
    dataset = MistakeDataset(
        train_ds, graphs,
        clip_len_sec=clip_len_sec, stride_sec=stride_sec, fps=fps,
        neg_per_pos=neg_per_pos, max_windows=max_windows, seed=seed,
        prompt_variant=prompt_variant, context_mode=context_mode, corrupt_rate=corrupt_rate,
        context_sec=context_sec, context_fps=context_fps,
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
    if gradient_checkpointing:
        # Activations dominate training memory (8 frames of context roughly
        # doubles them); recompute them in the backward pass instead of storing.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("gradient checkpointing enabled", flush=True)
    model.print_trainable_parameters()
    belief = prompt_variant in BELIEF_VARIANTS
    letter_prefix_ids = processor.tokenizer(STEP_PREFIX, add_special_tokens=False)["input_ids"] if belief else None

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
            "prompt_variant": prompt_variant, "context_mode": context_mode,
            "corrupt_rate": corrupt_rate, "step_loss_weight": step_loss_weight,
            "recipe_holdout": recipe_holdout,
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
                prompt_inputs = processor(text=[text_in], images=imgs, return_tensors="pt")
                prompt_len = prompt_inputs["input_ids"].shape[1]
                loss = completion_loss(model, inputs, prompt_len,
                                       step_loss_weight if belief else 1.0, letter_prefix_ids)
                loss_accum = loss_accum + loss / grad_accum

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
    (out_path / "train_config.json").write_text(json.dumps({
        "prompt_variant": prompt_variant, "context_mode": context_mode, "corrupt_rate": corrupt_rate,
        "step_loss_weight": step_loss_weight, "recipe_holdout": recipe_holdout, "epochs": epochs,
        "lr": lr, "lora_r": lora_r, "lora_alpha": lora_alpha, "max_windows": max_windows,
        "neg_per_pos": neg_per_pos, "stride_sec": stride_sec, "seed": seed,
        "context_sec": context_sec, "context_fps": context_fps,
        "gradient_checkpointing": gradient_checkpointing,
        "n_windows": len(dataset), "n_pos": dataset.n_pos}, indent=1))
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
