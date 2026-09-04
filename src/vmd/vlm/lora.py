from __future__ import annotations
from pathlib import Path
from typing import List, Optional


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def add_lora(model, config=None):
    from peft import get_peft_model
    if config is None:
        config = get_lora_config()
    return get_peft_model(model, config)


def save_adapter(model, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))


def load_adapter(model, adapter_dir: Path):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, str(adapter_dir))
