from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from vmd.data.mistake_classes import MISTAKE_CLASSES
from vmd.vlm.boolean_scoring import SCORE_VERSION, deviation_probability

_PROMPTS_DIR = Path(__file__).parent / "prompts"
# "graph" = full procedural-graph conditioning; "nograph" = graph block stripped
# (task name kept) for the graph-off ablation. Same JSON output schema.
_TEMPLATES = {
    "graph": (_PROMPTS_DIR / "reason.txt").read_text(),
    "nograph": (_PROMPTS_DIR / "reason_nograph.txt").read_text(),
    # Same graph CONTENT as "graph", but the whole step list is dumped and the
    # model is not told where in it the person should be. Isolates the value of
    # the walked step pointer from the value of having the plan at all.
    "staticdump": (_PROMPTS_DIR / "reason_staticdump.txt").read_text(),
    # GraphTrack: lettered step menu + tracked belief state, compact JSON output
    # whose first field is the step letter (read as a distribution, see
    # vmd.vlm.step_scoring). "belief_nograph" keeps the menu, drops the state.
    "belief": (_PROMPTS_DIR / "reason_belief.txt").read_text(),
    "belief_nograph": (_PROMPTS_DIR / "reason_belief_nograph.txt").read_text(),
    # Dedicated step recogniser: menu only, longer look-back context, step letter
    # as the only output. Paired with a separate content detector at eval time.
    "stepreco": (_PROMPTS_DIR / "reason_stepreco.txt").read_text(),
}
STEP_ONLY_VARIANTS = ("stepreco",)

_VALID_CLASSES = set(MISTAKE_CLASSES)
_VALID_UNCERTAINTY = {
    "none", "occlusion", "ambiguous_action",
    "valid_variation", "short_clip", "weak_graph_match",
}


class VLMPrediction:
    def __init__(self, raw: dict) -> None:
        self.observed_action: str = raw.get("observed_action", "")
        self.observed_objects: List[str] = raw.get("observed_objects", [])
        self.step_id: Optional[str] = raw.get("step_id")
        # GraphTrack menu letter ("A".."Y" or "-"), None for the legacy schema.
        self.step_letter: Optional[str] = raw.get("step")
        self.deviation: bool = bool(raw.get("deviation", False))
        self.mistake_class: str = raw.get("mistake_class", "no_mistake")
        # The model's *verbalized* confidence is degenerate after LoRA (it learned
        # to emit a constant, e.g. 0.9). We keep it for reference but `confidence`
        # is overwritten downstream with the token-probability of the deviation
        # decision (see Qwen25VL._run / _parse_output), which actually varies.
        self.verbalized_confidence: float = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        self.confidence: float = self.verbalized_confidence
        self.uncertainty_reason: str = raw.get("uncertainty_reason", "none")
        self.rationale: str = raw.get("rationale", "")
        self.expected_vs_observed: str = raw.get("expected_vs_observed", "")

        # clamp
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.mistake_class not in _VALID_CLASSES:
            self.mistake_class = "no_mistake"
        if self.uncertainty_reason not in _VALID_UNCERTAINTY:
            self.uncertainty_reason = "none"


class Qwen25VL:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        lora_adapter_dir: Optional[str] = None,
        prompt_variant: str = "graph",
        revision: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        torch_dtype = getattr(torch, dtype)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            revision=revision,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.max_new_tokens = max_new_tokens
        if prompt_variant not in _TEMPLATES:
            raise ValueError(f"prompt_variant {prompt_variant!r} not in {sorted(_TEMPLATES)}")
        self.prompt_variant = prompt_variant

        if lora_adapter_dir is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_dir)
            self.model = self.model.merge_and_unload()

        self.score_version = SCORE_VERSION
        self.model.eval()
        self._label_table = None

    @property
    def label_table(self):
        from vmd.vlm.step_scoring import StepLabelTable
        if self._label_table is None:
            self._label_table = StepLabelTable(self.processor.tokenizer)
        return self._label_table

    def predict_scored(
        self,
        frames: np.ndarray,
        graph_context: dict,
        history_summary: str = "",
        n_steps: int = 0,
        history_len: int = 3,
    ):
        """``predict`` plus the step distribution read from the letter token.

        Returns (VLMPrediction, step_probs or None, emitted_menu_index or None).
        ``step_probs`` has ``n_steps + 1`` entries; the last is the transit label.
        """
        from vmd.vlm.step_scoring import step_probs_from_scores
        prompt = _build_prompt(graph_context, history_summary, history_len,
                               variant=self.prompt_variant)
        images = [Image.fromarray(f) for f in frames]
        raw_output, token_conf, gen_ids, scores = self._run(prompt, images, keep_scores=True)
        pred = _parse_output(raw_output, token_conf)
        probs, emitted = (None, None)
        if n_steps > 0 and scores:
            try:
                probs, emitted = step_probs_from_scores(gen_ids.tolist(), scores, self.label_table, n_steps,
                                                        tokenizer=self.processor.tokenizer)
            except ValueError:
                probs, emitted = None, None
        return pred, probs, emitted

    def predict(
        self,
        frames: np.ndarray,
        graph_context: dict,
        history_summary: str = "",
        history_len: int = 3,
    ) -> VLMPrediction:
        prompt = _build_prompt(graph_context, history_summary, history_len,
                               variant=self.prompt_variant)
        images = [Image.fromarray(f) for f in frames]
        raw_output, token_conf, _, _ = self._run(prompt, images)
        return _parse_output(raw_output, token_conf)

    def _run(self, text_prompt: str, images: List[Image.Image], keep_scores: bool = False):
        """Return text, conditional boolean score, generated ids and optional logits.

        The score is normalized over boolean candidates at the generated field
        prefix. It is not automatically calibrated real-world uncertainty.
        """
        import torch
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
        token_conf = self._deviation_token_confidence(gen_ids, out.scores)
        return decoded, token_conf, gen_ids, (out.scores if keep_scores else None)

    def _deviation_token_confidence(self, gen_ids, scores) -> Optional[float]:
        """Conditional true-vs-false score; no sequence-confidence fallback."""
        return deviation_probability(self.processor.tokenizer, gen_ids.tolist(), scores)


def _build_prompt(
    ctx: dict, history_summary: str, history_len: int, variant: str = "graph"
) -> str:
    replacements = {
        "{task_name}": ctx.get("task_name", "Unknown"),
        "{current_step_id}": ctx.get("current_step_id", "unknown"),
        "{current_step_desc}": ctx.get("current_step_desc", ""),
        "{valid_next_steps}": ctx.get("valid_next_steps", "[]"),
        "{required_objects}": ctx.get("required_objects", "[]"),
        "{expected_states}": ctx.get("expected_states", "[]"),
        "{valid_variations}": ctx.get("valid_variations", "[]"),
        "{all_steps}": ctx.get("all_steps", "[]"),
        # GraphTrack placeholders; legacy variants never contain them.
        "{step_menu}": ctx.get("step_menu", ""),
        "{tracked_letter}": ctx.get("tracked_letter", "?"),
        "{tracked_desc}": ctx.get("tracked_desc", "unknown"),
        "{tracked_p}": ctx.get("tracked_p", "?"),
        "{completed_letters}": ctx.get("completed_letters", "none"),
        "{next_letters}": ctx.get("next_letters", "unknown"),
        "{context_sec}": str(ctx.get("context_sec", 4)),
        "{history_len}": str(history_len),
        "{history_summary}": history_summary or "No prior clips.",
    }
    prompt = _TEMPLATES[variant]
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_output(raw: str, token_conf: Optional[float] = None) -> VLMPrediction:
    """Keep parse/scoring failures explicit rather than making normal predictions.

    Invalid records are saved for diagnosis, but cannot enter primary metrics.
    """
    data = None
    m = _JSON_RE.search(raw)
    if m:
        try:
            parsed = json.loads(m.group())
            if (isinstance(parsed, dict) and type(parsed.get("deviation")) is bool
                    and parsed.get("mistake_class") in _VALID_CLASSES):
                data = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        pred = VLMPrediction(data or {})
    except (TypeError, ValueError):
        data = None
        pred = VLMPrediction({})
    pred.parse_ok = data is not None
    pred.score_version = SCORE_VERSION
    pred.score_status = ("ok" if pred.parse_ok and token_conf is not None else
                         "parse_error" if not pred.parse_ok else "missing_boolean_score")
    pred.confidence = float(token_conf) if pred.score_status == "ok" else None
    if not pred.parse_ok:
        pred.uncertainty_reason = "weak_graph_match"
        pred.rationale = f"Failed to parse VLM output: {raw[:200]}"
    return pred
