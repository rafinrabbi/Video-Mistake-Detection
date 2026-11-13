"""vmd CLI — build-graph | infer | eval | train"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("build-graph")
def build_graph_cmd(
    task_name: str = typer.Option(..., help="Task name"),
    task_description: str = typer.Option(..., help="Task description"),
    task_id: Optional[str] = typer.Option(None),
    cache_dir: Path = typer.Option(Path("data/graphs")),
    model: str = typer.Option("gpt-4o"),
) -> None:
    """Build + cache a procedural graph for a task."""
    from vmd.graph.builder import build_graph
    graph = build_graph(task_name, task_description, task_id, cache_dir, model)
    out = cache_dir / f"{graph.task_id}.json"
    typer.echo(f"Graph saved: {out}  ({len(graph.steps)} steps)")


@app.command("infer")
def infer_cmd(
    video: Path = typer.Argument(..., help="Path to video file"),
    graph_path: Path = typer.Argument(..., help="Path to procedural graph JSON"),
    model_id: str = typer.Option("Qwen/Qwen2.5-VL-7B-Instruct"),
    lora_dir: Optional[Path] = typer.Option(None),
    clip_len: float = typer.Option(4.0),
    stride: float = typer.Option(2.0),
    fps: float = typer.Option(1.0),
    out: Optional[Path] = typer.Option(None),
) -> None:
    """Run per-clip inference on a single video."""
    from vmd.graph.schema import ProceduralGraph
    from vmd.vlm.qwen25vl import Qwen25VL
    from vmd.data.egoops import MistakeAnnotation
    from vmd.data.sampler import online_clips
    from vmd.eval.online import _summarise_history

    graph = ProceduralGraph.model_validate_json(graph_path.read_text())
    vlm = Qwen25VL(model_id, lora_adapter_dir=str(lora_dir) if lora_dir else None)

    results = []
    history = []
    for i, window in enumerate(online_clips(
        video, video.stem, graph.task_id, duration_sec=99999,
        annotations=[], clip_len_sec=clip_len, stride_sec=stride,
        fps=fps, load_frames=True,
    )):
        if window.frames is None:
            break
        step = graph.steps[min(i, len(graph.steps) - 1)]
        next_steps = graph.valid_next_steps(step.id)
        ctx = {
            "task_name": graph.task_name,
            "current_step_id": step.id,
            "current_step_desc": step.description,
            "valid_next_steps": str([s.description for s in next_steps]),
            "required_objects": str(step.required_object_ids),
            "expected_states": str(step.produced_state_ids),
            "valid_variations": str(step.valid_variations),
        }
        pred = vlm.predict(window.frames, ctx, _summarise_history(history[-3:]))
        history.append(pred)
        results.append({
            "start_sec": window.start_sec,
            "end_sec": window.end_sec,
            "deviation": pred.deviation,
            "mistake_class": pred.mistake_class,
            "confidence": pred.confidence,
            "uncertainty_reason": pred.uncertainty_reason,
            "rationale": pred.rationale,
        })
        score_text = f"{pred.confidence:.2f}" if pred.confidence is not None else pred.score_status
        typer.echo(f"[{window.start_sec:.1f}s] deviation={pred.deviation} class={pred.mistake_class} conf={score_text}")

    if out:
        out.write_text(json.dumps(results, indent=2))
        typer.echo(f"Saved to {out}")


@app.command("eval")
def eval_cmd(
    config: Path = typer.Argument(..., help="Hydra config YAML"),
    split: str = typer.Option("val"),
    tau: float = typer.Option(0.5, help="fixed confidence threshold"),
    dataset_name: str = typer.Option(
        "captaincook4d", "--dataset", help="captaincook4d | cc4d"
    ),
) -> None:
    """Run online (causal) evaluation on a CaptainCook4D split."""
    from omegaconf import OmegaConf
    from vmd.data import get_dataset
    from vmd.graph.schema import ProceduralGraph
    from vmd.vlm.qwen25vl import Qwen25VL
    from vmd.eval.online import run_online_eval
    from vmd.eval.report import print_results

    cfg = OmegaConf.load(config)
    raw_dir = Path(cfg.dataset.raw_dir)
    graphs_dir = Path(cfg.graph_cache_dir)

    dataset = get_dataset(dataset_name, raw_dir, split)
    graphs = {
        p.stem: ProceduralGraph.model_validate_json(p.read_text())
        for p in graphs_dir.glob("*.json")
    }
    vlm = Qwen25VL(cfg.vlm.model_id)

    results = {"online": run_online_eval(dataset.clips, graphs, vlm, fixed_tau=tau)}
    print_results(results)


@app.command("export-cc4d-graphs")
def export_cc4d_graphs_cmd(
    annotations_root: Path = typer.Option(
        Path("data/raw/captaincook4d/annotations"),
        help="CaptainCook4D annotations dir (contains task_graphs/)",
    ),
    cache_dir: Path = typer.Option(Path("data/graphs")),
) -> None:
    """Convert CaptainCook4D gold task graphs to ProceduralGraph cache."""
    from vmd.data.captaincook4d_graphs import export_cc4d_graphs
    written = export_cc4d_graphs(annotations_root, cache_dir)
    typer.echo(f"Wrote {len(written)} graphs to {cache_dir}: {', '.join(written)}")


@app.command("train")
def train_cmd(
    model_id: str = typer.Option("Qwen/Qwen2.5-VL-7B-Instruct"),
    raw_dir: Path = typer.Option(Path("data/raw/captaincook4d")),
    graphs_dir: Path = typer.Option(Path("data/graphs")),
    output_dir: Path = typer.Option(Path("outputs/lora_adapters")),
    epochs: int = typer.Option(3),
    lr: float = typer.Option(2e-4),
    dataset_name: str = typer.Option("captaincook4d", "--dataset"),
) -> None:
    """Fine-tune Qwen2.5-VL-7B with LoRA on CaptainCook4D."""
    from vmd.train.lora_train import train
    train(
        model_id=model_id,
        raw_dir=str(raw_dir),
        graphs_dir=str(graphs_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        lr=lr,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    app()
