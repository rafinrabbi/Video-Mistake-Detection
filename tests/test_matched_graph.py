from dataclasses import asdict, replace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from vmd.eval.matched_graph import control_space, replay_clip, weighted_ap, weighted_ap_preparation
from vmd.eval import matched_graph
from vmd.tracking.belief import BeliefFilter, FilterConfig, StepSpace


def space():
    # Branching topology, including a root that is not the first menu entry.
    return StepSpace("toy", ["a", "b", "c"], [0, 1, 2], ["a", "b", "c"],
                     [[2], [], []], [[], [], [0]])


def test_controls_replace_both_initial_prior_and_edges_without_mutating_dag():
    s = space()
    before = asdict(s)
    linear = control_space(s, "linear")
    uniform = control_space(s, "uniform")
    assert linear.roots == [0]
    assert linear.pred == [[], [0], [1]]
    assert uniform.roots == [0, 1, 2]
    f = BeliefFilter(uniform, FilterConfig(mode="uniform"))
    assert np.allclose(f.belief[:3], f.belief[0])
    assert asdict(s) == before


def test_replay_updates_current_observation_and_rejects_misaligned_sources():
    row = dict(clip_id="x", start_sec=0., end_sec=4., true_label=0,
               true_class="no_mistake", task_id="toy", confidence=.2,
               step_probs=[0., 0., 1., 0.], gold_step_idx_strict=2)
    out = replay_clip([row], [row], space(), 20., "uniform")
    assert out[0]["step_posterior_idx"] == 2
    assert "step_posterior_idx" not in row
    with pytest.raises(ValueError, match="align"):
        replay_clip([row], [dict(row, end_sec=8.)], space(), 20., "dag")
    with pytest.raises(ValueError, match="emission"):
        replay_clip([dict(row, step_probs=[float("nan")] * 4)], [row], space(), 20., "dag")


def test_bootstrap_ap_handles_ties_and_zero_weight_recordings():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([.9, .9, .5, .5, .1, .0])
    for w in [np.ones(6), np.array([0, 0, 2, 2, 1, 1])]:
        assert weighted_ap(weighted_ap_preparation(y, p), w) == pytest.approx(
            average_precision_score(y, p, sample_weight=w))


def test_uniform_replay_does_not_read_original_dependency_edges():
    s = space()
    changed = replace(s, succ=[[], [0], [1]], pred=[[1], [2], []])
    rows = [dict(clip_id="x", start_sec=i * 4., end_sec=(i + 1) * 4.,
                 true_label=0, true_class="no_mistake", task_id="toy", confidence=.2,
                 step_probs=p, gold_step_idx_strict=0)
            for i, p in enumerate([[.1, .2, .6, .1], [.8, .1, .05, .05], [.1, .8, .05, .05]])]
    assert replay_clip(rows, rows, s, 20., "uniform") == replay_clip(rows, rows, changed, 20., "uniform")


def test_crossfit_excludes_entire_held_out_recordings(monkeypatch):
    def fit(rows, names):
        return {"training_ids": sorted({r["clip_id"] for r in rows})}

    def apply(rows, model):
        assert not ({r["clip_id"] for r in rows} & set(model["training_ids"]))
        return [dict(r) for r in rows]

    monkeypatch.setattr(matched_graph, "fit_fusion", fit)
    monkeypatch.setattr(matched_graph, "apply_fusion", apply)
    rows = [dict(clip_id=cid, start_sec=i, true_label=i % 2)
            for cid in ("a", "b", "c", "d") for i in range(4)]
    folds = {"a": 0, "b": 0, "c": 1, "d": 1}
    out, _ = matched_graph.crossfit_fusion(rows, ("content_logit",), folds)
    assert len(out) == len(rows)
    assert all(r["crossfit_fold"] == folds[r["clip_id"]] for r in out)
