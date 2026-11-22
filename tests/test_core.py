"""Deterministic core: CSV parsing, metrics, validation, baseline comparison, status machine, paths, cache."""
from __future__ import annotations

import numpy as np
import pytest

from core import analysis
from core.paths import PathValidationError, safe_filename, safe_resolve
from core.registry import InvalidTransition, Registry, RegistryError
from llm.provider import CachedProvider, ChatMessage, ResponseCache


def test_metrics_match_hand_calculation():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    m = analysis.compute_classification_metrics(y_true, y_pred)
    # tp=3 fp=1 fn=1 tn=3
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(0.75)


def test_zero_division_gives_zero_not_nan():
    m = analysis.compute_classification_metrics(np.array([0, 0]), np.array([0, 0]))
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_result_csv_blank_and_text_values_are_missing_not_zero(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("metric,value\naccuracy,0.9\nprecision,\nrecall,n/a\nf1,0.8\n")
    found = analysis.parse_result_csv(p)
    assert found == {"accuracy": 0.9, "f1": 0.8}
    assert analysis.missing_metrics(found) == ["precision", "recall"]


def test_bad_csvs_are_rejected(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(analysis.AnalysisError):
        analysis.parse_result_csv(p)
    with pytest.raises(analysis.AnalysisError):
        analysis.parse_predictions_csv(p)
    q = tmp_path / "multi.csv"
    q.write_text("y_true,y_pred\n0,2\n")
    with pytest.raises(analysis.AnalysisError):
        analysis.parse_predictions_csv(q)


def test_baseline_comparison_respects_direction():
    deltas = {d.metric: d for d in analysis.compare_to_baseline(
        {"f1": 0.8, "loss": 0.3}, {"f1": 0.7, "loss": 0.4})}
    assert deltas["f1"].improved and deltas["f1"].delta == pytest.approx(0.1)
    assert deltas["loss"].improved  # lower loss is better


def test_status_machine(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    e = reg.create_experiment("x")
    with pytest.raises(InvalidTransition):
        reg.transition(e.id, "analyzed")
    reg.transition(e.id, "running")
    reg.transition(e.id, "completed")
    assert reg.get(e.id).status == "completed"
    assert [ev["kind"] for ev in reg.events(e.id)] == ["created", "status", "status"]


def test_registry_rejects_non_finite_metrics_atomically(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    e = reg.create_experiment("x")
    with pytest.raises(RegistryError):
        reg.record_metrics(e.id, {"accuracy": 0.9, "f1": float("nan")}, "file")
    assert reg.metrics(e.id) == {}  # the valid one was rolled back too


def test_path_validation(tmp_path):
    assert safe_resolve(tmp_path, "a/b.csv") == (tmp_path / "a/b.csv").resolve()
    for bad in ("../x.csv", "/etc/passwd", "a/../../x"):
        with pytest.raises(PathValidationError):
            safe_resolve(tmp_path, bad)
    for bad in (".env", "a/b", "", "x" * 200):
        with pytest.raises(PathValidationError):
            safe_filename(bad)


def test_path_validation_rejects_symlink_escape(tmp_path):
    try:
        (tmp_path / "link").symlink_to(tmp_path.anchor, target_is_directory=True)
    except NotImplementedError:
        pytest.skip("Symlinks are not supported on this platform")
    except OSError as err:
        if getattr(err, "winerror", None) != 1314:
            raise
        pytest.skip("Creating symlinks requires Windows Developer Mode or elevated privileges")

    with pytest.raises(PathValidationError):
        safe_resolve(tmp_path, "link/etc/passwd")


def test_cache_avoids_repeat_model_calls(tmp_path, scripted):
    inner = scripted('{"x": 1}')
    cached = CachedProvider(inner, ResponseCache(tmp_path / "c.sqlite"))
    msgs = [ChatMessage("user", "same question")]
    first = cached.chat(msgs, json_mode=True)
    second = cached.chat(msgs, json_mode=True)
    assert len(inner.prompts) == 1
    assert not first.cached and second.cached and second.text == first.text
    cached.chat([ChatMessage("user", "different")])
    assert len(inner.prompts) == 2
