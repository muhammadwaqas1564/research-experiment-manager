"""Deterministic analysis: CSV parsing, metric calculation, validation and baseline comparison.

No LLM is involved anywhere in this module. All numbers shown to the user come from here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")
HIGHER_IS_BETTER: dict[str, bool] = {"accuracy": True, "precision": True, "recall": True, "f1": True, "loss": False}
PRIMARY_METRIC = "f1"
# Hard cap so a huge file can't stall the app (and is never shipped to an LLM).
MAX_CSV_BYTES = 50 * 1024 * 1024


class AnalysisError(ValueError):
    pass


def _check_size(path: Path) -> None:
    if not path.exists():
        raise AnalysisError(f"File not found: {path.name}")
    if path.stat().st_size > MAX_CSV_BYTES:
        raise AnalysisError(f"{path.name} exceeds the {MAX_CSV_BYTES // 1_048_576} MB limit")


def parse_result_csv(path: Path | str) -> dict[str, float]:
    """Parse a summary result file with columns ``metric,value``.

    Blank, non-numeric or non-finite values are treated as missing (not as zero).
    """
    path = Path(path)
    _check_size(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas raises several types
        raise AnalysisError(f"Could not parse {path.name}: {exc}") from exc
    cols = {c.strip().lower(): c for c in df.columns}
    if "metric" not in cols or "value" not in cols:
        raise AnalysisError(f"{path.name} must have 'metric' and 'value' columns")
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row[cols["metric"]]).strip().lower()
        val = pd.to_numeric(row[cols["value"]], errors="coerce")
        if name and name != "nan" and pd.notna(val) and math.isfinite(float(val)):
            out[name] = float(val)
    return out


def parse_predictions_csv(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Parse saved predictions with columns ``y_true,y_pred`` (binary 0/1 labels)."""
    path = Path(path)
    _check_size(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise AnalysisError(f"Could not parse {path.name}: {exc}") from exc
    cols = {c.strip().lower(): c for c in df.columns}
    if "y_true" not in cols or "y_pred" not in cols:
        raise AnalysisError(f"{path.name} must have 'y_true' and 'y_pred' columns")
    df = df[[cols["y_true"], cols["y_pred"]]].dropna()
    if df.empty:
        raise AnalysisError(f"{path.name} contains no usable rows")
    y_true = df.iloc[:, 0].astype(int).to_numpy()
    y_pred = df.iloc[:, 1].astype(int).to_numpy()
    if not set(np.unique(np.concatenate([y_true, y_pred]))) <= {0, 1}:
        raise AnalysisError(f"{path.name}: only binary 0/1 labels are supported")
    return y_true, y_pred


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Binary classification metrics, positive class = 1. Zero-division yields 0.0."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise AnalysisError("y_true and y_pred must be non-empty and the same length")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = float(np.mean(y_true == y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def missing_metrics(found: dict[str, float], required: tuple[str, ...] = REQUIRED_METRICS) -> list[str]:
    return [m for m in required if m not in found]


@dataclass
class MetricDelta:
    metric: str
    candidate: float
    baseline: float
    delta: float
    improved: bool


def compare_to_baseline(candidate: dict[str, float], baseline: dict[str, float]) -> list[MetricDelta]:
    """Compare shared metrics. 'improved' respects each metric's direction."""
    out: list[MetricDelta] = []
    for name in sorted(set(candidate) & set(baseline)):
        delta = candidate[name] - baseline[name]
        better = HIGHER_IS_BETTER.get(name, True)
        out.append(MetricDelta(name, candidate[name], baseline[name], delta, delta > 0 if better else delta < 0))
    return out


def validate_metric_values(values: dict[str, float]) -> list[str]:
    """Sanity checks used in final verification. Returns a list of problems."""
    problems = []
    for name, v in values.items():
        if not math.isfinite(v):
            problems.append(f"{name} is not finite")
        elif name in REQUIRED_METRICS and not 0.0 <= v <= 1.0:
            problems.append(f"{name}={v} is outside [0, 1]")
    return problems
