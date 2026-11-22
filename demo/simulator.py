"""Deterministic stand-in for a real training run.

It produces binary predictions whose quality depends on the experiment parameters, on a
fixed evaluation split, so the demo is reproducible and needs no ML dependencies.
Replace this with your real training/evaluation job.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import analysis

N_EVAL = 600
POSITIVE_RATE = 0.35


def _eval_split() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    y = (rng.random(N_EVAL) < POSITIVE_RATE).astype(int)
    noise = rng.normal(0.0, 1.0, N_EVAL)
    return y, noise


def simulate_predictions(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    y, noise = _eval_split()
    c = float(params.get("C", 1.0))
    quality = 2.0 - 0.35 * abs(math.log2(max(c, 1e-6) / 2.0))  # best near C=2
    shift = 0.35 if params.get("class_weight") == "balanced" else 0.0
    score = y * quality + noise + shift - 1.4
    prob = 1.0 / (1.0 + np.exp(-score))
    y_pred = (prob >= float(params.get("threshold", 0.5))).astype(int)
    return y, y_pred


def write_run_outputs(results_dir: Path, exp_id: str, params: dict[str, Any],
                      omit_metrics: tuple[str, ...] = ()) -> tuple[str, str]:
    """Write predictions and a summary CSV. Returns paths relative to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    y, y_pred = simulate_predictions(params)
    pred_name, res_name = f"{exp_id}_predictions.csv", f"{exp_id}_results.csv"
    pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(results_dir / pred_name, index=False)
    metrics = analysis.compute_classification_metrics(y, y_pred)
    rows = [{"metric": k, "value": round(v, 6)} for k, v in metrics.items() if k not in omit_metrics]
    pd.DataFrame(rows).to_csv(results_dir / res_name, index=False)
    return res_name, pred_name
