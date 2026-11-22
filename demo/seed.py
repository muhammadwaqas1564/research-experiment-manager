"""Seed a registry with a small, realistic starting state for the demo and the UI."""
from __future__ import annotations

from core.config import Settings
from core.registry import Registry
from demo.simulator import write_run_outputs

BASE_PARAMS = {"C": 1.0, "threshold": 0.5, "class_weight": "none"}


def seed_demo(registry: Registry, settings: Settings) -> None:
    if registry.list():
        return
    settings.ensure_dirs()
    with registry.transaction():
        registry.create_experiment(
            "Logistic regression baseline", exp_id="EXP-001", model_type="logistic_regression",
            hypothesis="Reference model with default settings.", params=BASE_PARAMS,
            status="completed", is_baseline=True,
        )
        res, pred = write_run_outputs(settings.results_dir, "EXP-001", BASE_PARAMS)
        registry.set_artifacts("EXP-001", result_path=res, predictions_path=pred)

        p2 = {**BASE_PARAMS, "C": 2.0}
        registry.create_experiment(
            "Weaker regularisation (C=2)", exp_id="EXP-002", model_type="logistic_regression",
            hypothesis="Less regularisation should improve separation.", params=p2,
            status="completed", parent_id="EXP-001",
        )
        # The evaluation script crashed before writing F1 — predictions were saved though.
        res, pred = write_run_outputs(settings.results_dir, "EXP-002", p2, omit_metrics=("f1",))
        registry.set_artifacts("EXP-002", result_path=res, predictions_path=pred)

        registry.create_experiment(
            "Stronger regularisation (C=0.25)", exp_id="EXP-003", model_type="logistic_regression",
            hypothesis="Check whether the baseline is overfitting.", params={**BASE_PARAMS, "C": 0.25},
            status="running", parent_id="EXP-001",
        )
