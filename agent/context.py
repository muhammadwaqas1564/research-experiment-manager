"""Builds compact, relevant context for the model and verifies cited numbers against the DB."""
from __future__ import annotations

import re
from typing import Any, Iterable

from core import analysis
from core.paths import PathValidationError, safe_resolve
from core.registry import Experiment, Registry
from core.config import Settings

EXP_MENTION = re.compile(r"\bEXP-\d{3,}\b", re.IGNORECASE)
MAX_EXPERIMENTS_IN_CONTEXT = 12
METRIC_TOLERANCE = 5e-4


def mentioned_ids(text: str) -> list[str]:
    return sorted({m.upper() for m in EXP_MENTION.findall(text or "")})


def _artifact_exists(settings: Settings, rel: str | None) -> bool:
    if not rel:
        return False
    try:
        return safe_resolve(settings.results_dir, rel).exists()
    except PathValidationError:
        return False


def file_metrics_preview(settings: Settings, exp: Experiment) -> dict[str, float]:
    """Read-only parse of the result file (so context reflects metrics not yet recorded)."""
    if not _artifact_exists(settings, exp.result_path):
        return {}
    try:
        return analysis.parse_result_csv(safe_resolve(settings.results_dir, exp.result_path))
    except analysis.AnalysisError:
        return {}


def experiment_card(settings: Settings, exp: Experiment, baseline: Experiment | None) -> dict[str, Any]:
    raw = {k: v for k, v in {**file_metrics_preview(settings, exp), **exp.metrics}.items()
           if k in analysis.HIGHER_IS_BETTER}
    known = {k: round(v, 4) for k, v in raw.items()}
    card: dict[str, Any] = {
        "id": exp.id, "name": exp.name[:80], "status": exp.status, "model_type": exp.model_type,
        "params": exp.params, "metrics": known, "missing_metrics": analysis.missing_metrics(known),
        "has_predictions": _artifact_exists(settings, exp.predictions_path),
        "has_result_file": _artifact_exists(settings, exp.result_path),
    }
    if exp.is_baseline:
        card["is_baseline"] = True
    if exp.parent_id:
        card["parent_id"] = exp.parent_id
    if exp.hypothesis:
        card["hypothesis"] = exp.hypothesis[:200]
    p = analysis.PRIMARY_METRIC
    if baseline and not exp.is_baseline and p in known and p in baseline.metrics:
        card[f"{p}_delta_vs_baseline"] = round(raw[p] - baseline.metrics[p], 4)
    return card


def build_context(registry: Registry, settings: Settings, focus: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Only the relevant records: mentioned IDs + baseline, else the most recent few."""
    exps = registry.list()
    baseline = registry.baseline()
    focus = set(focus)
    if focus:
        chosen = [e for e in exps if e.id in focus or e.is_baseline or e.id in {x.parent_id for x in exps if x.id in focus}]
    else:
        chosen = exps[-MAX_EXPERIMENTS_IN_CONTEXT:]
        if baseline and baseline not in chosen:
            chosen = [baseline] + chosen[1:]
    return [experiment_card(settings, e, baseline) for e in chosen]


def verify_cited_metrics(registry: Registry, cited: Iterable[Any]) -> list[str]:
    """Every number the model cites must match the registry. Returns a list of problems."""
    problems = []
    for c in cited:
        if not registry.exists(c.experiment_id):
            problems.append(f"{c.experiment_id} is not a registered experiment")
            continue
        actual = registry.metrics(c.experiment_id).get(c.metric.lower())
        if actual is None:
            problems.append(f"{c.experiment_id}.{c.metric} is not recorded, but the model cited {c.value}")
        elif abs(actual - c.value) > METRIC_TOLERANCE:
            problems.append(f"{c.experiment_id}.{c.metric}: model said {c.value}, registry has {actual:.4f}")
    return problems


# A decimal number, including one that ends a sentence ("... F1 of 0.97.").
_DECIMAL = re.compile(r"(?<![\w.])[-+]?\d*\.\d+(?!\d)")


def unsupported_numbers(text: str, context: list[dict[str, Any]]) -> list[str]:
    """Decimal numbers in free text that don't correspond to any value in the context."""
    known: set[float] = set()

    def walk(x: Any) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            known.add(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(context)
    bad = []
    for tok in _DECIMAL.findall(text or ""):
        v = float(tok)
        decimals = len(tok.split(".")[1])
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        # allow percentages of known fractions (e.g. 0.8123 -> 81.23)
        if not any(abs(v - k) <= tol or abs(v - 100 * k) <= tol for k in known):
            bad.append(tok)
    return bad
