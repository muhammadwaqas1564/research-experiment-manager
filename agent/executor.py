"""Executor: runs validated plan steps through the allowlisted tools, then verifies the result.

The executor never talks to the LLM. Approval-gated steps are skipped unless approved.
Final verification is deterministic and re-derives computed metrics from source files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from agent.planner import Plan, ValidatedStep
from agent.tools import ToolContext, ToolResult, run_tool
from core import analysis
from core.paths import PathValidationError, safe_resolve
from core.registry import STATUSES, Registry, UnknownExperiment
from core.config import Settings


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ExecutionReport:
    results: list[ToolResult] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def ok(self) -> bool:
        return self.verified and all(r.ok for r in self.results)


class Executor:
    def __init__(self, registry: Registry, settings: Settings):
        self.registry = registry
        self.settings = settings
        self.ctx = ToolContext(registry, settings)

    def execute(self, plan: Plan, approved: Iterable[int] | bool = ()) -> ExecutionReport:
        """``approved``: True approves every gated step; otherwise a set of step indices."""
        approve_all = approved is True
        approved_idx = set() if isinstance(approved, bool) else set(approved)
        report = ExecutionReport()
        for i, step in enumerate(plan.steps):
            if step.requires_approval and not (approve_all or i in approved_idx):
                report.skipped.append({"index": i, "action": step.action, "experiment_id": step.experiment_id,
                                       "reason": "awaiting researcher approval"})
                continue
            report.results.append(run_tool(self.ctx, step.action, step.args))
        report.checks = self.verify(report.results, plan.steps)
        return report

    def run_steps(self, steps: list[ValidatedStep], approved: Iterable[int] | bool = ()) -> ExecutionReport:
        return self.execute(Plan("", "", steps, "direct"), approved)

    # ------------------------------------------------------------------ verification
    def verify(self, results: list[ToolResult], steps: list[ValidatedStep] | None = None) -> list[Check]:
        checks: list[Check] = []
        touched = {r.experiment_id for r in results if r.experiment_id}
        for exp_id in sorted(touched):
            try:
                exp = self.registry.get(exp_id)
            except UnknownExperiment:
                checks.append(Check(f"{exp_id} exists", False, "referenced experiment not in registry"))
                continue
            checks.append(Check(f"{exp_id} status valid", exp.status in STATUSES, exp.status))
            problems = analysis.validate_metric_values(exp.metrics)
            checks.append(Check(f"{exp_id} metric values sane", not problems, "; ".join(problems)))
            if exp.status == "analyzed":
                missing = analysis.missing_metrics(exp.metrics)
                checks.append(Check(f"{exp_id} analyzed with complete metrics", not missing,
                                    f"missing {missing}" if missing else ""))
            checks.extend(self._recheck_computed(exp_id, exp.predictions_path))
        for r in results:
            task = r.data.get("task_file") if r.ok else None
            if task:
                try:
                    exists = safe_resolve(self.settings.tasks_dir, task).exists()
                except PathValidationError:
                    exists = False
                checks.append(Check(f"task file {task} written", exists))
        return checks

    def _recheck_computed(self, exp_id: str, predictions_rel: str | None) -> list[Check]:
        sources = self.registry.metric_sources(exp_id)
        computed = [m for m, s in sources.items() if s == "computed"]
        if not computed:
            return []
        try:
            path = safe_resolve(self.settings.results_dir, predictions_rel or "")
            y_true, y_pred = analysis.parse_predictions_csv(path)
        except (PathValidationError, analysis.AnalysisError) as exc:
            return [Check(f"{exp_id} computed metrics reproducible", False, str(exc))]
        fresh = analysis.compute_classification_metrics(y_true, y_pred)
        stored = self.registry.metrics(exp_id)
        bad = [m for m in computed if not abs(fresh.get(m, float("nan")) - stored[m]) <= 1e-9]
        return [Check(f"{exp_id} computed metrics reproducible", not bad, f"mismatch: {bad}" if bad else "")]
