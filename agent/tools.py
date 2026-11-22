"""Allowlisted tool registry.

The LLM can only *propose* actions by name. Each action maps to a deterministic Python
handler here with a Pydantic argument schema. Nothing outside this registry can run:
no eval, no shell, no arbitrary file access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from core import analysis
from core.config import Settings
from core.paths import PathValidationError, safe_resolve
from core.registry import EXPERIMENT_ID_RE, Registry, RegistryError
from core.tasks import write_task_file


@dataclass
class ToolContext:
    registry: Registry
    settings: Settings


@dataclass
class ToolResult:
    tool: str
    ok: bool
    message: str
    experiment_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ToolArgsError(ValueError):
    pass


# --------------------------------------------------------------------------- argument schemas
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpArgs(_Args):
    experiment_id: str

    @field_validator("experiment_id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not EXPERIMENT_ID_RE.match(v):
            raise ValueError("experiment_id must look like EXP-001")
        return v


ParamValue = Union[int, float, str, bool]


class ProposeArgs(_Args):
    parent_id: str
    name: str = Field(min_length=3, max_length=120)
    hypothesis: str = Field(min_length=3, max_length=500)
    params: dict[str, ParamValue] = Field(default_factory=dict, max_length=20)

    @field_validator("parent_id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not EXPERIMENT_ID_RE.match(v):
            raise ValueError("parent_id must look like EXP-001")
        return v


class TaskArgs(ExpArgs):
    title: str = Field(min_length=3, max_length=120)
    instructions: str = Field(min_length=3, max_length=2000)


class StatusArgs(ExpArgs):
    new_status: str
    note: str = Field(default="", max_length=300)


# --------------------------------------------------------------------------- helpers
def _artifact(ctx: ToolContext, rel: str | None) -> Path | None:
    if not rel:
        return None
    return safe_resolve(ctx.settings.results_dir, rel)


def _record_file_metrics(ctx: ToolContext, exp_id: str) -> tuple[dict[str, float], list[str]]:
    exp = ctx.registry.get(exp_id)
    path = _artifact(ctx, exp.result_path)
    if path is None or not path.exists():
        return {}, ["No result file is registered for this experiment"]
    found = analysis.parse_result_csv(path)
    known = {k: v for k, v in found.items() if k in analysis.HIGHER_IS_BETTER}
    if known:
        ctx.registry.record_metrics(exp_id, known, source="file")
    return known, []


# --------------------------------------------------------------------------- handlers
def inspect_artifacts(ctx: ToolContext, a: ExpArgs) -> ToolResult:
    exp = ctx.registry.get(a.experiment_id)
    status = {}
    for label, rel in (("result_file", exp.result_path), ("predictions_file", exp.predictions_path)):
        p = _artifact(ctx, rel)
        status[label] = {"registered": bool(rel), "exists": bool(p and p.exists())}
    return ToolResult("inspect_artifacts", True, "Artifacts inspected", a.experiment_id, status)


def validate_metrics(ctx: ToolContext, a: ExpArgs) -> ToolResult:
    found, warnings = _record_file_metrics(ctx, a.experiment_id)
    all_metrics = ctx.registry.metrics(a.experiment_id)
    missing = analysis.missing_metrics(all_metrics)
    msg = "All required metrics present" if not missing else f"Missing metrics: {', '.join(missing)}"
    return ToolResult("validate_metrics", True, msg, a.experiment_id,
                      {"from_file": found, "missing": missing}, warnings)


def calculate_missing_metric(ctx: ToolContext, a: ExpArgs) -> ToolResult:
    exp = ctx.registry.get(a.experiment_id)
    file_metrics, warnings = _record_file_metrics(ctx, a.experiment_id)
    current = ctx.registry.metrics(a.experiment_id)
    missing = analysis.missing_metrics(current)
    if not missing:
        return ToolResult("calculate_missing_metric", True, "Nothing to calculate; no metrics missing",
                          a.experiment_id, {"computed": {}}, warnings)
    pred = _artifact(ctx, exp.predictions_path)
    if pred is None or not pred.exists():
        return ToolResult("calculate_missing_metric", False,
                          f"Cannot compute {', '.join(missing)}: no saved predictions available",
                          a.experiment_id, {"missing": missing}, warnings)
    y_true, y_pred = analysis.parse_predictions_csv(pred)
    computed = analysis.compute_classification_metrics(y_true, y_pred)
    # Cross-check values that the result file did report.
    for name, v in file_metrics.items():
        if name in computed and abs(computed[name] - v) > 1e-3:
            warnings.append(f"{name}: result file says {v:.4f} but predictions give {computed[name]:.4f}")
    to_record = {m: computed[m] for m in missing if m in computed}
    ctx.registry.record_metrics(a.experiment_id, to_record, source="computed")
    return ToolResult("calculate_missing_metric", True,
                      "Computed " + ", ".join(f"{k}={v:.4f}" for k, v in to_record.items()),
                      a.experiment_id, {"computed": to_record, "n_rows": int(y_true.size)}, warnings)


def _mark_analyzed_if_complete(ctx: ToolContext, exp_id: str) -> bool:
    exp = ctx.registry.get(exp_id)
    if exp.status == "completed" and not analysis.missing_metrics(exp.metrics):
        ctx.registry.transition(exp_id, "analyzed", "metrics complete")
        return True
    return False


def compare_to_baseline(ctx: ToolContext, a: ExpArgs) -> ToolResult:
    base = ctx.registry.baseline()
    if base is None:
        return ToolResult("compare_to_baseline", False, "No baseline experiment is registered", a.experiment_id)
    # Make sure both sides reflect their result files before comparing.
    for exp_id in {base.id, a.experiment_id}:
        if analysis.missing_metrics(ctx.registry.metrics(exp_id)):
            _record_file_metrics(ctx, exp_id)
    _mark_analyzed_if_complete(ctx, base.id)
    base = ctx.registry.get(base.id)
    exp = ctx.registry.get(a.experiment_id)
    if base.id == exp.id:
        return ToolResult("compare_to_baseline", True, "This experiment is the baseline", a.experiment_id,
                          {"baseline_id": base.id, "deltas": []})
    deltas = analysis.compare_to_baseline(exp.metrics, base.metrics)
    data = {"baseline_id": base.id,
            "deltas": [d.__dict__ for d in deltas],
            "missing": analysis.missing_metrics(exp.metrics)}
    # Deterministic rule: a completed experiment with complete metrics becomes 'analyzed'.
    if _mark_analyzed_if_complete(ctx, exp.id):
        data["status_change"] = "completed -> analyzed"
    primary = next((d for d in deltas if d.metric == analysis.PRIMARY_METRIC), None)
    msg = (f"{analysis.PRIMARY_METRIC} {primary.candidate:.4f} vs baseline {primary.baseline:.4f} "
           f"({primary.delta:+.4f})") if primary else "No shared primary metric with baseline"
    return ToolResult("compare_to_baseline", True, msg, exp.id, data)


def summarize_experiment(ctx: ToolContext, a: ExpArgs) -> ToolResult:
    exp = ctx.registry.get(a.experiment_id)
    return ToolResult("summarize_experiment", True, f"{exp.id} is {exp.status}", exp.id, exp.to_dict())


def propose_experiment(ctx: ToolContext, a: ProposeArgs) -> ToolResult:
    parent = ctx.registry.get(a.parent_id)
    params = {**parent.params, **a.params}
    with ctx.registry.transaction():
        exp = ctx.registry.create_experiment(
            a.name, hypothesis=a.hypothesis, model_type=parent.model_type, params=params,
            parent_id=parent.id, status="planned",
        )
        path = write_task_file(ctx.settings.tasks_dir, exp.id, a.name, a.hypothesis, params)
    return ToolResult("propose_experiment", True, f"Registered {exp.id} (planned)", exp.id,
                      {"task_file": path.name, "params": params})


def create_task_file(ctx: ToolContext, a: TaskArgs) -> ToolResult:
    exp = ctx.registry.get(a.experiment_id)
    path = write_task_file(ctx.settings.tasks_dir, exp.id, a.title, a.instructions, exp.params)
    return ToolResult("create_task_file", True, f"Wrote {path.name}", exp.id, {"task_file": path.name})


def update_status(ctx: ToolContext, a: StatusArgs) -> ToolResult:
    exp = ctx.registry.transition(a.experiment_id, a.new_status, a.note)
    return ToolResult("update_status", True, f"{exp.id} is now {exp.status}", exp.id)


def no_action(ctx: ToolContext, a: _Args) -> ToolResult:
    return ToolResult("no_action", True, "No action needed")


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], ToolResult]
    requires_approval: bool
    mutates: bool


TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("inspect_artifacts", "Check which result/prediction files exist for an experiment.",
             ExpArgs, inspect_artifacts, False, False),
    ToolSpec("validate_metrics", "Parse the result CSV, record metrics, list missing required metrics.",
             ExpArgs, validate_metrics, False, True),
    ToolSpec("calculate_missing_metric", "Compute missing metrics locally from saved predictions.",
             ExpArgs, calculate_missing_metric, False, True),
    ToolSpec("compare_to_baseline", "Compare an experiment's metrics to the baseline; marks it analyzed when complete.",
             ExpArgs, compare_to_baseline, False, True),
    ToolSpec("summarize_experiment", "Return the registry record for an experiment.",
             ExpArgs, summarize_experiment, False, False),
    ToolSpec("propose_experiment", "Register a new planned experiment derived from parent_id and write its task file.",
             ProposeArgs, propose_experiment, True, True),
    ToolSpec("create_task_file", "Write a Markdown task file for an existing experiment.",
             TaskArgs, create_task_file, False, True),
    ToolSpec("update_status", "Change an experiment's status (validated state machine).",
             StatusArgs, update_status, True, True),
    ToolSpec("no_action", "Use when nothing should be done.", _Args, no_action, False, False),
]}


def build_args(tool: str, experiment_id: str | None, arguments: dict[str, Any]) -> BaseModel:
    spec = TOOLS.get(tool)
    if spec is None:
        raise ToolArgsError(f"Unsupported action: {tool}")
    payload = dict(arguments or {})
    fields = spec.args_model.model_fields
    if experiment_id and "experiment_id" in fields:
        payload.setdefault("experiment_id", experiment_id)
    if experiment_id and "parent_id" in fields:
        payload.setdefault("parent_id", experiment_id)
    try:
        return spec.args_model.model_validate(payload)
    except ValidationError as exc:
        raise ToolArgsError(f"Invalid arguments for {tool}: {exc.errors()[0]['msg']}") from exc


def run_tool(ctx: ToolContext, tool: str, args: BaseModel) -> ToolResult:
    spec = TOOLS[tool]
    try:
        return spec.handler(ctx, args)
    except (RegistryError, analysis.AnalysisError, PathValidationError) as exc:
        exp_id = getattr(args, "experiment_id", None) or getattr(args, "parent_id", None)
        return ToolResult(tool, False, str(exc), exp_id)


def tool_catalog_text() -> str:
    lines = []
    for t in TOOLS.values():
        extra = [k for k in t.args_model.model_fields if k not in ("experiment_id",)]
        sig = f" arguments: {', '.join(extra)}" if extra else ""
        lines.append(f"- {t.name}: {t.description}{sig}" + (" [requires approval]" if t.requires_approval else ""))
    return "\n".join(lines)
