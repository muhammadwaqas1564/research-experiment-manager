"""Planner: turns a researcher's request into a validated, allowlisted plan.

Flow: build compact context -> ask Groq GPT-OSS for JSON -> validate with Pydantic ->
check every action against the tool registry and every experiment ID against SQLite.
If the model is unavailable or its output is unusable, a deterministic planner is used.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from agent import prompts
from agent.context import build_context, file_metrics_preview, mentioned_ids
from agent.schemas import PlanOutput, PlanStep, StructuredOutputError, extract_json
from agent.tools import TOOLS, ToolArgsError, build_args, tool_catalog_text
from core import analysis
from core.config import Settings
from core.registry import Registry
from llm.provider import ChatMessage, LLMError, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ValidatedStep:
    action: str
    experiment_id: str | None
    args: BaseModel
    reason: str
    required_evidence: list[str] = field(default_factory=list)
    proposed_next_step: str = ""
    requires_approval: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "experiment_id": self.experiment_id, "reason": self.reason,
                "required_evidence": self.required_evidence, "proposed_next_step": self.proposed_next_step,
                "requires_approval": self.requires_approval, "arguments": self.args.model_dump()}


@dataclass
class Plan:
    request: str
    summary: str
    steps: list[ValidatedStep]
    source: str  # "llm" | "fallback"
    warnings: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    llm_metadata: dict[str, Any] = field(default_factory=dict)


class Planner:
    def __init__(self, registry: Registry, settings: Settings, provider: LLMProvider | None = None):
        self.registry = registry
        self.settings = settings
        self.provider = provider

    # ------------------------------------------------------------------ public
    def plan(self, request: str) -> Plan:
        request = (request or "").strip() or "Analyse the registered experiments."
        if self.provider is None:
            return self.fallback_plan(request, ["LLM unavailable: using the deterministic planner."])
        try:
            return self._llm_plan(request)
        except LLMError as exc:
            return self.fallback_plan(request, [f"Groq request failed: {exc}. Using the deterministic planner."])
        except (StructuredOutputError, ValidationError) as exc:
            msg = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            return self.fallback_plan(request, [f"Rejected malformed model output ({msg}). Using the deterministic planner."])

    # ------------------------------------------------------------------ LLM path
    def _llm_plan(self, request: str) -> Plan:
        context = build_context(self.registry, self.settings, mentioned_ids(request))
        messages = [
            ChatMessage("system", prompts.BASE_SYSTEM),
            ChatMessage("user", prompts.planner_prompt(tool_catalog_text(), context, request)),
        ]
        resp = self.provider.chat(messages, json_mode=True, temperature=0.1, max_tokens=1500)
        parsed = PlanOutput.model_validate(extract_json(resp.text))

        steps, rejected = [], []
        for i, raw in enumerate(parsed.steps):
            try:
                steps.append(self._validate_step(raw))
            except (ValidationError, ToolArgsError, ValueError) as exc:
                reason = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
                rejected.append({"index": i, "step": raw, "reason": reason})
        warnings = [f"Rejected step {r['index'] + 1}: {r['reason']}" for r in rejected]
        meta = resp.metadata()
        if not steps:
            plan = self.fallback_plan(request, warnings + ["No valid steps in model output. Using the deterministic planner."])
            plan.rejected, plan.llm_metadata = rejected, meta
            return plan
        return Plan(request, parsed.summary, steps, "llm", warnings, rejected, meta)

    def _validate_step(self, raw: dict[str, Any]) -> ValidatedStep:
        step = PlanStep.model_validate(raw)
        spec = TOOLS[step.action]
        if step.action != "no_action":
            ref = step.experiment_id or step.arguments.get("experiment_id") or step.arguments.get("parent_id")
            if not ref:
                raise ValueError(f"{step.action} requires an experiment_id")
            if not self.registry.exists(ref):
                raise ValueError(f"{ref} is not a registered experiment")
            step.experiment_id = ref
        args = build_args(step.action, step.experiment_id, step.arguments)
        return ValidatedStep(
            action=step.action, experiment_id=step.experiment_id, args=args, reason=step.reason,
            required_evidence=step.required_evidence, proposed_next_step=step.proposed_next_step,
            # The registry's approval rule always wins; the model can only make it stricter.
            requires_approval=spec.requires_approval or step.requires_approval,
        )

    # ------------------------------------------------------------------ deterministic path
    def fallback_plan(self, request: str, warnings: list[str] | None = None) -> Plan:
        ids = [i for i in mentioned_ids(request) if self.registry.exists(i)]
        targets = [self.registry.get(i) for i in ids] if ids else [
            e for e in self.registry.list() if e.status in ("completed", "analyzed") or e.is_baseline
        ]
        steps: list[ValidatedStep] = []

        def add(action: str, exp_id: str, reason: str, evidence: list[str], nxt: str) -> None:
            steps.append(ValidatedStep(action, exp_id, build_args(action, exp_id, {}), reason, evidence, nxt,
                                       TOOLS[action].requires_approval))

        for exp in targets:
            if exp.status not in ("completed", "analyzed") and not exp.is_baseline:
                add("summarize_experiment", exp.id, f"{exp.id} is {exp.status}; no results to analyse yet.",
                    [], "Wait for the run to complete")
                continue
            known = {**file_metrics_preview(self.settings, exp), **exp.metrics}
            missing = analysis.missing_metrics(known)
            add("validate_metrics", exp.id, "Record metrics from the result file and check completeness.",
                ["result_file"], "Compute anything missing")
            if missing:
                add("calculate_missing_metric", exp.id,
                    f"Missing {', '.join(missing)}; recompute locally from saved predictions.",
                    ["saved_predictions", "ground_truth_labels"], "Compare against the baseline")
            if not exp.is_baseline:
                add("compare_to_baseline", exp.id, "Quantify the change versus the baseline.",
                    ["baseline_metrics"], "Prioritise next experiments")
        if not steps:
            steps.append(ValidatedStep("no_action", None, build_args("no_action", None, {}),
                                       "No completed experiments need analysis.", [], "", False))
        summary = f"Deterministic analysis plan for {len(targets)} experiment(s)."
        return Plan(request, summary, steps, "fallback", warnings or [])
