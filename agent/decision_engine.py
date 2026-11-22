"""Decision engine: deterministic evidence + candidate generation, LLM explanation and ranking.

Numbers and candidates are always produced here in Python. Groq GPT-OSS may only:
  * explain what the evidence shows,
  * rank the existing candidate IDs,
  * cite metrics, which are checked against the registry.
Any invented candidate or mismatched number causes the model output to be discarded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent import prompts
from agent.context import verify_cited_metrics
from agent.planner import ValidatedStep
from agent.schemas import DecisionOutput, StructuredOutputError, extract_json
from agent.tools import TOOLS, build_args
from core import analysis
from core.registry import Experiment, Registry
from llm.provider import ChatMessage, LLMError, LLMProvider

logger = logging.getLogger(__name__)
P = analysis.PRIMARY_METRIC


@dataclass
class Candidate:
    candidate_id: str
    parent_id: str
    title: str
    hypothesis: str
    param_changes: dict[str, Any]
    score: float
    priority: int = 3
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Decision:
    summary: str
    evidence: dict[str, Any]
    candidates: list[Candidate]
    caveats: list[str]
    source: str  # "llm" | "fallback"
    warnings: list[str] = field(default_factory=list)
    llm_metadata: dict[str, Any] = field(default_factory=dict)

    def proposal_steps(self, top_n: int = 1) -> list[ValidatedStep]:
        """Turn the top candidates into approval-gated propose_experiment steps."""
        steps = []
        for c in self.candidates[:top_n]:
            args = build_args("propose_experiment", c.parent_id,
                              {"name": c.title, "hypothesis": c.hypothesis, "params": c.param_changes})
            steps.append(ValidatedStep("propose_experiment", c.parent_id, args, c.rationale or c.hypothesis,
                                       ["baseline_comparison"], "Run the new experiment",
                                       TOOLS["propose_experiment"].requires_approval))
        return steps


class DecisionEngine:
    def __init__(self, registry: Registry, provider: LLMProvider | None = None):
        self.registry = registry
        self.provider = provider

    # ------------------------------------------------------------------ evidence
    def evidence(self) -> dict[str, Any]:
        base = self.registry.baseline()
        exps = self.registry.list()
        scored = [e for e in exps if P in e.metrics and e.status in ("completed", "analyzed")]
        best = max(scored, key=lambda e: (e.metrics[P], e.id), default=None)
        rows = []
        for e in scored:
            row: dict[str, Any] = {"id": e.id, "status": e.status, "params": e.params,
                                   "metrics": {k: round(v, 4) for k, v in e.metrics.items()}}
            if base and P in base.metrics and not e.is_baseline:
                row[f"{P}_delta_vs_baseline"] = round(e.metrics[P] - base.metrics[P], 4)
            rows.append(row)
        return {
            "primary_metric": P,
            "baseline_id": base.id if base else None,
            "best_id": best.id if best else None,
            "experiments": rows,
            "incomplete": [e.id for e in exps if e.status in ("completed", "analyzed")
                           and analysis.missing_metrics(e.metrics)],
            "pending": [e.id for e in exps if e.status in ("planned", "running")],
        }

    def candidates(self) -> list[Candidate]:
        """Rule-based parameter moves from the best experiment so far."""
        scored = [e for e in self.registry.list() if P in e.metrics and e.status in ("completed", "analyzed")]
        if not scored:
            return []
        best = max(scored, key=lambda e: (e.metrics[P], e.id))
        parent = self.registry.get(best.parent_id) if best.parent_id else None
        m, prm = best.metrics, best.params
        tried = [e.params for e in self.registry.list()]
        out: list[Candidate] = []

        def add(key: str, title: str, hyp: str, changes: dict[str, Any], score: float) -> None:
            new_params = {**prm, **changes}
            if any(new_params == t for t in tried) or not changes:
                return  # don't propose something already registered
            out.append(Candidate(f"C{len(out) + 1}-{key}", best.id, title, hyp, changes, round(score, 4)))

        gap = m.get("precision", 0) - m.get("recall", 0)
        if "threshold" in prm and isinstance(prm["threshold"], (int, float)):
            step = -0.05 if gap > 0 else 0.05
            t = round(min(0.9, max(0.1, prm["threshold"] + step)), 2)
            add("threshold", f"Decision threshold {t}",
                f"{'Recall' if gap > 0 else 'Precision'} lags; moving the threshold to {t} should rebalance and lift {P}.",
                {"threshold": t}, 1.0 + abs(gap))
        if "C" in prm and isinstance(prm["C"], (int, float)):
            trend = None
            if parent and isinstance(parent.params.get("C"), (int, float)) and P in parent.metrics \
                    and parent.params["C"] != prm["C"]:
                went_up = prm["C"] > parent.params["C"]
                helped = m[P] >= parent.metrics[P]
                trend = 2.0 if went_up == helped else 0.5
            for factor in ([trend] if trend else [2.0, 0.5]):
                c = round(prm["C"] * factor, 4)
                why = ("Continue the regularisation change that improved " + P) if trend else \
                      f"Probe regularisation {'weaker' if factor > 1 else 'stronger'} around the current best"
                add(f"C{'up' if factor > 1 else 'down'}", f"Regularisation C={c:g}", why + ".",
                    {"C": c}, 0.8 if trend else 0.7)
        if "class_weight" in prm and prm["class_weight"] != "balanced" and gap > 0.05:
            add("cw", "Balanced class weights", "Recall is well below precision; balanced weights target the minority class.",
                {"class_weight": "balanced"}, 0.9 + gap)
        if "n_estimators" in prm and isinstance(prm["n_estimators"], int):
            add("trees", f"n_estimators={min(prm['n_estimators'] * 2, 2000)}", "More trees usually reduce variance.",
                {"n_estimators": min(prm["n_estimators"] * 2, 2000)}, 0.6)
        if "learning_rate" in prm and isinstance(prm["learning_rate"], (int, float)):
            add("lr", f"learning_rate={prm['learning_rate'] / 2:g}", "A smaller step size may generalise better.",
                {"learning_rate": prm["learning_rate"] / 2}, 0.5)
        out.sort(key=lambda c: -c.score)
        for i, c in enumerate(out):
            c.priority = min(5, i + 1)
            c.rationale = c.hypothesis
        return out

    # ------------------------------------------------------------------ decide
    def decide(self) -> Decision:
        ev = self.evidence()
        cands = self.candidates()
        if self.provider is None or not cands:
            reason = [] if self.provider is not None else ["LLM unavailable: deterministic explanation."]
            return self._fallback(ev, cands, reason)
        try:
            return self._llm_decide(ev, cands)
        except LLMError as exc:
            return self._fallback(ev, cands, [f"Groq request failed: {exc}. Deterministic explanation shown."])
        except (StructuredOutputError, ValidationError, ValueError) as exc:
            msg = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            return self._fallback(ev, cands, [f"Rejected model output ({msg}). Deterministic explanation shown."])

    def _llm_decide(self, ev: dict[str, Any], cands: list[Candidate]) -> Decision:
        payload = {**ev, "candidates": [{"candidate_id": c.candidate_id, "title": c.title, "parent_id": c.parent_id,
                                          "param_changes": c.param_changes, "hypothesis": c.hypothesis} for c in cands]}
        messages = [ChatMessage("system", prompts.BASE_SYSTEM), ChatMessage("user", prompts.decision_prompt(payload))]
        resp = self.provider.chat(messages, json_mode=True, temperature=0.2, max_tokens=1200)
        out = DecisionOutput.model_validate(extract_json(resp.text))

        by_id = {c.candidate_id: c for c in cands}
        unknown = [r.candidate_id for r in out.ranked if r.candidate_id not in by_id]
        if unknown:
            raise ValueError(f"model referenced unknown candidates {unknown}")
        problems = verify_cited_metrics(self.registry, out.cited_metrics)
        if problems:
            raise ValueError("cited metrics do not match the registry: " + "; ".join(problems[:3]))

        ranked: list[Candidate] = []
        for r in sorted(out.ranked, key=lambda r: r.priority):
            c = by_id[r.candidate_id]
            if c in ranked:
                continue
            c.priority, c.rationale = r.priority, r.rationale
            ranked.append(c)
        ranked += [c for c in cands if c not in ranked]  # keep anything the model skipped, at the end
        return Decision(out.summary, ev, ranked, out.caveats, "llm", llm_metadata=resp.metadata())

    def _fallback(self, ev: dict[str, Any], cands: list[Candidate], warnings: list[str]) -> Decision:
        return Decision(self.explain(ev), ev, cands, self._caveats(ev), "fallback", warnings)

    # ------------------------------------------------------------------ deterministic text
    def explain(self, ev: dict[str, Any]) -> str:
        if not ev["experiments"]:
            return "No completed experiments have a recorded primary metric yet."
        rows = {r["id"]: r for r in ev["experiments"]}
        parts = []
        base = rows.get(ev["baseline_id"])
        if base:
            parts.append(f"Baseline {base['id']} has {P}={base['metrics'][P]:.4f}.")
        best = rows.get(ev["best_id"])
        if best and best is not base:
            d = best.get(f"{P}_delta_vs_baseline")
            parts.append(f"Best so far is {best['id']} with {P}={best['metrics'][P]:.4f}"
                         + (f" ({d:+.4f} vs baseline)." if d is not None else "."))
        elif best is base:
            parts.append("No experiment has beaten the baseline yet.")
        return " ".join(parts)

    @staticmethod
    def _caveats(ev: dict[str, Any]) -> list[str]:
        out = ["Single evaluation split per experiment; differences under ~0.01 may be noise."]
        if ev["incomplete"]:
            out.append(f"Incomplete metrics: {', '.join(ev['incomplete'])}.")
        if ev["pending"]:
            out.append(f"Still pending: {', '.join(ev['pending'])}.")
        return out
