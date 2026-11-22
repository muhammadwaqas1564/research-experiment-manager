"""Research assistant used by the Streamlit chat: answers questions about registered experiments.

Answers from Groq are accepted only if every cited metric matches the registry and every
decimal number in the prose appears in the context that was sent. Otherwise the
deterministic answer is shown instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent import prompts
from agent.context import build_context, mentioned_ids, unsupported_numbers, verify_cited_metrics
from agent.schemas import AssistantOutput, StructuredOutputError, extract_json
from core import analysis
from core.config import Settings
from core.registry import Registry
from llm.provider import ChatMessage, LLMError, LLMProvider


@dataclass
class Answer:
    text: str
    source: str  # "llm" | "fallback"
    warnings: list[str] = field(default_factory=list)
    llm_metadata: dict[str, Any] = field(default_factory=dict)


class ResearchAssistant:
    def __init__(self, registry: Registry, settings: Settings, provider: LLMProvider | None = None):
        self.registry = registry
        self.settings = settings
        self.provider = provider

    def answer(self, question: str) -> Answer:
        context = build_context(self.registry, self.settings, mentioned_ids(question))
        if self.provider is None:
            return Answer(self.fallback(question, context), "fallback", ["LLM unavailable: deterministic answer."])
        try:
            messages = [ChatMessage("system", prompts.BASE_SYSTEM),
                        ChatMessage("user", prompts.assistant_prompt(context, question))]
            resp = self.provider.chat(messages, json_mode=True, temperature=0.2, max_tokens=900)
            out = AssistantOutput.model_validate(extract_json(resp.text))
            problems = verify_cited_metrics(self.registry, out.cited_metrics)
            stray = unsupported_numbers(out.answer, context)
            if problems or stray:
                detail = problems[:2] + ([f"unsupported numbers {stray[:3]}"] if stray else [])
                return Answer(self.fallback(question, context), "fallback",
                              ["Model answer rejected: " + "; ".join(detail)])
            return Answer(out.answer, "llm", llm_metadata=resp.metadata())
        except LLMError as exc:
            return Answer(self.fallback(question, context), "fallback", [f"Groq request failed: {exc}"])
        except (StructuredOutputError, ValidationError) as exc:
            return Answer(self.fallback(question, context), "fallback", [f"Malformed model output rejected: {exc}"])

    # ------------------------------------------------------------------ deterministic answer
    def fallback(self, question: str, context: list[dict[str, Any]]) -> str:
        q = question.lower()
        ids = mentioned_ids(question)
        cards = [c for c in context if c["id"] in ids] if ids else context
        if not cards:
            return "No experiments are registered yet."
        p = analysis.PRIMARY_METRIC
        if not ids and any(w in q for w in ("best", "top", "highest", "winner")):
            scored = [c for c in cards if p in c["metrics"]]
            if not scored:
                return f"No experiment has a recorded {p} yet."
            best = max(scored, key=lambda c: c["metrics"][p])
            cards = [best]
            header = f"**Best by {p}:**\n\n"
        elif not ids and "missing" in q:
            cards = [c for c in cards if c["missing_metrics"]]
            header = "**Experiments with missing metrics:**\n\n" if cards else "No experiments have missing metrics.\n"
        else:
            header = ""
        lines = []
        for c in cards:
            metrics = ", ".join(f"{k}={v:.4f}" for k, v in sorted(c["metrics"].items())) or "no metrics yet"
            extra = []
            if c.get("is_baseline"):
                extra.append("baseline")
            if f"{p}_delta_vs_baseline" in c:
                extra.append(f"{p} Δ vs baseline {c[f'{p}_delta_vs_baseline']:+.4f}")
            if c["missing_metrics"]:
                extra.append("missing: " + ", ".join(c["missing_metrics"]))
            lines.append(f"- **{c['id']}** · {c['name']} · _{c['status']}_ · {metrics}"
                         + (f" ({'; '.join(extra)})" if extra else ""))
        return header + "\n".join(lines)
