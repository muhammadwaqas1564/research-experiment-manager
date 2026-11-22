"""Offline stand-in for Groq that returns well-formed JSON, so the full LLM path
(prompting -> JSON -> Pydantic validation -> registry checks) runs with zero API calls.
"""
from __future__ import annotations

import json
import re

from llm.provider import ChatMessage, LLMResponse


class MockGroqProvider:
    name = "mock-groq"

    def __init__(self, model: str = "openai/gpt-oss-20b (mocked)"):
        self.model = model
        self.calls = 0

    def chat(self, messages: list[ChatMessage], *, json_mode: bool = False,
             temperature: float = 0.2, max_tokens: int = 1024) -> LLMResponse:
        self.calls += 1
        prompt = messages[-1].content
        if "TASK: PLAN" in prompt:
            text = self._plan(prompt)
        elif "TASK: DECIDE" in prompt:
            text = self._decide(prompt)
        else:
            text = json.dumps({"answer": "See the experiment table for details.", "cited_metrics": []})
        return LLMResponse(text=text, model=self.model, provider=self.name, request_id=f"mock-{self.calls}",
                           finish_reason="stop", usage={"total_tokens": 0})

    @staticmethod
    def _context(prompt: str, start: str, end: str) -> list | dict:
        blob = prompt.split(start, 1)[1].split(end, 1)[0].strip()
        return json.loads(blob)

    def _plan(self, prompt: str) -> str:
        cards = self._context(prompt, "Experiment context (computed locally):", "Researcher request:")
        request = prompt.rsplit("Researcher request:", 1)[1]
        ids = set(re.findall(r"EXP-\d{3,}", request))
        steps = []
        for c in cards:
            if (ids and c["id"] not in ids) or c["status"] not in ("completed", "analyzed"):
                continue
            steps.append({"action": "validate_metrics", "experiment_id": c["id"],
                          "reason": "Record file metrics and check completeness.",
                          "required_evidence": ["result_file"], "proposed_next_step": "Fill gaps",
                          "requires_approval": False})
            if c["missing_metrics"] and c["has_predictions"]:
                steps.append({"action": "calculate_missing_metric", "experiment_id": c["id"],
                              "reason": f"The {', '.join(c['missing_metrics'])} score is missing from the result file.",
                              "required_evidence": ["saved_predictions", "ground_truth_labels", "evaluation_split"],
                              "proposed_next_step": "Inspect available evaluation artifacts",
                              "requires_approval": False})
            if not c.get("is_baseline"):
                steps.append({"action": "compare_to_baseline", "experiment_id": c["id"],
                              "reason": "Quantify the effect against the baseline.",
                              "required_evidence": ["baseline_metrics"], "proposed_next_step": "Rank next experiments",
                              "requires_approval": False})
        return json.dumps({"summary": "Complete missing metrics, then compare with the baseline.",
                           "steps": steps[:8] or [{"action": "no_action", "reason": "Nothing to analyse."}]})

    def _decide(self, prompt: str) -> str:
        ev = json.loads(prompt.rsplit("Evidence:", 1)[1])
        best = next((e for e in ev["experiments"] if e["id"] == ev["best_id"]), None)
        cited = []
        if best:
            cited.append({"experiment_id": best["id"], "metric": ev["primary_metric"],
                          "value": best["metrics"][ev["primary_metric"]]})
        ranked = [{"candidate_id": c["candidate_id"], "priority": min(5, i + 1),
                   "rationale": f"(mock) {c['hypothesis']}"} for i, c in enumerate(ev["candidates"])]
        return json.dumps({"summary": f"(mock) {ev['best_id']} is currently best on {ev['primary_metric']}.",
                           "ranked": ranked, "cited_metrics": cited,
                           "caveats": ["Mocked model output for offline demonstration."]})
