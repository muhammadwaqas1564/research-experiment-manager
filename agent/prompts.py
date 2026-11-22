"""System instructions and reusable prompt templates.

Prompts are short on purpose (free-tier token budgets) and only ever receive compact,
pre-computed context — never raw CSVs or log directories.
"""
from __future__ import annotations

import json
from typing import Any

BASE_SYSTEM = """You are the planning assistant inside a research experiment manager.
Hard rules:
- You never execute code. You may only propose actions from the allowlisted tool catalog.
- All metric values come from deterministic tools. Never invent, estimate or round-trip-guess a metric.
  If a value is not in the provided context, say it is unknown.
- Only reference experiment IDs that appear in the context.
- Be concise."""

PLANNER_TASK = """TASK: PLAN
Interpret the researcher's request and propose an ordered plan using ONLY these tools:
{tools}

Return a single JSON object, no prose, with this shape:
{{"summary": "<one sentence>",
  "steps": [{{"action": "<tool name>", "experiment_id": "<EXP-### or null>",
             "reason": "<why>", "required_evidence": ["<artifact or fact needed>"],
             "proposed_next_step": "<what happens after>", "requires_approval": <true|false>,
             "arguments": {{}}}}]}}
Use at most 8 steps. Put tool-specific arguments (e.g. name, hypothesis, params, new_status) in "arguments".

Experiment context (computed locally):
{context}

Researcher request: {request}"""

DECISION_TASK = """TASK: DECIDE
Explain the evidence and prioritise the candidate next experiments. The candidates and every
number below were computed deterministically. You may only rank the given candidate_ids;
do not create new candidates or new numbers.

Return a single JSON object, no prose:
{{"summary": "<2-3 sentences explaining what the evidence shows>",
  "ranked": [{{"candidate_id": "<id>", "priority": <1-5, 1 = do first>, "rationale": "<why>"}}],
  "cited_metrics": [{{"experiment_id": "<EXP-###>", "metric": "<name>", "value": <number copied exactly>}}],
  "caveats": ["<limitations of the evidence>"]}}

Evidence:
{evidence}"""

ASSISTANT_TASK = """TASK: ANSWER
Answer the researcher's question using only the experiment records below. If the answer is
not in the records, say so. Copy numbers exactly as given; do not compute new statistics.

Return a single JSON object, no prose:
{{"answer": "<markdown answer>",
  "cited_metrics": [{{"experiment_id": "<EXP-###>", "metric": "<name>", "value": <number copied exactly>}}]}}

Experiment records:
{context}

Question: {question}"""


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True, default=str)


def planner_prompt(tools: str, context: Any, request: str) -> str:
    return PLANNER_TASK.format(tools=tools, context=compact_json(context), request=request.strip()[:1000])


def decision_prompt(evidence: Any) -> str:
    return DECISION_TASK.format(evidence=compact_json(evidence))


def assistant_prompt(context: Any, question: str) -> str:
    return ASSISTANT_TASK.format(context=compact_json(context), question=question.strip()[:1000])
