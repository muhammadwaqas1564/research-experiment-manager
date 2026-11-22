"""Pydantic schemas for structured LLM output, plus a safe JSON extractor."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.tools import TOOLS
from core.registry import EXPERIMENT_ID_RE


class StructuredOutputError(ValueError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text. Tolerates ```json fences; nothing else."""
    if not isinstance(text, str) or not text.strip():
        raise StructuredOutputError("Empty model output")
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise StructuredOutputError("Model output must be a JSON object")
    return data


def _check_exp_id(v: Optional[str]) -> Optional[str]:
    if v in (None, "", "null"):
        return None
    if not EXPERIMENT_ID_RE.match(v):
        raise ValueError(f"invalid experiment id {v!r}")
    return v


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str
    experiment_id: Optional[str] = None
    reason: str = Field(min_length=1, max_length=500)
    required_evidence: list[str] = Field(default_factory=list, max_length=10)
    proposed_next_step: str = Field(default="", max_length=400)
    requires_approval: bool = False
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def _allowlisted(cls, v: str) -> str:
        if v not in TOOLS:
            raise ValueError(f"unsupported action {v!r}")
        return v

    @field_validator("experiment_id")
    @classmethod
    def _exp(cls, v: Optional[str]) -> Optional[str]:
        return _check_exp_id(v)


class PlanOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1, max_length=600)
    steps: list[dict[str, Any]] = Field(min_length=1, max_length=8)  # validated one by one


class CitedMetric(BaseModel):
    model_config = ConfigDict(extra="ignore")

    experiment_id: str
    metric: str
    value: float

    @field_validator("experiment_id")
    @classmethod
    def _exp(cls, v: Optional[str]) -> Optional[str]:
        return _check_exp_id(v)


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    priority: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1, max_length=600)


class DecisionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1, max_length=1200)
    ranked: list[RankedCandidate] = Field(min_length=1, max_length=10)
    cited_metrics: list[CitedMetric] = Field(default_factory=list, max_length=40)
    caveats: list[str] = Field(default_factory=list, max_length=10)


class AssistantOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1, max_length=6000)
    cited_metrics: list[CitedMetric] = Field(default_factory=list, max_length=40)
