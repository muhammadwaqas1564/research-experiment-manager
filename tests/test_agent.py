"""Planner / executor / decision engine / assistant with mocked Groq responses."""
from __future__ import annotations

import json

import pytest

from agent.assistant import ResearchAssistant
from agent.decision_engine import DecisionEngine
from agent.executor import Executor
from agent.planner import Planner
from llm.groq_client import GroqClient
from llm.provider import LLMRateLimitError, LLMUnavailableError
from tests.conftest import FAKE_KEY, FakeGroqSDK, sdk_completion, sdk_error

GOOD_PLAN = {
    "summary": "Fill the missing F1 for EXP-002, then compare to baseline.",
    "steps": [
        {"action": "calculate_missing_metric", "experiment_id": "EXP-002",
         "reason": "The F1-score is missing from the result file.",
         "required_evidence": ["saved_predictions", "ground_truth_labels", "evaluation_split"],
         "proposed_next_step": "Inspect available evaluation artifacts", "requires_approval": False},
        {"action": "compare_to_baseline", "experiment_id": "EXP-002", "reason": "Quantify improvement.",
         "requires_approval": False},
    ],
}


# ------------------------------------------------------------------ planning workflow
def test_mocked_groq_response_drives_planning_workflow(registry, settings):
    """End-to-end through the real GroqClient with a fake SDK: no network, no key, no credits."""
    sdk = FakeGroqSDK(sdk_completion(json.dumps(GOOD_PLAN)))
    client = GroqClient(api_key=FAKE_KEY, client=sdk)
    plan = Planner(registry, settings, client).plan("EXP-002 is missing F1")
    assert plan.source == "llm" and [s.action for s in plan.steps] == ["calculate_missing_metric", "compare_to_baseline"]
    assert plan.llm_metadata["request_id"] == "chatcmpl-123"
    # Prompt is compact: no raw CSV rows, only the relevant experiments.
    prompt = sdk.calls[0]["messages"][1]["content"]
    assert "y_true" not in prompt and "EXP-003" not in prompt and "EXP-002" in prompt
    assert len(prompt) < 6000

    report = Executor(registry, settings).execute(plan)
    assert report.ok, [c for c in report.checks if not c.passed]
    assert registry.metric_sources("EXP-002")["f1"] == "computed"
    assert registry.get("EXP-002").status == "analyzed"


@pytest.mark.parametrize("bad_output", [
    "this is not json",
    "```json\n{\"summary\": \"x\"}\n```",            # missing steps
    json.dumps({"summary": "x", "steps": []}),         # empty steps
    json.dumps(["not", "an", "object"]),
    json.dumps({"summary": "x", "steps": [{"action": "run_shell", "experiment_id": "EXP-002",
                                           "reason": "rm -rf /"}]}),
    json.dumps({"summary": "x", "steps": [{"action": "compare_to_baseline", "experiment_id": "EXP-999",
                                           "reason": "invented id"}]}),
    json.dumps({"summary": "x", "steps": [{"action": "update_status", "experiment_id": "EXP-002",
                                           "reason": "bad args", "arguments": {"new_status": 5, "evil": 1}}]}),
])
def test_malformed_structured_output_is_rejected_safely(registry, settings, scripted, bad_output):
    before = registry.snapshot()
    plan = Planner(registry, settings, scripted(bad_output)).plan("Analyse EXP-002")
    assert plan.source == "fallback"
    assert plan.warnings
    assert all(s.action in {"validate_metrics", "calculate_missing_metric", "compare_to_baseline",
                            "summarize_experiment", "no_action"} for s in plan.steps)
    assert registry.snapshot() == before  # planning never writes


def test_invalid_steps_are_dropped_and_valid_ones_kept(registry, settings, scripted):
    out = dict(GOOD_PLAN, steps=GOOD_PLAN["steps"] + [
        {"action": "exec_python", "experiment_id": "EXP-002", "reason": "x"},
        {"action": "summarize_experiment", "experiment_id": "EXP-404", "reason": "x"},
    ])
    plan = Planner(registry, settings, scripted(out)).plan("go")
    assert plan.source == "llm" and len(plan.steps) == 2 and len(plan.rejected) == 2
    assert any("unsupported action" in w for w in plan.warnings)
    assert any("not a registered experiment" in w for w in plan.warnings)


def test_registry_approval_rule_overrides_model(registry, settings, scripted):
    out = {"summary": "s", "steps": [{"action": "update_status", "experiment_id": "EXP-003",
                                      "reason": "mark failed", "requires_approval": False,
                                      "arguments": {"new_status": "failed"}}]}
    plan = Planner(registry, settings, scripted(out)).plan("EXP-003 crashed")
    assert plan.steps[0].requires_approval is True
    rep = Executor(registry, settings).execute(plan)  # not approved
    assert rep.skipped and registry.get("EXP-003").status == "running"
    rep = Executor(registry, settings).execute(plan, approved={0})
    assert registry.get("EXP-003").status == "failed"


@pytest.mark.parametrize("error", [LLMRateLimitError("Groq rate limit reached"), LLMUnavailableError("down")])
def test_api_failure_falls_back_with_clear_message(registry, settings, scripted, error):
    plan = Planner(registry, settings, scripted(error)).plan("Analyse EXP-002")
    assert plan.source == "fallback" and "Groq request failed" in plan.warnings[0]
    assert any(s.action == "calculate_missing_metric" for s in plan.steps)


# ------------------------------------------------------------------ decision engine
def _analysed(registry, settings):
    plan = Planner(registry, settings, None).plan("analyse EXP-002")
    assert Executor(registry, settings).execute(plan).ok


def test_decision_with_valid_llm_ranking(registry, settings, scripted):
    _analysed(registry, settings)
    cands = DecisionEngine(registry).candidates()
    last = cands[-1].candidate_id
    f1 = registry.metrics("EXP-002")["f1"]
    out = {"summary": "EXP-002 beats the baseline.", "ranked": [{"candidate_id": last, "priority": 1, "rationale": "r"}],
           "cited_metrics": [{"experiment_id": "EXP-002", "metric": "f1", "value": round(f1, 4)}], "caveats": []}
    d = DecisionEngine(registry, scripted(out)).decide()
    assert d.source == "llm" and d.candidates[0].candidate_id == last
    assert len(d.candidates) == len(cands)  # skipped candidates are kept, not lost


@pytest.mark.parametrize("mutate", ["invented_metric", "unknown_candidate", "garbage"])
def test_decision_rejects_invented_numbers_and_candidates(registry, settings, scripted, mutate):
    _analysed(registry, settings)
    cid = DecisionEngine(registry).candidates()[0].candidate_id
    out = {"summary": "s", "ranked": [{"candidate_id": cid, "priority": 1, "rationale": "r"}],
           "cited_metrics": [{"experiment_id": "EXP-002", "metric": "f1", "value": 0.99}]}  # invented
    if mutate == "unknown_candidate":
        out["cited_metrics"] = []
        out["ranked"][0]["candidate_id"] = "C9-made-up"
    if mutate == "garbage":
        out = "{not json"
    d = DecisionEngine(registry, scripted(out)).decide()
    assert d.source == "fallback" and d.warnings
    assert "0.99" not in d.summary


# ------------------------------------------------------------------ assistant
def test_assistant_accepts_grounded_answer(registry, settings, scripted):
    _analysed(registry, settings)
    f1 = round(registry.metrics("EXP-002")["f1"], 4)
    out = {"answer": f"EXP-002 has the best F1 ({f1}).",
           "cited_metrics": [{"experiment_id": "EXP-002", "metric": "f1", "value": f1}]}
    ans = ResearchAssistant(registry, settings, scripted(out)).answer("best?")
    assert ans.source == "llm" and str(f1) in ans.text


def test_assistant_rejects_hallucinated_numbers(registry, settings, scripted):
    _analysed(registry, settings)
    out = {"answer": "EXP-002 reached an F1 of 0.9731.", "cited_metrics": []}
    ans = ResearchAssistant(registry, settings, scripted(out)).answer("What is the F1 of EXP-002?")
    assert ans.source == "fallback" and "0.9731" not in ans.text
    assert "unsupported numbers" in ans.warnings[0]


def test_assistant_works_without_llm(registry, settings):
    ans = ResearchAssistant(registry, settings, None).answer("Which experiments have missing metrics?")
    assert ans.source == "fallback" and "EXP-002" in ans.text and "f1" in ans.text
