"""Registry integrity under API failures, the credit-free three-cycle demo, and a UI smoke test."""
from __future__ import annotations

import pytest

from agent.decision_engine import DecisionEngine
from agent.executor import Executor
from agent.planner import Planner
from agent.tools import ToolContext, run_tool, build_args
from demo.mock_llm import MockGroqProvider
from demo.run_demo import run_demo
from llm.groq_client import GroqClient
from llm.provider import LLMUnavailableError
from tests.conftest import FAKE_KEY, FakeGroqSDK, sdk_error


def test_api_failures_do_not_corrupt_registry(registry, settings, scripted):
    """Every LLM entry point fails; the registry must be byte-for-byte unchanged."""
    before = registry.snapshot()
    sdk = FakeGroqSDK(*[sdk_error("InternalServerError", 500) for _ in range(20)])
    client = GroqClient(api_key=FAKE_KEY, client=sdk, max_retries=1, sleep=lambda s: None)
    Planner(registry, settings, client).plan("Analyse EXP-002")
    DecisionEngine(registry, client).decide()
    from agent.assistant import ResearchAssistant
    ResearchAssistant(registry, settings, client).answer("status?")
    assert registry.snapshot() == before


def test_failure_mid_transaction_rolls_back(registry, settings, monkeypatch):
    """If something blows up inside a multi-write tool, nothing partial is persisted."""
    before = registry.snapshot()
    files_before = sorted(p.name for p in settings.tasks_dir.iterdir())

    def boom(*a, **k):
        raise LLMUnavailableError("simulated outage during write")

    monkeypatch.setattr("agent.tools.write_task_file", boom)
    args = build_args("propose_experiment", "EXP-002", {"name": "New run", "hypothesis": "hyp", "params": {"C": 3}})
    with pytest.raises(LLMUnavailableError):
        run_tool(ToolContext(registry, settings), "propose_experiment", args)
    assert registry.snapshot() == before
    assert sorted(p.name for p in settings.tasks_dir.iterdir()) == files_before


@pytest.mark.parametrize("mode", ["offline", "mock"])
def test_three_cycle_demo_runs_without_api_credits(tmp_path, mode):
    provider = MockGroqProvider() if mode == "mock" else None
    lines: list[str] = []
    summary = run_demo(tmp_path / "demo", provider, out=lines.append)
    assert len(summary["cycles"]) == 3 and summary["all_verified"]
    exps = summary["experiments"]
    assert exps["EXP-002"]["f1"] is not None              # missing F1 was computed
    assert exps["EXP-004"]["status"] == "analyzed"          # cycle-1 proposal was run and analysed
    assert summary["next_planned"] and exps[summary["next_planned"]]["status"] == "planned"
    expected = "llm" if mode == "mock" else "fallback"
    assert all(c["plan_source"] == expected for c in summary["cycles"])
    if provider:
        assert provider.calls >= 7  # every LLM entry point was exercised, all mocked
    assert (tmp_path / "demo" / "tasks" / f"{summary['next_planned']}-task.md").exists()


def test_demo_refuses_to_wipe_foreign_directory(tmp_path):
    victim = tmp_path / "important"
    victim.mkdir()
    (victim / "thesis.tex").write_text("do not delete")
    with pytest.raises(SystemExit):
        run_demo(victim, None, out=lambda s: None)
    assert (victim / "thesis.tex").exists()


def test_streamlit_app_runs_in_fallback_mode(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("REM_DATA_DIR", str(tmp_path / "ui"))
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert "GROQ_API_KEY" in at.sidebar.warning[0].value
    at.sidebar.button[0].click().run()               # load demo data
    at.tabs[1].button[0].click().run()               # propose plan
    at.tabs[1].button[1].click().run()               # execute
    assert not at.exception
    assert any("Final verification" in s.value and "passed" in s.value for s in at.success)
    at.chat_input[0].set_value("Which experiment is best?").run()
    assert not at.exception
