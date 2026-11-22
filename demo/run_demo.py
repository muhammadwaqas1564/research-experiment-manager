"""Three-cycle demonstration of the research loop: analyse -> decide -> run -> analyse.

    python -m demo.run_demo                 # deterministic, no LLM, no API calls (default)
    python -m demo.run_demo --llm mock      # full LLM code path with a mocked Groq, no API calls
    python -m demo.run_demo --llm groq      # real Groq GPT-OSS (uses your free-tier quota)

The demo auto-approves gated steps so it can run unattended; the UI asks instead.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable

from agent.assistant import ResearchAssistant
from agent.decision_engine import DecisionEngine
from agent.executor import ExecutionReport, Executor
from agent.planner import Planner
from core.config import PROJECT_ROOT, Settings
from core.registry import Registry
from demo.mock_llm import MockGroqProvider
from demo.seed import seed_demo
from demo.simulator import write_run_outputs
from llm.provider import LLMProvider, get_provider


def _report(out: Callable[[str], None], rep: ExecutionReport) -> None:
    for r in rep.results:
        out(f"    {'✓' if r.ok else '✗'} {r.tool:<26} {r.experiment_id or '':<8} {r.message}")
        for w in r.warnings:
            out(f"      ! {w}")
    for s in rep.skipped:
        out(f"    … {s['action']} {s['experiment_id']} skipped ({s['reason']})")
    failed = [c for c in rep.checks if not c.passed]
    out(f"    verification: {len(rep.checks) - len(failed)}/{len(rep.checks)} checks passed"
        + ("" if not failed else " — FAILED: " + "; ".join(f"{c.name} {c.detail}" for c in failed)))


def _simulate_run(reg: Registry, settings: Settings, exp_id: str, out: Callable[[str], None]) -> None:
    exp = reg.get(exp_id)
    reg.transition(exp_id, "running", "demo simulator")
    res, pred = write_run_outputs(settings.results_dir, exp_id, exp.params)
    reg.set_artifacts(exp_id, result_path=res, predictions_path=pred)
    reg.transition(exp_id, "completed", "demo simulator")
    out(f"    ran {exp_id} with {exp.params} (simulated training job)")


def run_demo(data_dir: Path, provider: LLMProvider | None, out: Callable[[str], None] = print) -> dict[str, Any]:
    marker = data_dir / ".rem-demo"
    if data_dir.exists() and any(data_dir.iterdir()):
        if not marker.exists():
            raise SystemExit(f"Refusing to reset {data_dir}: it is not a demo directory created by this script.")
        shutil.rmtree(data_dir)
    settings = Settings.from_env(data_dir)
    settings.ensure_dirs()
    marker.write_text("Created by demo/run_demo.py; safe to delete.\n")
    reg = Registry(settings.db_path)
    seed_demo(reg, settings)
    planner, executor = Planner(reg, settings, provider), Executor(reg, settings)
    engine = DecisionEngine(reg, provider)
    summary: dict[str, Any] = {"cycles": [], "all_verified": True}

    def cycle(n: int, request: str) -> tuple[Any, Any]:
        out(f"\n=== Cycle {n} ===\n  request: {request}")
        plan = planner.plan(request)
        out(f"  plan ({plan.source}): {plan.summary}")
        for w in plan.warnings:
            out(f"    ! {w}")
        rep = executor.execute(plan)
        _report(out, rep)
        decision = engine.decide()
        out(f"  decision ({decision.source}): {decision.summary}")
        for c in decision.candidates[:3]:
            out(f"    P{c.priority} {c.candidate_id}: {c.title} — {c.rationale}")
        summary["all_verified"] &= rep.verified
        summary["cycles"].append({"plan_source": plan.source, "decision_source": decision.source,
                                  "verified": rep.verified, "steps": len(plan.steps)})
        return decision, rep

    def propose(decision: Any) -> str | None:
        steps = decision.proposal_steps(1)
        if not steps:
            out("  no further candidates to propose")
            return None
        out("  [demo] auto-approving proposal: " + steps[0].args.name)
        rep = executor.run_steps(steps, approved=True)
        _report(out, rep)
        summary["all_verified"] &= rep.verified
        return rep.results[0].experiment_id if rep.results and rep.results[0].ok else None

    d1, _ = cycle(1, "EXP-002 finished but its F1 score looks missing. Fill it in and compare with the baseline.")
    new_id = propose(d1)
    if new_id:
        _simulate_run(reg, settings, new_id, out)
    d2, _ = cycle(2, f"Analyse {new_id} against the baseline." if new_id else "Analyse completed experiments.")
    new_id2 = propose(d2)
    if new_id2:
        _simulate_run(reg, settings, new_id2, out)
    d3, _ = cycle(3, "Analyse every completed experiment and recommend what to try next.")
    final_id = propose(d3)

    answer = ResearchAssistant(reg, settings, provider).answer("Which experiment is best so far?")
    out(f"\n=== Assistant ({answer.source}) ===\n{answer.text}")

    summary["experiments"] = {e.id: {"status": e.status, "f1": e.metrics.get("f1")} for e in reg.list()}
    summary["next_planned"] = final_id
    out("\n=== Registry ===")
    for k, v in summary["experiments"].items():
        out(f"  {k}: {v['status']:<10} f1={v['f1'] if v['f1'] is None else round(v['f1'], 4)}")
    out(f"\nAll verification checks passed: {summary['all_verified']}")
    reg.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["offline", "mock", "groq"], default="offline")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "demo"))
    a = ap.parse_args()
    provider: LLMProvider | None = None
    if a.llm == "mock":
        provider = MockGroqProvider()
    elif a.llm == "groq":
        status = get_provider(Path(a.data_dir).parent / "llm_cache.sqlite")
        print(status.message)
        provider = status.provider
    summary = run_demo(Path(a.data_dir), provider)
    raise SystemExit(0 if summary["all_verified"] else 1)


if __name__ == "__main__":
    main()
