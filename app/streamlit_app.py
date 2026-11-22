"""Streamlit UI for the research experiment manager.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent.assistant import ResearchAssistant  # noqa: E402
from agent.decision_engine import DecisionEngine  # noqa: E402
from agent.executor import Executor  # noqa: E402
from agent.planner import Planner  # noqa: E402
from core import analysis  # noqa: E402
from core.config import Settings  # noqa: E402
from core.paths import PathValidationError, safe_filename, safe_resolve  # noqa: E402``
from core.registry import TRANSITIONS, Registry, RegistryError  # noqa: E402
from demo.seed import seed_demo  # noqa: E402
from llm.provider import get_provider  # noqa: E402

st.set_page_config(page_title="Research Experiment Manager", page_icon="🧪", layout="wide")


@st.cache_resource
def services():
    settings = Settings.from_env()
    settings.ensure_dirs()
    registry = Registry(settings.db_path)
    status = get_provider(settings.cache_path)
    return settings, registry, status


settings, registry, llm_status = services()
provider = llm_status.provider
executor = Executor(registry, settings)


def source_badge(source: str) -> str:
    return "🤖 Groq GPT-OSS" if source == "llm" else "⚙️ deterministic"


def show_report(rep) -> None:
    for r in rep.results:
        (st.success if r.ok else st.error)(f"**{r.tool}** {r.experiment_id or ''} — {r.message}")
        for w in r.warnings:
            st.warning(w)
    for s in rep.skipped:
        st.info(f"Skipped **{s['action']}** {s['experiment_id'] or ''}: {s['reason']}")
    if rep.checks:
        passed = sum(c.passed for c in rep.checks)
        (st.success if rep.verified else st.error)(f"Final verification: {passed}/{len(rep.checks)} checks passed")
        with st.expander("Verification details"):
            st.dataframe(pd.DataFrame([c.__dict__ for c in rep.checks]), hide_index=True, width="stretch")


# ------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🧪 Experiment manager")
    if llm_status.available:
        st.success(llm_status.message)
    else:
        st.warning(llm_status.message)
    st.caption("Metrics, comparisons, status changes and file writes are always computed locally. "
               "The model only plans, explains and answers questions.")
    exps = registry.list()
    st.metric("Registered experiments", len(exps))
    if not exps and st.button("Load demo experiments"):
        seed_demo(registry, settings)
        st.rerun()

tab_exp, tab_plan, tab_next, tab_chat = st.tabs(["Experiments", "Plan & run", "Next experiments", "Assistant"])

# ------------------------------------------------------------------------------ experiments
with tab_exp:
    exps = registry.list()
    if not exps:
        st.info("No experiments yet. Register one below, or load the demo experiments from the sidebar.")
    else:
        rows = []
        for e in exps:
            rows.append({"id": e.id, "name": e.name, "status": e.status, "baseline": e.is_baseline,
                         **{m: e.metrics.get(m) for m in analysis.REQUIRED_METRICS},
                         "missing": ", ".join(analysis.missing_metrics(e.metrics))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={m: st.column_config.NumberColumn(format="%.4f") for m in analysis.REQUIRED_METRICS})

        sel = st.selectbox("Experiment details", [e.id for e in exps])
        exp = registry.get(sel)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{exp.name}** · _{exp.status}_" + (" · baseline" if exp.is_baseline else ""))
            if exp.hypothesis:
                st.caption(exp.hypothesis)
            st.json(exp.params, expanded=False)
            sources = registry.metric_sources(exp.id)
            if exp.metrics:
                st.dataframe(pd.DataFrame([{"metric": k, "value": v, "source": sources.get(k)}
                                           for k, v in exp.metrics.items()]), hide_index=True)
            base = registry.baseline()
            if base and base.id != exp.id and exp.metrics and base.metrics:
                st.markdown(f"**vs baseline {base.id}**")
                st.dataframe(pd.DataFrame([d.__dict__ for d in analysis.compare_to_baseline(exp.metrics, base.metrics)]),
                             hide_index=True)
        with c2:
            allowed = sorted(TRANSITIONS.get(exp.status, set()))
            if allowed:
                with st.form(f"status-{exp.id}"):
                    new = st.selectbox("Change status", allowed)
                    note = st.text_input("Note")
                    if st.form_submit_button("Apply"):
                        try:
                            registry.transition(exp.id, new, note)
                            st.rerun()
                        except RegistryError as err:
                            st.error(str(err))
            with st.form(f"upload-{exp.id}"):
                st.markdown("**Attach results**")
                res = st.file_uploader("Summary CSV (metric,value)", type="csv", key=f"r-{exp.id}")
                pred = st.file_uploader("Predictions CSV (y_true,y_pred)", type="csv", key=f"p-{exp.id}")
                if st.form_submit_button("Save files"):
                    try:
                        saved = {}
                        for kind, f in (("result_path", res), ("predictions_path", pred)):
                            if f is not None:
                                name = safe_filename(f"{exp.id}_{'results' if kind == 'result_path' else 'predictions'}.csv")
                                safe_resolve(settings.results_dir, name).write_bytes(f.getvalue())
                                saved[kind] = name
                        if saved:
                            registry.set_artifacts(exp.id, **saved)
                            st.success("Saved " + ", ".join(saved.values()))
                    except (PathValidationError, RegistryError) as err:
                        st.error(str(err))
            with st.expander("History"):
                st.dataframe(pd.DataFrame(registry.events(exp.id))[["created_at", "kind", "detail"]], hide_index=True)

    with st.expander("Register a new experiment"):
        with st.form("new-exp"):
            name = st.text_input("Name")
            hyp = st.text_area("Hypothesis")
            model_type = st.text_input("Model type", "logistic_regression")
            params_raw = st.text_area("Parameters (JSON)", '{"C": 1.0, "threshold": 0.5}')
            parent = st.selectbox("Parent", ["(none)"] + [e.id for e in registry.list()])
            is_base = st.checkbox("This is the baseline", value=registry.baseline() is None)
            if st.form_submit_button("Register"):
                try:
                    params = json.loads(params_raw or "{}")
                    if not isinstance(params, dict):
                        raise ValueError("Parameters must be a JSON object")
                    e = registry.create_experiment(name or "Untitled", hypothesis=hyp, model_type=model_type,
                                                   params=params, is_baseline=is_base,
                                                   parent_id=None if parent == "(none)" else parent)
                    st.success(f"Registered {e.id}")
                    st.rerun()
                except (ValueError, RegistryError) as err:
                    st.error(str(err))

# ------------------------------------------------------------------------------ plan & run
with tab_plan:
    request = st.text_area("What should the agent do?",
                           "EXP-002 finished but its F1 score looks missing. Fill it in and compare with the baseline.")
    if st.button("Propose plan", type="primary"):
        with st.spinner("Planning…"):
            st.session_state["plan"] = Planner(registry, settings, provider).plan(request)
    plan = st.session_state.get("plan")
    if plan:
        st.markdown(f"**Plan** ({source_badge(plan.source)}): {plan.summary}")
        for w in plan.warnings:
            st.warning(w)
        if plan.rejected:
            with st.expander(f"{len(plan.rejected)} rejected model step(s)"):
                st.json(plan.rejected)
        approved = set()
        for i, s in enumerate(plan.steps):
            label = f"{i + 1}. `{s.action}` {s.experiment_id or ''} — {s.reason}"
            if s.requires_approval:
                if st.checkbox(f"Approve: {label}", key=f"approve-{i}"):
                    approved.add(i)
            else:
                st.markdown(label)
        if plan.llm_metadata:
            st.caption(f"model {plan.llm_metadata.get('model')} · {plan.llm_metadata.get('usage')} · "
                       f"{'cached' if plan.llm_metadata.get('cached') else 'live'}")
        if st.button("Execute plan"):
            show_report(executor.execute(plan, approved))
            st.session_state.pop("plan", None)
            st.session_state.pop("decision", None)

# ------------------------------------------------------------------------------ next experiments
with tab_next:
    if st.button("Analyse evidence", type="primary"):
        with st.spinner("Weighing evidence…"):
            st.session_state["decision"] = DecisionEngine(registry, provider).decide()
    dec = st.session_state.get("decision")
    if dec:
        st.markdown(f"**Summary** ({source_badge(dec.source)})")
        st.write(dec.summary)
        for w in dec.warnings:
            st.warning(w)
        if not dec.candidates:
            st.info("No new candidates. Complete or analyse more experiments first.")
        for i, c in enumerate(dec.candidates):
            with st.container(border=True):
                st.markdown(f"**P{c.priority} · {c.title}** (from {c.parent_id})")
                st.caption(c.rationale)
                st.code(json.dumps(c.param_changes), language="json")
                if st.button("Register as planned experiment", key=f"prop-{c.candidate_id}"):
                    show_report(executor.run_steps(dec.proposal_steps(len(dec.candidates))[i:i + 1], approved=True))
                    st.session_state.pop("decision", None)
        if dec.caveats:
            st.caption("Caveats: " + " ".join(dec.caveats))

# ------------------------------------------------------------------------------ assistant
with tab_chat:
    st.caption("Ask about registered experiments. Answers are checked against the registry; "
               "unsupported numbers are rejected.")
    history = st.session_state.setdefault("chat", [])
    for role, text, src in history:
        with st.chat_message(role):
            st.markdown(text)
            if src:
                st.caption(source_badge(src))
    if q := st.chat_input("e.g. Which experiment is best so far?"):
        ans = ResearchAssistant(registry, settings, provider).answer(q)
        text = ans.text + ("\n\n" + "\n".join(f"> ⚠️ {w}" for w in ans.warnings) if ans.warnings else "")
        history += [("user", q, None), ("assistant", text, ans.source)]
        st.rerun()
