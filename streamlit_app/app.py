"""Streamlit demo app for the patient-trial matching agent.

Drop into your repo as streamlit_app/app.py
Run with: streamlit run streamlit_app/app.py
"""
import json
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import run_agent
from data.load_fixtures import load_all, all_patients, get_ground_truth
from tools.trial_search import build_index


st.set_page_config(page_title="Trial Matching Agent", layout="wide", page_icon="🧬")


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource(show_spinner="Loading fixtures...")
def init_fixtures(snapshot: str):
    load_all(snapshot)
    build_index()
    return snapshot


def render_event(event: dict):
    """Render a single trace event into the current container."""
    t = event.get("type")
    iter_num = event.get("iter", "?")

    if t == "agent_thinking":
        content = event.get("content") or ""
        if content.strip():
            with st.chat_message("assistant", avatar="🧠"):
                st.caption(f"Turn {iter_num} · reasoning")
                st.markdown(content[:1200])

    elif t == "tool_call":
        with st.chat_message("user", avatar="🛠"):
            st.caption(f"Turn {iter_num} · calling tool")
            st.markdown(f"**`{event['name']}`**")
            args_str = json.dumps(event.get("args", {}), indent=2, default=str)
            if len(args_str) > 400:
                args_str = args_str[:400] + "\n... (truncated)"
            st.code(args_str, language="json")

    elif t == "tool_result":
        with st.chat_message("assistant", avatar="📥"):
            st.caption(f"Result from {event.get('name', '?')}")
            result_str = json.dumps(event.get("result", {}), indent=2, default=str)
            if len(result_str) > 700:
                result_str = result_str[:700] + "\n... (truncated)"
            st.code(result_str, language="json")

    elif t == "final":
        with st.chat_message("assistant", avatar="✅"):
            st.markdown("**Agent complete**")


def patient_label(p: dict) -> str:
    """Sidebar dropdown label for a patient."""
    gt = get_ground_truth(p["patient_id"])
    short = p["patient_id"][:8]
    if gt:
        ind = ", ".join(gt.get("indications", []))[:45]
        return f"{short}... [{gt.get('priority', '?')}] {ind}"
    return short


# ============================================================
# SESSION STATE
# ============================================================

if "snapshot" not in st.session_state:
    st.session_state.snapshot = "v1"
if "trace" not in st.session_state:
    st.session_state.trace = []
if "result" not in st.session_state:
    st.session_state.result = None

init_fixtures(st.session_state.snapshot)


# ============================================================
# UI
# ============================================================

st.title("🧬 Patient–Trial Matching Agent")
st.caption("Track 1 · Clinical Decision Support · Pfizer Medical Intelligence sub-track")


# ----- SIDEBAR -----

with st.sidebar:
    st.header("Demo Controls")

    patients = all_patients()
    selected_patient = st.selectbox(
        "Patient",
        options=patients,
        format_func=patient_label,
    )

    st.divider()

    new_snapshot = st.radio(
        "Trial criteria snapshot",
        options=["v1", "v2"],
        horizontal=True,
        help="Toggle to demonstrate continuous re-evaluation when trial criteria evolve.",
        index=0 if st.session_state.snapshot == "v1" else 1,
    )

    if new_snapshot != st.session_state.snapshot:
        st.session_state.snapshot = new_snapshot
        init_fixtures.clear()
        init_fixtures(new_snapshot)
        st.success(f"Loaded {new_snapshot}. Re-run to see updated matches.")

    st.divider()

    run = st.button("🔍 Find Matches", type="primary", use_container_width=True)

    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.trace = []
        st.session_state.result = None
        st.rerun()


# ----- MAIN LAYOUT: PATIENT (left) + TRACE (right) -----

col_patient, col_trace = st.columns([1, 1])

with col_patient:
    st.subheader("Patient")

    gt = get_ground_truth(selected_patient["patient_id"])
    if gt:
        c1, c2, c3 = st.columns(3)
        c1.metric("Priority", gt.get("priority", "?").upper())
        c2.metric("Indications", gt.get("indication_count", "?"))
        c3.metric("Multi-Indication", "✓" if gt.get("is_multi_indication") else "✗")
        st.caption(f"**GT indications:** {', '.join(gt.get('indications', []))}")
        st.caption(f"**GT expected actions:** {', '.join(gt.get('expected_actions', []))}")

    demo = selected_patient.get("demographics", {})
    st.caption(f"**Age:** {demo.get('age', '?')} · **Gender:** {demo.get('gender', '?')}")

    summary = selected_patient.get("summary", {})

    with st.expander("Active conditions"):
        for cond in summary.get("active_conditions", []):
            st.markdown(f"- {cond}")

    with st.expander("Current medications"):
        for med in summary.get("active_medications", []):
            st.markdown(f"- {med}")

    with st.expander("Full patient bundle"):
        st.json(selected_patient, expanded=False)

with col_trace:
    st.subheader("Agent Reasoning Trace")
    trace_box = st.container(height=700, border=True)


# ----- AGENT RUN -----

if run:
    st.session_state.trace = []
    st.session_state.result = None

    with trace_box:
        with st.spinner("Agent reasoning..."):

            def on_event(event):
                st.session_state.trace.append(event)
                render_event(event)

            try:
                result = run_agent(selected_patient, trace_callback=on_event)
                st.session_state.result = result
            except Exception as e:
                st.error(f"Agent crashed: {e}")

elif st.session_state.trace:
    # Re-render saved trace on page reload
    with trace_box:
        for event in st.session_state.trace:
            render_event(event)


# ----- RANKED OUTPUT -----

if st.session_state.result:
    st.divider()
    result = st.session_state.result

    st.subheader("🎯 Ranked Trial Matches")

    if "ranked_matches" in result and result["ranked_matches"]:
        for match in result["ranked_matches"]:
            with st.container(border=True):
                c_rank, c_main = st.columns([1, 6])

                with c_rank:
                    st.markdown(f"## #{match.get('rank', '?')}")
                    score = match.get("score", 0)
                    if isinstance(score, (int, float)):
                        st.metric("Score", f"{score:.2f}")

                with c_main:
                    st.markdown(f"**{match.get('trial_id', '?')}**")
                    st.markdown(match.get("title", ""))
                    summary_text = match.get("summary", "")
                    if summary_text:
                        st.markdown(f"_{summary_text}_")

                    verdict = match.get("verdict_detail", {})
                    overall = verdict.get("overall", "?")
                    color_map = {"PASS": "🟢", "FAIL": "🔴", "PARTIAL": "🟡"}
                    st.markdown(f"{color_map.get(overall, '⚪')} **Overall:** {overall}")

                    with st.expander("Per-criterion verdicts"):
                        for cv in verdict.get("criteria_verdicts", []):
                            v = cv.get("verdict", "?")
                            icon = {"PASS": "✅", "FAIL": "❌", "UNKNOWN": "⚠️"}.get(v, "❓")
                            criterion_text = cv.get("criterion", "")
                            if len(criterion_text) > 220:
                                criterion_text = criterion_text[:220] + "..."
                            st.markdown(f"{icon} **{v}** — {criterion_text}")
                            rationale = cv.get("rationale", "")
                            if rationale:
                                st.caption(rationale)

        # Missing data summary
        if result.get("missing_data_summary"):
            st.divider()
            st.warning("⚠️ Missing data — surface to clinician for next steps")
            for item in result["missing_data_summary"]:
                blocks = item.get("blocks_trials", [])
                blocks_str = ", ".join(blocks) if blocks else ""
                line = f"- **{item.get('field', '?')}**"
                if blocks_str:
                    line += f" blocks: {blocks_str}"
                st.markdown(line)

        # Cross-indication alerts
        if result.get("cross_indication_alerts"):
            st.divider()
            st.info("⊕ Cross-indication matches surfaced")
            for alert in result["cross_indication_alerts"]:
                st.markdown(
                    f"- **{alert.get('trial_id', '?')}** "
                    f"({alert.get('indication', '')}): {alert.get('reason', '')}"
                )

    elif "final_text" in result:
        st.markdown(result["final_text"])

    else:
        st.json(result)