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
from agent.pipeline import enrich_patient_dict
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


_STAGES = ["Searching trials", "Parsing criteria", "Checking eligibility", "Validating", "Ranking"]
_TOOL_TO_STAGE = {
    "trial_search": 0,
    "parse_criteria": 1,
    "check_eligibility": 2,
    "validate_verdicts": 3,
    "rank_with_rationale": 4,
}


def _plain_english(event: dict) -> str:
    """Return a one-sentence plain-English summary of a tool event."""
    name = event.get("name", "")
    args = event.get("args", {})
    result = event.get("result", {})
    t = event.get("type")

    if t == "tool_call":
        if name == "trial_search":
            return f"Searching for trials matching: *{args.get('query', '')}*"
        if name == "parse_criteria":
            return f"Parsing eligibility criteria for **{args.get('trial_id', '')}**"
        if name == "check_eligibility":
            return f"Checking patient eligibility for **{args.get('trial_id', '')}**"
        if name == "validate_verdicts":
            return "Validating eligibility verdicts for consistency"
        if name == "rank_with_rationale":
            return "Ranking matched trials and writing rationale"

    if t == "tool_result":
        if name == "trial_search":
            n = len(result.get("trials", []))
            return f"Found **{n}** candidate trial{'s' if n != 1 else ''}"
        if name == "parse_criteria":
            inc = len(result.get("inclusion", []))
            exc = len(result.get("exclusion", []))
            return f"Extracted **{inc}** inclusion and **{exc}** exclusion criteria"
        if name == "check_eligibility":
            overall = result.get("overall", "?")
            trial_id = result.get("trial_id", "")
            verdicts = result.get("criteria_verdicts", [])
            passes = sum(1 for v in verdicts if v.get("verdict") == "PASS")
            total = len(verdicts)
            icon = {"PASS": "🟢", "FAIL": "🔴", "PARTIAL": "🟡"}.get(overall, "⚪")
            return f"{icon} **{overall}** for {trial_id} — {passes}/{total} criteria met"
        if name == "validate_verdicts":
            issues = result.get("issues", [])
            if issues:
                return f"Found **{len(issues)}** verdict issue{'s' if len(issues) != 1 else ''} to review"
            return "All verdicts validated — no issues found"
        if name == "rank_with_rationale":
            n = len(result.get("ranked_matches", []))
            return f"Ranked **{n}** trial{'s' if n != 1 else ''} — results ready"

    return ""


def render_progress(trace: list) -> None:
    """Render a named stage progress bar based on the tools called so far."""
    reached = -1
    for event in trace:
        stage = _TOOL_TO_STAGE.get(event.get("name", ""), -1)
        if stage > reached:
            reached = stage

    cols = st.columns(len(_STAGES))
    for i, (col, label) in enumerate(zip(cols, _STAGES)):
        if i < reached:
            col.success(f"✓ {label}")
        elif i == reached:
            col.info(f"▶ {label}")
        else:
            col.markdown(f"<span style='color:grey'>○ {label}</span>", unsafe_allow_html=True)


def render_event(event: dict):
    """Render a single trace event as a plain-English summary."""
    t = event.get("type")
    summary = _plain_english(event)

    if t == "agent_thinking":
        content = event.get("content") or ""
        if content.strip():
            with st.expander(f"Agent reasoning (turn {event.get('iter', '?')})", expanded=False):
                st.markdown(content[:1200])

    elif t == "tool_call" and summary:
        st.markdown(f"⟶ {summary}")

    elif t == "tool_result" and summary:
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{summary}", unsafe_allow_html=True)

    elif t == "final":
        st.success("Analysis complete")


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

    with st.expander("Active conditions"):
        for cond in selected_patient.get("conditions", []):
            if cond.get("active"):
                st.markdown(f"- {cond.get('description', cond.get('code', '?'))}")

    with st.expander("Current medications"):
        for med in selected_patient.get("medications", []):
            st.markdown(f"- {med.get('description', med.get('code', '?'))}")

    with st.expander("Full patient bundle"):
        st.json(selected_patient, expanded=False)

with col_trace:
    st.subheader("Agent Reasoning Trace")
    progress_slot = st.empty()
    trace_box = st.container(height=620, border=True)


# ----- AGENT RUN -----

if run:
    st.session_state.trace = []
    st.session_state.result = None

    with trace_box:
        with st.spinner("Agent reasoning..."):

            def on_event(event):
                st.session_state.trace.append(event)
                with progress_slot.container():
                    render_progress(st.session_state.trace)
                render_event(event)

            try:
                enriched = enrich_patient_dict(selected_patient)
                result = run_agent(enriched, trace_callback=on_event)
                st.session_state.result = result
            except Exception as e:
                st.error(f"Agent crashed: {e}")

elif st.session_state.trace:
    with progress_slot.container():
        render_progress(st.session_state.trace)
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
                    adj_score = match.get("adjusted_score")
                    if adj_score is not None and isinstance(adj_score, (int, float)):
                        st.metric("Adjusted", f"{adj_score:.2f}", help="Score if care gaps addressed")

                with c_main:
                    st.markdown(f"**{match.get('trial_id', '?')}**")
                    st.markdown(match.get("title", ""))

                    match_pct = match.get("match_pct", "")
                    adj_pct = match.get("adjusted_match_pct", "")
                    if match_pct:
                        st.markdown(f"**Match:** {match_pct}")
                    if adj_pct:
                        st.markdown(f"**Adjusted:** {adj_pct}")

                    summary_text = match.get("summary", "")
                    if summary_text:
                        st.markdown(f"_{summary_text}_")

                    verdict = match.get("verdict_detail", {})
                    overall = verdict.get("overall", "?")
                    color_map = {"PASS": "🟢", "FAIL": "🔴", "PARTIAL": "🟡"}
                    st.markdown(f"{color_map.get(overall, '⚪')} **Overall:** {overall}")

                    resolvable = match.get("resolvable_criteria", [])
                    if resolvable:
                        with st.expander(f"🔧 {len(resolvable)} resolvable criterion — care gap fix unlocks eligibility"):
                            for rc in resolvable:
                                st.markdown(f"**Criterion:** {rc.get('criterion', '')}")
                                st.markdown(f"**Care gap:** {rc.get('care_gap', '')}")
                                st.success(f"Action: {rc.get('action', '')}")

                    with st.expander("Per-criterion verdicts"):
                        for cv in verdict.get("criteria_verdicts", []):
                            v = cv.get("verdict", "?")
                            resolvable_flag = cv.get("resolvable", False)
                            if resolvable_flag:
                                icon = "🔧"
                            else:
                                icon = {"PASS": "✅", "FAIL": "❌", "UNKNOWN": "⚠️"}.get(v, "❓")
                            criterion_text = cv.get("criterion", "")
                            if len(criterion_text) > 220:
                                criterion_text = criterion_text[:220] + "..."
                            label = f"{v} (resolvable)" if resolvable_flag else v
                            st.markdown(f"{icon} **{label}** — {criterion_text}")
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