"""Streamlit demo app for the patient-trial matching agent.

Drop into your repo as streamlit_app/app.py
Run with: streamlit run streamlit_app/app.py
"""
import base64
import streamlit as st
import streamlit.components.v1 as components

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import run_agent
from agent.pipeline import enrich_patient_dict
from data.load_fixtures import load_all, all_patients, get_ground_truth
from tools.trial_search import build_index


st.set_page_config(page_title="Trial Matching Agent", layout="wide", page_icon="🧬")


# ============================================================
# LOGO
# ============================================================

def _load_logo_b64() -> str:
    p = Path(__file__).parent / "assets" / "covalence_logo.png"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

_LOGO_B64 = _load_logo_b64()


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource(show_spinner="Loading fixtures...")
def init_fixtures():
    load_all()
    build_index()


_STAGES = ["Searching trials", "Parsing criteria", "Checking eligibility", "Validating", "Ranking"]
_TOOL_TO_STAGE = {
    "trial_search": 0,
    "parse_criteria": 1,
    "check_eligibility": 2,
    "validate_verdicts": 3,
    "rank_with_rationale": 4,
}


def _plain_english(event: dict) -> str:
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


def _classify_outcome(result: dict) -> str:
    if not result:
        return "error"
    if result.get("outcome") == "no_match":
        return "no_match"
    if result.get("outcome") == "error":
        return "error"
    matches = result.get("ranked_matches", [])
    if isinstance(matches, list) and len(matches) > 0:
        return "success"
    if "ranked_matches" in result:
        return "no_match"
    return "error"


def patient_label(p: dict) -> str:
    gt = get_ground_truth(p["patient_id"])
    short = p["patient_id"][:8]
    if gt:
        ind = ", ".join(gt.get("indications", []))[:45]
        return f"{short}... [{gt.get('priority', '?')}] {ind}"
    return short


def patient_tab_label(pid: str) -> str:
    return pid[:8] + "..."


def render_patient_info(patient: dict):
    st.subheader("Patient")
    gt = get_ground_truth(patient["patient_id"])
    if gt:
        c1, c2, c3 = st.columns(3)
        c1.metric("Priority", gt.get("priority", "?").upper())
        c2.metric("Indications", gt.get("indication_count", "?"))
        c3.metric("Multi-Indication", "✓" if gt.get("is_multi_indication") else "✗")
        st.caption(f"**GT indications:** {', '.join(gt.get('indications', []))}")
        st.caption(f"**GT expected actions:** {', '.join(gt.get('expected_actions', []))}")

    demo = patient.get("demographics", {})
    st.caption(f"**Age:** {demo.get('age', '?')} · **Gender:** {demo.get('gender', '?')}")

    with st.expander("Active conditions"):
        for cond in patient.get("conditions", []):
            if cond.get("active"):
                st.markdown(f"- {cond.get('description', cond.get('code', '?'))}")

    with st.expander("Current medications"):
        for med in patient.get("medications", []):
            st.markdown(f"- {med.get('description', med.get('code', '?'))}")

    with st.expander("Full patient bundle"):
        st.json(patient, expanded=False)


def render_right_pane(session: dict):
    result = session.get("result")
    if not result:
        return

    st.divider()
    st.subheader("🎯 Ranked Trial Matches")

    outcome = _classify_outcome(result)

    if outcome == "no_match":
        st.info("No clinical trials matched this patient's indication profile.")

    elif outcome == "error":
        st.warning("The agent was unable to complete trial matching for this patient.")

    elif outcome == "success":
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
                            icon = "🔧" if resolvable_flag else {"PASS": "✅", "FAIL": "❌", "UNKNOWN": "⚠️"}.get(v, "❓")
                            criterion_text = cv.get("criterion", "")
                            if len(criterion_text) > 220:
                                criterion_text = criterion_text[:220] + "..."
                            label = f"{v} (resolvable)" if resolvable_flag else v
                            st.markdown(f"{icon} **{label}** — {criterion_text}")
                            if cv.get("rationale"):
                                st.caption(cv["rationale"])

        if result.get("missing_data_summary"):
            st.divider()
            st.warning("⚠️ Missing data — surface to clinician for next steps")
            for item in result["missing_data_summary"]:
                blocks = item.get("blocks_trials", [])
                line = f"- **{item.get('field', '?')}**"
                if blocks:
                    line += f" blocks: {', '.join(blocks)}"
                st.markdown(line)

        if result.get("cross_indication_alerts"):
            st.divider()
            st.info("⊕ Cross-indication matches surfaced")
            for alert in result["cross_indication_alerts"]:
                st.markdown(
                    f"- **{alert.get('trial_id', '?')}** "
                    f"({alert.get('indication', '')}): {alert.get('reason', '')}"
                )


# ============================================================
# INIT
# ============================================================

init_fixtures()
patients = all_patients()
patients_by_id = {p["patient_id"]: p for p in patients}


# ============================================================
# SESSION STATE
# ============================================================

if "open_patients" not in st.session_state:
    st.session_state.open_patients = []
if "patient_sessions" not in st.session_state:
    st.session_state.patient_sessions = {}
if "run_for" not in st.session_state:
    st.session_state.run_for = None
if "sidebar_selected_pid" not in st.session_state:
    st.session_state.sidebar_selected_pid = (
        patients[0]["patient_id"] if patients else None
    )


# ============================================================
# SIDEBAR CSS
# ============================================================

_SIDEBAR_CSS = """
<style>
/* ── Base ── */
[data-testid="stSidebar"] { background: #ffffff; }
[data-testid="stSidebarContent"] { padding-top: 0 !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0 !important; }

/* Kill the flex gap Streamlit puts between every sidebar widget */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* Zero out all element-container margins in sidebar */
[data-testid="stSidebar"] [data-testid="element-container"] {
    margin: 0 !important;
    padding: 0 !important;
}

/* Completely collapse wrapper divs for hidden trigger buttons
   (the button itself is display:none but the parent divs still take space) */
[data-testid="stSidebar"] [data-testid="element-container"]:has(button[data-testid="stBaseButton-primary"]) {
    display: none !important;
}

/* All iframes in sidebar: no border */
[data-testid="stSidebar"] iframe {
    border: none !important;
    display: block !important;
}

/* ── Find Matches button (only secondary button in sidebar) ── */
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    background: #f0f2f5 !important;
    border: none !important;
    border-radius: 22px !important;
    padding: 9px 20px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #1a1a1a !important;
    box-shadow: none !important;
    outline: none !important;
    transition: background 0.15s ease !important;
    justify-content: center !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
    background: #e4e6eb !important;
}

/* ── Hidden trigger buttons (type="primary"): invisible, zero-size ── */
/* JS clicks these to communicate patient selection back to Python */
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    display: none !important;
}
</style>
"""


# ============================================================
# BRANDING HTML (iframe: has JS for collapse toggle)
# ============================================================

def _branding_html(logo_b64: str) -> str:
    logo_tag = (
        f"<img class='logo' src='data:image/png;base64,{logo_b64}' alt='Covalence' />"
        if logo_b64 else "<span style='font-size:22px;line-height:1'>🧬</span>"
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#ffffff;overflow:hidden;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
.row{{display:flex;align-items:center;justify-content:space-between;
      padding:14px 14px 12px 14px;height:58px}}
.left{{display:flex;align-items:center;gap:10px}}
.logo{{width:28px;height:28px;object-fit:contain}}
.name{{font-size:17px;font-weight:600;color:#0d0d0d;letter-spacing:-0.3px}}
.toggle{{all:unset;display:flex;align-items:center;justify-content:center;
         width:30px;height:30px;border:1.5px solid #d0d0d0;border-radius:7px;
         cursor:pointer;background:#fff;color:#555;flex-shrink:0;
         transition:background 0.12s ease}}
.toggle:hover{{background:#f5f5f5}}
</style>
</head>
<body>
<div class="row">
  <div class="left">
    {logo_tag}
    <span class="name">Covalence</span>
  </div>
  <button class="toggle" onclick="collapseSidebar()" title="Collapse sidebar">
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2"/>
      <line x1="9" y1="3" x2="9" y2="21"/>
      <polyline points="15 8 19 12 15 16"/>
    </svg>
  </button>
</div>
<script>
function collapseSidebar() {{
  var p = window.parent;
  // The Streamlit sidebar toggle is inside stSidebarContent but outside stSidebarUserContent
  var sidebarContent = p.document.querySelector('[data-testid="stSidebarContent"]');
  if (sidebarContent) {{
    var userContent = sidebarContent.querySelector('[data-testid="stSidebarUserContent"]');
    var btns = sidebarContent.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) {{
      if (!userContent || !userContent.contains(btns[i])) {{
        btns[i].click();
        return;
      }}
    }}
  }}
  // Fallback selectors (vary by Streamlit version)
  var selectors = [
    '[data-testid="collapsedControl"]',
    'button[aria-label="Close sidebar"]',
    'button[aria-label="Collapse sidebar"]',
  ];
  for (var j = 0; j < selectors.length; j++) {{
    var btn = p.document.querySelector(selectors[j]);
    if (btn) {{ btn.click(); return; }}
  }}
}}
</script>
</body>
</html>"""


# ============================================================
# PATIENT LIST HTML (rendered in iframe: pure HTML, no Streamlit chrome)
# ============================================================

def _patient_list_html(patients: list, open_patients: list, selected_pid: str) -> str:
    items = []
    for i, p in enumerate(patients):
        pid = p["patient_id"]
        raw_label = patient_label(p)
        label = raw_label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        data_label = raw_label.replace('"', "&quot;")
        has_tab = pid in open_patients
        is_selected = (pid == selected_pid) and not has_tab

        if has_tab:
            cls = "item disabled"
            onclick = ""
        elif is_selected:
            cls = "item selected"
            onclick = f'onclick="sel({i})"'
        else:
            cls = "item"
            onclick = f'onclick="sel({i})"'

        items.append(f'<div class="{cls}" data-label="{data_label}" {onclick}>{label}</div>')

    items_html = "\n".join(items)

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;width:100%;background:#ffffff;overflow:hidden;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
.wrap{{display:flex;flex-direction:column;height:100%;overflow:hidden}}

/* ── Search bar ── */
.search-bar{{
  display:flex;align-items:center;gap:9px;
  padding:7px 14px;border-radius:20px;
  margin:2px 0 24px 0;
  cursor:text;transition:background 0.15s ease;flex-shrink:0;
}}
.search-bar:focus-within{{background:#f0f2f5}}
.search-icon{{flex-shrink:0;display:flex;align-items:center;color:#888}}
.search-input{{
  border:none;outline:none;background:transparent;
  font-size:13.5px;color:#1a1a1a;flex:1;min-width:0;
  font-family:inherit;cursor:text;
}}
.search-input::placeholder{{color:#888}}
.clear-btn{{
  display:none;align-items:center;justify-content:center;
  width:18px;height:18px;border-radius:50%;border:none;
  background:transparent;cursor:pointer;color:#888;
  font-size:13px;line-height:1;flex-shrink:0;padding:0;
}}
.clear-btn:hover{{background:#e0e0e0;color:#444}}

/* ── Patient list ── */
.scroll{{
  flex:1;overflow-y:auto;overflow-x:hidden;
  scrollbar-width:thin;scrollbar-color:#d0d0d0 transparent;
  padding:2px 0;
}}
.scroll::-webkit-scrollbar{{width:4px}}
.scroll::-webkit-scrollbar-thumb{{background:#d0d0d0;border-radius:4px}}
.scroll::-webkit-scrollbar-track{{background:transparent}}
.item{{
  display:block;width:100%;padding:7px 14px;border-radius:18px;
  cursor:pointer;font-size:13.5px;font-weight:400;color:#1a1a1a;
  line-height:1.4;transition:background 0.1s ease;background:transparent;
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
  user-select:none;border:none;outline:none;text-align:left;
}}
.item:not(.disabled):not(.selected):hover{{background:#f0f2f5}}
.item.selected{{background:#e2e5e9;font-weight:500;color:#0d0d0d}}
.item.selected:hover{{background:#d7dce2}}
.item.disabled{{color:#bbb;cursor:default;pointer-events:none}}
</style>
</head>
<body>
<div class="wrap">
  <div class="search-bar" onclick="document.getElementById('q').focus()">
    <span class="search-icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
    </span>
    <input id="q" class="search-input" type="text" placeholder="Search patient"
           oninput="onSearch()" autocomplete="off" spellcheck="false" />
    <button id="clr" class="clear-btn" onclick="clearSearch(event)" title="Clear">✕</button>
  </div>
  <div class="scroll">
{items_html}
  </div>
</div>
<script>
function onSearch() {{
  var val = document.getElementById('q').value;
  document.getElementById('clr').style.display = val ? 'flex' : 'none';
  filter(val);
}}
function clearSearch(e) {{
  e.stopPropagation();
  var q = document.getElementById('q');
  q.value = '';
  document.getElementById('clr').style.display = 'none';
  filter('');
  q.focus();
}}
function filter(query) {{
  var tokens = query.trim().toLowerCase().split(/\s+/).filter(function(t){{return t.length>0;}});
  document.querySelectorAll('.item').forEach(function(el) {{
    if (!tokens.length) {{ el.style.display=''; return; }}
    var text = (el.getAttribute('data-label')||el.textContent).toLowerCase();
    el.style.display = tokens.every(function(tok){{return text.indexOf(tok)>=0;}}) ? '' : 'none';
  }});
}}
function sel(idx) {{
  var p = window.parent;
  var triggers = p.document.querySelectorAll(
    '[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]'
  );
  if (triggers[idx]) triggers[idx].click();
}}
</script>
</body>
</html>"""


# ============================================================
# UI
# ============================================================

st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)

st.title("🧬 Patient–Trial Matching Agent")
st.caption("Track 1 · Clinical Decision Support · Pfizer Medical Intelligence sub-track")


# ----- SIDEBAR -----

with st.sidebar:
    # Branding row (iframe: has JS for collapse toggle)
    components.html(_branding_html(_LOGO_B64), height=58, scrolling=False)

    # Find Matches pill button
    if st.button("Find matches", use_container_width=True, key="find_matches_btn"):
        pid = st.session_state.sidebar_selected_pid
        if pid and pid not in st.session_state.open_patients:
            patient = patients_by_id[pid]
            st.session_state.open_patients.append(pid)
            st.session_state.patient_sessions[pid] = {
                "patient": patient,
                "trace": [],
                "result": None,
            }
            st.session_state.run_for = pid

    # Hidden trigger buttons — one per patient, type="primary" so CSS hides them.
    # The patient list iframe clicks these via JS to communicate selection to Python.
    for i, p in enumerate(patients):
        pid = p["patient_id"]
        if st.button(" ", key=f"trig_{i}", type="primary", use_container_width=False):
            st.session_state.sidebar_selected_pid = pid
            st.rerun()

    # Visible patient list: pure HTML iframe — no Streamlit button chrome at all
    components.html(
        _patient_list_html(
            patients,
            st.session_state.open_patients,
            st.session_state.sidebar_selected_pid,
        ),
        height=760,
        scrolling=False,
    )


# ----- MAIN: PATIENT TABS -----

if not st.session_state.open_patients:
    st.info("Select a patient in the sidebar and click **Find matches** to open a tab.")
else:
    tab_labels = [patient_tab_label(pid) for pid in st.session_state.open_patients]
    tabs = st.tabs(tab_labels)

    for tab, pid in zip(tabs, st.session_state.open_patients):
        with tab:
            session = st.session_state.patient_sessions[pid]
            patient = session["patient"]

            col_patient, col_trace = st.columns([1, 1])

            with col_patient:
                render_patient_info(patient)

            with col_trace:
                st.subheader("Agent Reasoning Trace")
                progress_slot = st.empty()
                trace_box = st.container(height=620, border=True)

                if st.session_state.run_for == pid:
                    with trace_box:
                        with st.spinner("Agent reasoning..."):

                            def on_event(event, _session=session, _slot=progress_slot):
                                _session["trace"].append(event)
                                with _slot.container():
                                    render_progress(_session["trace"])
                                render_event(event)

                            try:
                                enriched = enrich_patient_dict(patient)
                                result = run_agent(enriched, trace_callback=on_event)
                                session["result"] = result
                            except Exception as e:
                                session["result"] = {"outcome": "error", "reason": str(e)}

                    st.session_state.run_for = None

                else:
                    if session["trace"]:
                        with progress_slot.container():
                            render_progress(session["trace"])
                    with trace_box:
                        for event in session["trace"]:
                            render_event(event)

            render_right_pane(session)
