import json
import os
import streamlit as st
import pandas as pd
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

from data.load_fixtures import all_patients, all_trials, get_trial

load_dotenv()
client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL"), api_key=os.getenv("NEBIUS_API_KEY"))
MODEL = os.getenv("MODEL_RECOMMEND", "meta-llama/Llama-3.3-70B-Instruct")


SITE_RECOMMENDATION_PROMPT = """You are a clinical trial site planning advisor. Given a trial's basic profile and the geographic distribution of candidate patients in our cohort, recommend up to 3 distinct locations to open enrollment sites.

For each recommended location, provide:
- The city and state
- The candidate patient count (must match the input distribution exactly)
- A 1-2 sentence rationale considering candidate pool size, urban density, proximity to academic medical centers, and clinical trial infrastructure

CRITICAL RULES:
1. Each recommended location MUST be a UNIQUE city/state combination. Never repeat the same location across multiple ranks.
2. The candidate_count for each location must match exactly what appears in the geographic_distribution input. Do not invent or duplicate numbers.
3. Rank locations by a combination of candidate density AND strategic value (e.g., proximity to academic medical centers).
4. If fewer than 3 distinct locations with meaningful candidate density exist in the input, return fewer recommendations. Better to return 1 or 2 strong recommendations than to pad with weak duplicates.

Return ONLY a JSON object with this shape:

{
  "recommendations": [
    {
      "rank": 1,
      "location": "Houston, TX",
      "candidate_count": 7,
      "rationale": "..."
    }
  ],
  "summary": "1-2 sentence overall recommendation"
}"""


def filter_candidates(trial: dict) -> list[dict]:
    """Deterministic pre-filter — no LLM needed."""
    candidates = []
    min_age = int(trial.get("min_age", "0 Years").split()[0]) if trial.get("min_age") else 0
    max_age_raw = trial.get("max_age")
    max_age = int(max_age_raw.split()[0]) if max_age_raw else 200
    trial_sex = trial.get("sex", "ALL").upper()
    trial_conditions = [c.lower() for c in trial.get("conditions", [])]
    
    for patient in all_patients():
        demo = patient.get("demographics", {})
        age = demo.get("age", 0)
        sex = demo.get("gender", "").upper()
        
        if not (min_age <= age <= max_age):
            continue
        if trial_sex != "ALL":
            if trial_sex == "FEMALE" and sex != "F":
                continue
            if trial_sex == "MALE" and sex != "M":
                continue
        
        patient_conditions = [
            c.get("description", "").lower()
            for c in patient.get("conditions", [])
            if c.get("active")
        ]
        match = any(
            any(word in pc for pc in patient_conditions)
            for tc in trial_conditions
            for word in tc.split()
            if len(word) > 4
        )
        if match:
            candidates.append(patient)
    return candidates


def filter_candidates_by_keywords(keywords: str) -> list[dict]:
    """Filter patients whose active conditions match any of the given keywords."""
    keyword_list = [k.strip().lower() for k in keywords.split() if len(k.strip()) > 3]
    if not keyword_list:
        return []
    
    candidates = []
    for patient in all_patients():
        patient_conditions = [
            c.get("description", "").lower()
            for c in patient.get("conditions", [])
            if c.get("active")
        ]
        # Match if any keyword appears in any active condition
        match = any(
            kw in pc
            for kw in keyword_list
            for pc in patient_conditions
        )
        if match:
            candidates.append(patient)
    return candidates


def recommend_sites_llm(trial_profile: dict, candidates: list[dict]) -> dict:
    """LLM-generated site recommendations with rationale."""
    # Aggregate geographic distribution
    location_counts = Counter()
    for p in candidates:
        demo = p.get("demographics", {})
        city = demo.get("city", "Unknown")
        state = demo.get("state", "Unknown")
        location_counts[f"{city}, {state}"] += 1
    
    user_message = json.dumps({
        **trial_profile,
        "total_candidates": len(candidates),
        "geographic_distribution": dict(location_counts.most_common(15)),
    }, indent=2)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SITE_RECOMMENDATION_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=3000,
    )    
    text = response.choices[0].message.content or ""
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"error": "failed to parse", "raw": text[:500]}


def render():
    st.title("🗺 Site Recommendation")
    st.caption("AI-powered recommendation for where to open trial enrollment sites")

    mode = st.radio(
        "Search by",
        options=["Existing trial", "Keywords / indication"],
        horizontal=True,
    )

    candidates = []
    trial_profile = {}

    if mode == "Existing trial":
        trials = all_trials()
        trial_options = {f"{t['nct_id']}: {t.get('title', '')[:80]}": t['nct_id'] for t in trials}
        selection = st.selectbox("Select trial", options=list(trial_options.keys()))
        nct_id = trial_options[selection]
        trial = get_trial(nct_id)

        st.markdown(f"**Conditions:** {', '.join(trial.get('conditions', []))}")
        st.markdown(f"**Age range:** {trial.get('min_age', '?')} – {trial.get('max_age', 'no max')}")
        st.markdown(f"**Sex:** {trial.get('sex', 'ALL')}")

        if st.button("Generate site recommendations", type="primary"):
            with st.spinner("Filtering candidate patients..."):
                candidates = filter_candidates(trial)
            trial_profile = {
                "trial_id": trial.get("nct_id"),
                "trial_conditions": trial.get("conditions", []),
                "trial_title": trial.get("title", ""),
            }

    else:  # Keywords mode
        keywords = st.text_input(
            "Enter keywords or indication",
            placeholder="e.g., diabetes type 2 obesity, or HER2+ breast cancer",
            help="Space-separated keywords. Only words longer than 3 characters are matched.",
        )

        if st.button("Generate site recommendations", type="primary") and keywords:
            with st.spinner("Filtering candidate patients..."):
                candidates = filter_candidates_by_keywords(keywords)
            trial_profile = {
                "search_keywords": keywords,
                "trial_conditions": [keywords],
                "trial_title": f"Custom search: {keywords}",
            }

    # Shared output rendering — runs once candidates exist
    if not trial_profile:
        return

    st.success(f"Found {len(candidates)} candidate patients")

    if not candidates:
        st.warning("No patients in cohort match this search.")
        return

    # Raw geographic distribution
    location_counts = Counter()
    for p in candidates:
        demo = p.get("demographics", {})
        location_counts[f"{demo.get('city')}, {demo.get('state')}"] += 1

    with st.expander("Raw geographic distribution"):
        df = pd.DataFrame(
            sorted(location_counts.items(), key=lambda x: -x[1])[:15],
            columns=["Location", "Candidates"]
        )
        st.dataframe(df)

    # LLM recommendation
    st.subheader("🎯 Recommended sites")
    with st.spinner("AI analyzing candidate pool..."):
        result = recommend_sites_llm(trial_profile, candidates)

    if "error" in result:
        st.error(result["error"])
        return

    for rec in result.get("recommendations", []):
        with st.container(border=True):
            col_rank, col_main = st.columns([1, 6])
            with col_rank:
                st.markdown(f"## #{rec.get('rank', '?')}")
                st.metric("Candidates", rec.get("candidate_count", 0))
            with col_main:
                st.markdown(f"### {rec.get('location', '?')}")
                st.write(rec.get("rationale", ""))

    if result.get("summary"):
        st.info(result["summary"])