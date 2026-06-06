"""Score and rank verdict objects. Deterministic scoring + LLM rationale paragraphs."""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from agent.prompts import RANK_WITH_RATIONALE_PROMPT
from data.load_fixtures import get_trial

load_dotenv()
client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL"), api_key=os.getenv("NEBIUS_API_KEY"))
MODEL = os.getenv("MODEL_RANK", "meta-llama/Llama-3.3-70B-Instruct")


def _compute_score(verdict: dict) -> float:
    """Deterministic scoring: pass rate, penalize unknowns, big penalty if any inclusion FAIL or exclusion match."""
    cvs = verdict.get("criteria_verdicts", [])
    if not cvs:
        return 0.0
    pass_count = sum(1 for c in cvs if c.get("verdict") == "PASS")
    unknown_count = sum(1 for c in cvs if c.get("verdict") == "UNKNOWN")
    total = len(cvs)
    any_blocking_fail = any(c.get("verdict") == "FAIL" for c in cvs)

    score = (pass_count / total) - 0.3 * (unknown_count / total)
    if any_blocking_fail:
        score -= 1.0
    return round(score, 3)


def rank_with_rationale(patient_id: str, verdict_objects: list[dict]) -> dict:

    if isinstance(verdict_objects, str):
        try:
            verdict_objects = json.loads(verdict_objects)
        except json.JSONDecodeError:
            return {"error": "verdict_objects was a malformed string", "patient_id": patient_id}
    if not isinstance(verdict_objects, list):
        return {"error": "verdict_objects must be a list", "patient_id": patient_id}

    # Step 1: deterministic scoring per trial
    scored = []
    for v in verdict_objects:
        scored.append({
            "trial_id": v.get("trial_id", "unknown"),
            "score": _compute_score(v),
            "verdict": v,
        })

    # Step 2: sort descending by score, keep top 3
    scored.sort(key=lambda x: -x["score"])
    top = scored[:3]

    # Step 3: build a slim LLM input — omit full criteria_verdicts to reduce output tokens.
    # We inject verdict_detail back from the original objects after the LLM responds.
    verdict_lookup = {t["trial_id"]: t["verdict"] for t in top}
    user_message = json.dumps({
        "patient_id": patient_id,
        "top_candidates": [
            {
                "trial_id": t["trial_id"],
                "title": (get_trial(t["trial_id"]) or {}).get("title", ""),
                "conditions": (get_trial(t["trial_id"]) or {}).get("conditions", []),
                "score": t["score"],
                "pass_count": t["verdict"].get("pass_count", 0),
                "fail_count": t["verdict"].get("fail_count", 0),
                "unknown_count": t["verdict"].get("unknown_count", 0),
                "resolvable_count": t["verdict"].get("resolvable_count", 0),
                "total_criteria": t["verdict"].get("total_criteria", 0),
                "overall": t["verdict"].get("overall", "?"),
                "resolvable_criteria": [
                    {"criterion": v["criterion"][:100], "rationale": v.get("rationale", "")}
                    for v in t["verdict"].get("criteria_verdicts", [])
                    if v.get("resolvable")
                ],
            }
            for t in top
        ],
    }, indent=2)

    # Step 4: LLM call to write rationale + cross-indication flags (no verdict_detail in output).
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RANK_WITH_RATIONALE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        result = _safe_json_parse(response.choices[0].message.content)
        # Inject verdict_detail from original objects so the UI can show per-criterion verdicts.
        for match in result.get("ranked_matches", []):
            tid = match.get("trial_id")
            if tid in verdict_lookup:
                match["verdict_detail"] = verdict_lookup[tid]
        return result
    except Exception as e:
        return {"error": f"LLM call failed: {e}", "patient_id": patient_id}


def _safe_json_parse(text: str) -> dict:
    if text is None or text == "":
        return {"error": "LLM returned empty content", "ranked_matches": []}
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"error": "failed to parse", "raw": text[:500]}