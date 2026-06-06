"""Evaluate a patient against a single trial's structured criteria."""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from agent.prompts import CHECK_ELIGIBILITY_PROMPT
from data.load_fixtures import get_trial, get_patient  # add get_patient
from tools.parse_criteria import parse_criteria


#load environment variables
load_dotenv()
#initialize OpenAI client
client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL_TOKENFACTORY", os.getenv("NEBIUS_BASE_URL")), api_key=os.getenv("NEBIUS_API_KEY"))
#set model, default to Llama 3.3 70B Instruct unless specified in environment variables
MODEL = os.getenv("MODEL_CHECK", "meta-llama/Llama-3.3-70B-Instruct")

#trim the patient bundle to what check_eligibility actually needs to save on tokens and reduce the size of the input data
def _trim_patient(patient: dict) -> dict:
    """Reduce patient bundle to what check_eligibility actually needs."""
    # Group observations by LOINC code sorted oldest-to-newest, then take the
    # last 3 per code so the LLM can see trends (e.g. eGFR declining over time).
    from collections import defaultdict
    obs_by_code: dict[str, list] = defaultdict(list)
    for obs in sorted(patient.get("observations", []), key=lambda o: o.get("date", "")):
        code = obs.get("code", "")
        if code and obs.get("category") in ("laboratory", "vital-signs"):
            obs_by_code[code].append(obs)
    recent_obs = []
    for readings in obs_by_code.values():
        recent_obs.extend(readings[-3:])

    return {
        "patient_id": patient.get("patient_id"),
        "demographics": patient.get("demographics", {}),
        "active_conditions": [c for c in patient.get("conditions", []) if c.get("active")],
        "current_medications": [m for m in patient.get("medications", []) if m.get("active")],
        "recent_observations": recent_obs,
    }

_eligibility_cache: dict[tuple, dict] = {}


#define check_eligibility function to evaluate a patient against a single trial's structured criteria
def check_eligibility(patient_id: str, trial_id: str) -> dict:
    cache_key = (patient_id, trial_id)
    if cache_key in _eligibility_cache:
        return _eligibility_cache[cache_key]

    #get the patient from the patient_search tool
    patient = get_patient(patient_id)
    #if the patient is not found, return an error
    if not patient:
        return {"error": f"patient {patient_id} not found"}

    #get the trial from the trial_search tool
    trial = get_trial(trial_id)
    #if the trial is not found, return an error
    if not trial:
        return {"error": f"trial {trial_id} not found"}

    #parse the trial's criteria into structured inclusion/exclusion lists using the parse_criteria tool
    parsed = parse_criteria(trial_id)
    if parsed.get("error"):
        return {"trial_id": trial_id, "error": f"could not parse criteria: {parsed.get('error')}"}
    #create a user message to the Nebius API to evaluate the patient against the trial's criteria, including the trial's ID, the inclusion and exclusion criteria, and the patient's data as a dictionary
    user_message = json.dumps({
        "trial_id": trial_id,
        "inclusion": parsed.get("inclusion", []),
        "exclusion": parsed.get("exclusion", []),
        "patient": _trim_patient(patient),
    }, indent=2)

    #call the Nebius API to evaluate the patient against the trial's criteria
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CHECK_ELIGIBILITY_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=4000,  # cap response length
        )
        #parse the response from the Nebius API to a JSON object
        result = _safe_json_parse(response.choices[0].message.content, trial_id)
        _eligibility_cache[cache_key] = result
        return result
    except Exception as e:
        return {"trial_id": trial_id, "error": f"LLM call failed: {e}"}
   


#convert the response from the Nebius API to a Python dictionary (JSON object)
def _safe_json_parse(text: str, trial_id: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        parsed = json.loads(text.strip())
        parsed.setdefault("trial_id", trial_id)
        return parsed
    except json.JSONDecodeError:
        return {"trial_id": trial_id, "error": "failed to parse", "raw": text}