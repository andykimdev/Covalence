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
client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL"), api_key=os.getenv("NEBIUS_API_KEY"))
#set model, default to Llama 3.3 70B Instruct unless specified in environment variables
MODEL = os.getenv("MODEL_CHECK", "meta-llama/Llama-3.3-70B-Instruct")

#trim the patient bundle to what check_eligibility actually needs to save on tokens and reduce the size of the input data
def _trim_patient(patient: dict) -> dict:
    """Reduce patient bundle to what check_eligibility actually needs."""
    return {
        "patient_id": patient.get("patient_id"),
        "demographics": patient.get("demographics", {}),
        "summary": patient.get("summary", {}),
        "active_conditions": [c for c in patient.get("conditions", []) if c.get("active")],
        "current_medications": [m for m in patient.get("medications", []) if m.get("active")],
        "recent_observations": patient.get("observations", [])[-15:],  # last 15
    }

#define check_eligibility function to evaluate a patient against a single trial's structured criteria
def check_eligibility(patient_id: str, trial_id: str) -> dict:
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
        # DEBUG — remove after we figure out what's going on
        print(f"[DEBUG check_eligibility {trial_id}]")
        print(f"  finish_reason: {response.choices[0].finish_reason}")
        print(f"  content: {response.choices[0].message.content!r}")
        print(f"  tool_calls: {response.choices[0].message.tool_calls}")
         #parse the response from the Nebius API to a JSON object
        result = _safe_json_parse(response.choices[0].message.content, trial_id)
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