import json, os, threading
from openai import OpenAI
from dotenv import load_dotenv
from agent.prompts import PARSE_CRITERIA_PROMPT
from data.load_fixtures import get_trial

load_dotenv()
client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL"), api_key=os.getenv("NEBIUS_API_KEY"))
model = os.getenv("MODEL_PARSE", "meta-llama/Llama-3.3-70B-Instruct")

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "fixtures", "parse_criteria_cache.json")
_cache_lock = threading.Lock()

def _load_cache() -> dict:
    """Load persisted parse_criteria results from disk if the cache file exists."""
    try:
        with open(_CACHE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache() -> None:
    """Persist the in-memory cache to disk so it survives server restarts."""
    with _cache_lock:
        snapshot = dict(_parse_cache)
    with open(_CACHE_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)

_parse_cache: dict[str, dict] = _load_cache()


#define parse_criteria function to parse a trial's raw criteria text into structured inclusion/exclusion lists
def parse_criteria(trial_id: str) -> dict:
    #check if the criteria for this trial is already in the cache
    if trial_id in _parse_cache:
        return _parse_cache[trial_id]

    #get the trial from the trial_search tool
    trial = get_trial(trial_id)
    if not trial:
        return {"error": f"trial {trial_id} not found"}

    raw_text = trial.get("eligibility_text", "")
    if not raw_text:
        return {"trial_id": trial_id, "inclusion": [], "exclusion": []}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PARSE_CRITERIA_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        temperature=0.0,
    )

    result = _safe_parse(response.choices[0].message.content, trial_id)
    with _cache_lock:
        _parse_cache[trial_id] = result
    save_cache()
    return result


def _safe_parse(text: str, trial_id: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        parsed = json.loads(text.strip())
        parsed["trial_id"] = trial_id
        return parsed
    except json.JSONDecodeError:
        return {"trial_id": trial_id, "error": "parse_fail", "raw": text[:500]}