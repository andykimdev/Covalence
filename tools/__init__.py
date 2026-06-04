"""Tool registry: schemas the agent sees + dispatcher that runs the real functions."""
from tools.trial_search import trial_search
from tools.parse_criteria import parse_criteria
from tools.check_eligibility import check_eligibility
from tools.validate_verdicts import validate_verdicts
from tools.rank_rationale import rank_with_rationale


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "trial_search",
            "description": (
                "Retrieve up to N candidate clinical trials relevant to a patient query. "
                "Use this first to get candidate trials. "
                "Call multiple times with different queries if the patient might fit trials "
                "across more than one indication area (e.g., primary diagnosis AND a comorbidity)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query, e.g. 'HER2+ metastatic breast cancer trastuzumab progression'"},
                    "top_n": {"type": "integer", "default": 14},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_criteria",
            "description": (
                "Extract structured inclusion and exclusion criteria for a single trial. "
                "Call this once per candidate trial after retrieval, before checking eligibility."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trial_id": {"type": "string", "description": "NCT ID, e.g. 'NCT01234567'"},
                },
                "required": ["trial_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": (
                "Evaluate a patient against a single trial's structured criteria. "
                "Returns per-criterion verdicts (PASS, FAIL, or UNKNOWN), an overall assessment, "
                "and a list of missing patient data needed to convert UNKNOWNs. "
                "Call this once per candidate trial after parse_criteria."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {"type": "object", "description": "patient JSON"},
                    "trial_id": {"type": "string", "description": "NCT ID"},
                },
                "required": ["patient", "trial_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_verdicts",
            "description": (
                "Self-check the verdict objects against the patient record. "
                "Flags any PASS verdicts that should have been UNKNOWN (patient field is null). "
                "Call this once after check_eligibility for all trials, before ranking. "
                "If issues are flagged, the agent should re-run check_eligibility on the affected trials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {"type": "object"},
                    "verdict_objects": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["patient", "verdict_objects"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_with_rationale",
            "description": (
                "Score and rank the verdict objects, return the top 3 with per-trial rationale. "
                "Call this once at the end after all eligibility checks and validation are complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "verdict_objects": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["patient_id", "verdict_objects"],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> dict:
    """Dispatch tool call by name. Returns dict ready to serialize back to the agent."""
    # the ** before the args is used to unpack the dictionary passed as an argument into keyword arguments
    # Lambda delays the call so only the selected tool runs when dispatch[name]() is invoked, so function call is delayed until we pick a tool to run    
    # Each dict value is a small function that will invoke the matching tool with args
    dispatch = {
        "trial_search": lambda: trial_search(**args),
        "parse_criteria": lambda: parse_criteria(**args),
        "check_eligibility": lambda: check_eligibility(**args),
        "validate_verdicts": lambda: validate_verdicts(**args),
        "rank_with_rationale": lambda: rank_with_rationale(**args),
    }
    if name not in dispatch:
        return {"error": f"unknown tool: {name}"}

    # call the lambda function that is the value of the key in the dispatch dictionary
    return dispatch[name]()