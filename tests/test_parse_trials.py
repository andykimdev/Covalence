"""Verify the offline parser converts raw trials into StructuredTrial objects."""
import json
from unittest.mock import patch


SAMPLE_LLM_OUTPUT = json.dumps({
    "min_age": 18, "max_age": 75, "sex": "all",
    "required_conditions": ["T2DM"],
    "excluded_conditions": [],
    "lab_thresholds": {"eGFR": {"min": 30, "max": 60}, "a1c": {"min": 7.0, "max": 10.5}},
    "required_medications": ["antidiabetic"],
    "excluded_medications": [],
    "residual_criteria": ["Stable dose of metformin for at least 8 weeks"],
})


def _raw_trial():
    return {
        "nct_id": "NCT00000001",
        "title": "Test Trial",
        "conditions": ["Type 2 Diabetes"],
        "eligibility_text": "Inclusion: adults 18-75 with T2DM, eGFR 30-60 ...",
    }


def test_parse_one_trial_returns_structured():
    from agent.parse_trials_offline import parse_one_trial
    with patch("agent.parse_trials_offline._call_llm", return_value=SAMPLE_LLM_OUTPUT):
        result = parse_one_trial(_raw_trial())
    assert result is not None
    assert result.nct_id == "NCT00000001"
    assert result.min_age == 18
    assert result.max_age == 75
    assert result.lab_thresholds["eGFR"].min_value == 30
    assert result.lab_thresholds["eGFR"].max_value == 60
    assert result.lab_thresholds["a1c"].min_value == 7.0
    assert "antidiabetic" in result.required_medications
    assert len(result.residual_criteria) == 1


def test_parse_handles_code_fences():
    from agent.parse_trials_offline import parse_one_trial
    fenced = "```json\n" + SAMPLE_LLM_OUTPUT + "\n```"
    with patch("agent.parse_trials_offline._call_llm", return_value=fenced):
        result = parse_one_trial(_raw_trial())
    assert result is not None
    assert result.min_age == 18


def test_parse_skips_unparseable_trial():
    from agent.parse_trials_offline import parse_one_trial
    with patch("agent.parse_trials_offline._call_llm", return_value="not json"):
        result = parse_one_trial(_raw_trial())
    assert result is None


def test_parse_handles_null_ages():
    from agent.parse_trials_offline import parse_one_trial
    output = json.dumps({
        "min_age": None, "max_age": None, "sex": "all",
        "required_conditions": [], "excluded_conditions": [],
        "lab_thresholds": {}, "required_medications": [],
        "excluded_medications": [], "residual_criteria": [],
    })
    with patch("agent.parse_trials_offline._call_llm", return_value=output):
        result = parse_one_trial(_raw_trial())
    assert result is not None
    assert result.min_age is None
    assert result.max_age is None
