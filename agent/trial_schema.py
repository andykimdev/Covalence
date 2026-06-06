"""Structured representation of a clinical trial's eligibility criteria."""
from dataclasses import dataclass, field


@dataclass
class LabThreshold:
    """A min/max range for a single lab. None means no bound on that side."""
    min_value: float | None
    max_value: float | None


@dataclass
class StructuredTrial:
    """A trial's eligibility criteria parsed into a standardized, matchable schema."""
    nct_id: str
    title: str
    conditions: list[str]                      # trial's listed conditions (raw text)
    min_age: int | None
    max_age: int | None
    sex: str                                   # "all", "male", "female"
    required_conditions: list[str]             # controlled-vocab short codes
    excluded_conditions: list[str]             # controlled-vocab short codes
    lab_thresholds: dict[str, LabThreshold]    # keyed by controlled-vocab lab name
    required_medications: list[str]            # controlled-vocab med classes
    excluded_medications: list[str]            # controlled-vocab med classes
    residual_criteria: list[str]               # free-text, LLM-checked only if finalist
