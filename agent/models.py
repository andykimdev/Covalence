"""Shared dataclasses for the Covalent pipeline."""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class LabReading:
    """A single lab measurement taken on a specific date."""
    value: float
    units: str
    date: date
    description: str


@dataclass
class Condition:
    """A single condition on the patient's problem list."""
    code: str
    description: str
    start: str
    active: bool


@dataclass
class ParsedPatient:
    """Clean structured representation of one patient for the pipeline."""
    patient_id: str
    age: int | None
    gender: str
    conditions: list[Condition]
    # Active medication descriptions are stored as lowercase strings for substring matching.
    active_medications: list[str]
    # Labs are grouped by LOINC code and sorted oldest-to-newest for trend detection.
    labs: dict[str, list[LabReading]] = field(default_factory=dict)
    raw_bundle: dict = field(default_factory=dict)


@dataclass
class InferredCondition:
    """A condition implied by lab values but not yet on the patient's problem list."""
    name: str
    description: str
    # SNOMED code is stored so downstream steps can add this condition to SNOMED code set lookups.
    snomed: int
    guideline: str
    evidence: str


@dataclass
class CareGap:
    """A missing guideline-recommended medication for a condition the patient has."""
    condition: str
    missing_drug: str
    guideline: str
    reason: str


@dataclass
class ExpandedIndication:
    """An indication the patient may be developing based on comorbidity graph evidence."""
    name: str
    confidence: str
    expansion_score: float
    recommendation: str


@dataclass
class ExpandedPatient:
    """The handoff object passed from Steps 1-4 to the LLM agent."""
    parsed: ParsedPatient
    inferred_conditions: list[InferredCondition]
    care_gaps: list[CareGap]
    expanded_indications: list[ExpandedIndication]
    # all_indications is the pre-merged list the agent uses as its trial search query basis.
    all_indications: list[str]
