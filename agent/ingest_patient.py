"""Parse raw patient bundle dicts into ParsedPatient objects for the pipeline."""
import json
from datetime import date, datetime
from pathlib import Path

from agent.clinical_codes import LOINC
from agent.models import Condition, LabReading, ParsedPatient

# Build the set of LOINC codes we care about from clinical_codes so we only
# store lab readings that one of the pipeline steps will actually use.
RELEVANT_LAB_CODES = set(LOINC.values())


def _parse_date(date_str: str) -> date | None:
    """Convert a YYYY-MM-DD string to a date object, returning None if malformed."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_bundle(bundle: dict) -> ParsedPatient:
    """Parse one raw patient bundle dict into a ParsedPatient."""
    demographics = bundle.get("demographics", {})
    age = demographics.get("age")
    gender = demographics.get("gender", "")

    conditions = []
    for raw_cond in bundle.get("conditions", []):
        conditions.append(Condition(
            code=raw_cond.get("code", ""),
            description=raw_cond.get("description", ""),
            start=raw_cond.get("start", ""),
            active=bool(raw_cond.get("active", False)),
        ))

    # Keep only active medications and lowercase their descriptions so that
    # care gap detection can match drug keywords without case sensitivity issues.
    active_medications = []
    for raw_med in bundle.get("medications", []):
        if raw_med.get("active"):
            active_medications.append(raw_med.get("description", "").lower())

    labs: dict[str, list[LabReading]] = {}
    for raw_obs in bundle.get("observations", []):
        code = raw_obs.get("code", "")
        if code not in RELEVANT_LAB_CODES:
            continue
        raw_value = raw_obs.get("value")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        reading_date = _parse_date(raw_obs.get("date", ""))
        if reading_date is None:
            continue
        reading = LabReading(
            value=value,
            units=raw_obs.get("units") or "",
            date=reading_date,
            description=raw_obs.get("description", ""),
        )
        if code not in labs:
            labs[code] = []
        labs[code].append(reading)

    # Sort each lab's readings from oldest to newest so the rules engine can
    # walk them in chronological order when checking time-based rules like
    # two eGFR readings at least 90 days apart.
    for code in labs:
        labs[code].sort(key=lambda r: r.date)

    return ParsedPatient(
        patient_id=bundle.get("patient_id", "unknown"),
        age=age,
        gender=gender,
        conditions=conditions,
        active_medications=active_medications,
        labs=labs,
        raw_bundle=bundle,
    )


def load_bundles(path: str | Path) -> list[ParsedPatient]:
    """Load all patient bundles from a JSON file and return a list of ParsedPatient objects."""
    path = Path(path)
    with open(path, "r") as f:
        raw_bundles = json.load(f)
    patients = [parse_bundle(bundle) for bundle in raw_bundles]
    print(f"Loaded {len(patients)} patients from {path.name}")
    return patients
