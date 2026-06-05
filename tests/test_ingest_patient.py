"""Verify that raw bundle dicts are parsed into correctly structured ParsedPatient objects."""
from agent.clinical_codes import LOINC


def _make_bundle(conditions=None, medications=None, observations=None):
    """Build a minimal patient bundle dict for testing."""
    return {
        "patient_id": "test-001",
        "demographics": {"age": 65, "gender": "F"},
        "conditions": conditions or [],
        "medications": medications or [],
        "observations": observations or [],
    }


def test_parse_bundle_demographics():
    from agent.ingest_patient import parse_bundle
    patient = parse_bundle(_make_bundle())
    assert patient.patient_id == "test-001"
    assert patient.age == 65
    assert patient.gender == "F"


def test_parse_bundle_active_medications_are_lowercased():
    from agent.ingest_patient import parse_bundle
    bundle = _make_bundle(medications=[
        {"description": "Metformin 500 MG Oral Tablet", "active": True},
        {"description": "Aspirin 81 MG", "active": False},
    ])
    patient = parse_bundle(bundle)
    assert patient.active_medications == ["metformin 500 mg oral tablet"]


def test_parse_bundle_inactive_medications_are_excluded():
    from agent.ingest_patient import parse_bundle
    bundle = _make_bundle(medications=[
        {"description": "Aspirin 81 MG", "active": False},
    ])
    patient = parse_bundle(bundle)
    assert patient.active_medications == []


def test_parse_bundle_labs_sorted_oldest_to_newest():
    from agent.ingest_patient import parse_bundle
    egfr_code = LOINC["EGFR"]
    bundle = _make_bundle(observations=[
        {"code": egfr_code, "value": 58.0, "units": "mL/min", "date": "2024-07-15", "description": "eGFR"},
        {"code": egfr_code, "value": 61.0, "units": "mL/min", "date": "2024-03-10", "description": "eGFR"},
    ])
    patient = parse_bundle(bundle)
    readings = patient.labs[egfr_code]
    assert len(readings) == 2
    assert readings[0].date < readings[1].date
    assert readings[0].value == 61.0


def test_parse_bundle_skips_non_relevant_observation_codes():
    from agent.ingest_patient import parse_bundle
    bundle = _make_bundle(observations=[
        {"code": "DALY", "value": 1.3, "units": "a", "date": "2024-01-01", "description": "DALY"},
    ])
    patient = parse_bundle(bundle)
    assert len(patient.labs) == 0


def test_parse_bundle_skips_observations_with_no_valid_date():
    from agent.ingest_patient import parse_bundle
    egfr_code = LOINC["EGFR"]
    bundle = _make_bundle(observations=[
        {"code": egfr_code, "value": 55.0, "units": "mL/min", "date": "", "description": "eGFR"},
    ])
    patient = parse_bundle(bundle)
    assert egfr_code not in patient.labs


def test_parse_bundle_skips_observations_with_non_numeric_value():
    from agent.ingest_patient import parse_bundle
    egfr_code = LOINC["EGFR"]
    bundle = _make_bundle(observations=[
        {"code": egfr_code, "value": "Negative", "units": "", "date": "2024-01-01", "description": "eGFR"},
    ])
    patient = parse_bundle(bundle)
    assert egfr_code not in patient.labs


def test_parse_bundle_stores_raw_bundle():
    from agent.ingest_patient import parse_bundle
    bundle = _make_bundle()
    patient = parse_bundle(bundle)
    assert patient.raw_bundle == bundle
