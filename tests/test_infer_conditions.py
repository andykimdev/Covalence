"""Verify that the rules engine correctly infers conditions from lab readings."""
from agent.clinical_codes import LOINC, SNOMED
from agent.ingest_patient import parse_bundle

EGFR = LOINC["EGFR"]
A1C = LOINC["A1C"]


def _patient(observations, conditions=None):
    """Build and parse a minimal bundle with the given observations and conditions."""
    bundle = {
        "patient_id": "test-inf",
        "demographics": {"age": 65, "gender": "M"},
        "conditions": conditions or [],
        "medications": [],
        "observations": observations,
    }
    return parse_bundle(bundle)


def test_infers_ckd_from_two_egfr_readings_90_days_apart():
    from agent.infer_conditions import infer_conditions
    patient = _patient([
        {"code": EGFR, "value": 55.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"},
        {"code": EGFR, "value": 52.0, "units": "mL/min", "date": "2024-05-01", "description": "eGFR"},
    ])
    result = infer_conditions(patient)
    assert any(ic.name == "CKD_STAGE3" for ic in result)


def test_does_not_infer_ckd_from_single_reading():
    from agent.infer_conditions import infer_conditions
    patient = _patient([
        {"code": EGFR, "value": 55.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"},
    ])
    result = infer_conditions(patient)
    assert not any(ic.name == "CKD_STAGE3" for ic in result)


def test_does_not_infer_ckd_when_readings_less_than_90_days_apart():
    from agent.infer_conditions import infer_conditions
    patient = _patient([
        {"code": EGFR, "value": 55.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"},
        {"code": EGFR, "value": 53.0, "units": "mL/min", "date": "2024-01-30", "description": "eGFR"},
    ])
    result = infer_conditions(patient)
    assert not any(ic.name == "CKD_STAGE3" for ic in result)


def test_does_not_infer_ckd_when_already_documented():
    from agent.infer_conditions import infer_conditions
    patient = _patient(
        observations=[
            {"code": EGFR, "value": 55.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"},
            {"code": EGFR, "value": 52.0, "units": "mL/min", "date": "2024-05-01", "description": "eGFR"},
        ],
        conditions=[
            {"code": str(SNOMED["CKD_STAGE3"]), "description": "CKD Stage 3", "start": "2023-01-01", "active": True}
        ],
    )
    result = infer_conditions(patient)
    assert not any(ic.name == "CKD_STAGE3" for ic in result)


def test_infers_t2dm_from_two_a1c_readings():
    from agent.infer_conditions import infer_conditions
    patient = _patient([
        {"code": A1C, "value": 7.1, "units": "%", "date": "2024-01-01", "description": "A1c"},
        {"code": A1C, "value": 6.8, "units": "%", "date": "2024-04-01", "description": "A1c"},
    ])
    result = infer_conditions(patient)
    assert any(ic.name == "T2DM" for ic in result)


def test_does_not_infer_t2dm_from_single_a1c_reading():
    from agent.infer_conditions import infer_conditions
    patient = _patient([
        {"code": A1C, "value": 7.1, "units": "%", "date": "2024-01-01", "description": "A1c"},
    ])
    result = infer_conditions(patient)
    assert not any(ic.name == "T2DM" for ic in result)


def test_inferred_condition_carries_snomed_code():
    from agent.infer_conditions import infer_conditions
    patient = _patient([
        {"code": EGFR, "value": 55.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"},
        {"code": EGFR, "value": 52.0, "units": "mL/min", "date": "2024-05-01", "description": "eGFR"},
    ])
    result = infer_conditions(patient)
    ckd = next(ic for ic in result if ic.name == "CKD_STAGE3")
    assert ckd.snomed == SNOMED["CKD_STAGE3"]


def test_inferred_condition_evidence_string_is_nonempty():
    from agent.infer_conditions import infer_conditions
    patient = _patient([
        {"code": EGFR, "value": 55.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"},
        {"code": EGFR, "value": 52.0, "units": "mL/min", "date": "2024-05-01", "description": "eGFR"},
    ])
    result = infer_conditions(patient)
    ckd = next(ic for ic in result if ic.name == "CKD_STAGE3")
    assert len(ckd.evidence) > 0
