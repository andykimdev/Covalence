"""Verify that the disease graph expansion correctly scores candidate indications."""
from agent.clinical_codes import LOINC, SNOMED
from agent.ingest_patient import parse_bundle

EGFR = LOINC["EGFR"]
T2DM_CODE = str(SNOMED["T2DM"])
CKD_CODE = str(SNOMED["CKD_STAGE3"])


def _patient(conditions=None, observations=None):
    """Build and parse a minimal bundle with the given conditions and observations."""
    bundle = {
        "patient_id": "test-exp",
        "demographics": {"age": 65, "gender": "M"},
        "conditions": conditions or [],
        "medications": [],
        "observations": observations or [],
    }
    return parse_bundle(bundle)


def test_expands_t2dm_to_ckd_with_borderline_egfr():
    from agent.expand_indications import expand_indications
    patient = _patient(
        conditions=[{"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True}],
        observations=[{"code": EGFR, "value": 65.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"}],
    )
    result = expand_indications(patient, [])
    assert any(ei.name == "CKD" for ei in result)


def test_does_not_expand_t2dm_to_ckd_with_normal_egfr():
    from agent.expand_indications import expand_indications
    patient = _patient(
        conditions=[{"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True}],
        observations=[{"code": EGFR, "value": 95.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"}],
    )
    result = expand_indications(patient, [])
    assert not any(ei.name == "CKD" for ei in result)


def test_expansion_score_matches_formula():
    from agent.expand_indications import expand_indications
    patient = _patient(
        conditions=[{"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True}],
        observations=[{"code": EGFR, "value": 65.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"}],
    )
    result = expand_indications(patient, [])
    ckd = next(ei for ei in result if ei.name == "CKD")
    # T2DM to CKD has normalized RR 0.593 (from RR 2.31), eGFR 65 falls in range (60, 75) with lab score 0.6.
    # Expected expansion_score is round(0.593 times 0.6, 3) = 0.356, which is above 0.35 so confidence is high.
    assert ckd.expansion_score == 0.356
    assert ckd.confidence == "high"


def test_does_not_expand_to_already_documented_condition():
    from agent.expand_indications import expand_indications
    patient = _patient(
        conditions=[
            {"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True},
            {"code": CKD_CODE, "description": "CKD Stage 3", "start": "2022-01-01", "active": True},
        ],
        observations=[{"code": EGFR, "value": 65.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"}],
    )
    result = expand_indications(patient, [])
    assert not any(ei.name == "CKD" for ei in result)


def test_high_confidence_for_score_above_0_35():
    from agent.expand_indications import expand_indications
    # CKD to CHF has the strongest edge with normalized weight 1.00 (from RR 3.21).
    # NT-proBNP of 400 falls in range (300, None) with lab evidence score 0.8.
    # Expected expansion_score is 1.00 times 0.8 = 0.800, which is above 0.35 so confidence is high.
    nt_probnp_code = LOINC["NT_PROBNP"]
    patient = _patient(
        conditions=[{"code": CKD_CODE, "description": "CKD Stage 3", "start": "2022-01-01", "active": True}],
        observations=[{"code": nt_probnp_code, "value": 400.0, "units": "pg/mL", "date": "2024-01-01", "description": "NT-proBNP"}],
    )
    result = expand_indications(patient, [])
    chf = next((ei for ei in result if ei.name == "CHF"), None)
    assert chf is not None
    assert chf.expansion_score == 0.8
    assert chf.confidence == "high"


def test_no_duplicate_expansions_when_multiple_paths_lead_to_same_node():
    from agent.expand_indications import expand_indications
    # Both T2DM (normalized 0.181) and OBESITY (normalized 0.330) have edges to HYPERTENSION.
    # With systolic BP 135 (lab score 1.0), both paths exceed the 0.15 threshold.
    # The deduplication should keep only the higher-scoring path, which is OBESITY at 0.330.
    systolic_code = LOINC["SYSTOLIC_BP"]
    obesity_code = str(SNOMED["OBESITY"])
    patient = _patient(
        conditions=[
            {"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True},
            {"code": obesity_code, "description": "Obesity", "start": "2021-01-01", "active": True},
        ],
        observations=[{"code": systolic_code, "value": 135.0, "units": "mmHg", "date": "2024-01-01", "description": "Systolic BP"}],
    )
    result = expand_indications(patient, [])
    hypertension_matches = [ei for ei in result if ei.name == "HYPERTENSION"]
    assert len(hypertension_matches) == 1
    assert hypertension_matches[0].expansion_score == 0.330
