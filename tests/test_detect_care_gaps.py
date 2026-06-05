"""Verify that care gaps are detected when guideline-recommended medications are absent."""
from agent.clinical_codes import SNOMED
from agent.ingest_patient import parse_bundle
from agent.models import InferredCondition

T2DM_CODE = str(SNOMED["T2DM"])
CKD_CODE = str(SNOMED["CKD_STAGE3"])


def _patient(conditions=None, medications=None):
    """Build and parse a minimal bundle with the given conditions and medications."""
    bundle = {
        "patient_id": "test-gap",
        "demographics": {"age": 65, "gender": "M"},
        "conditions": conditions or [],
        "medications": medications or [],
        "observations": [],
    }
    return parse_bundle(bundle)


def test_detects_missing_statin_for_t2dm():
    from agent.detect_care_gaps import detect_care_gaps
    patient = _patient(conditions=[
        {"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True}
    ])
    gaps = detect_care_gaps(patient, [])
    assert any(g.missing_drug == "statin" for g in gaps)


def test_no_gap_when_atorvastatin_present_for_t2dm():
    from agent.detect_care_gaps import detect_care_gaps
    patient = _patient(
        conditions=[{"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True}],
        medications=[{"description": "Atorvastatin 20 MG Oral Tablet", "active": True}],
    )
    gaps = detect_care_gaps(patient, [])
    assert not any(g.missing_drug == "statin" for g in gaps)


def test_detects_missing_ras_inhibitor_for_documented_ckd():
    from agent.detect_care_gaps import detect_care_gaps
    patient = _patient(conditions=[
        {"code": CKD_CODE, "description": "CKD Stage 3", "start": "2022-01-01", "active": True}
    ])
    gaps = detect_care_gaps(patient, [])
    assert any(g.missing_drug == "ACEi/ARB" for g in gaps)


def test_detects_missing_ras_inhibitor_for_inferred_ckd():
    from agent.detect_care_gaps import detect_care_gaps
    patient = _patient()
    inferred_ckd = InferredCondition(
        name="CKD_STAGE3",
        description="CKD Stage 3",
        snomed=SNOMED["CKD_STAGE3"],
        guideline="KDIGO 2024",
        evidence="eGFR 55 on 2024-01-01 and eGFR 52 on 2024-05-01, readings 121 days apart",
    )
    gaps = detect_care_gaps(patient, [inferred_ckd])
    assert any(g.missing_drug == "ACEi/ARB" for g in gaps)


def test_no_gap_when_lisinopril_present_for_ckd():
    from agent.detect_care_gaps import detect_care_gaps
    patient = _patient(
        conditions=[{"code": CKD_CODE, "description": "CKD Stage 3", "start": "2022-01-01", "active": True}],
        medications=[{"description": "Lisinopril 10 MG Oral Tablet", "active": True}],
    )
    gaps = detect_care_gaps(patient, [])
    assert not any(g.missing_drug == "ACEi/ARB" for g in gaps)


def test_no_gap_when_no_matching_condition():
    from agent.detect_care_gaps import detect_care_gaps
    patient = _patient()
    gaps = detect_care_gaps(patient, [])
    assert gaps == []


def test_care_gap_carries_guideline_citation():
    from agent.detect_care_gaps import detect_care_gaps
    patient = _patient(conditions=[
        {"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True}
    ])
    gaps = detect_care_gaps(patient, [])
    statin_gap = next(g for g in gaps if g.missing_drug == "statin")
    assert len(statin_gap.guideline) > 0
