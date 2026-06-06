"""Verify deterministic matching, provenance tracking, and provenance-weighted scoring."""
from agent.clinical_codes import LOINC, SNOMED


def _structured_trial(**overrides):
    base = {
        "nct_id": "NCT00000001", "title": "Test", "conditions": ["Type 2 Diabetes"],
        "min_age": 18, "max_age": 75, "sex": "all",
        "required_conditions": ["T2DM"], "excluded_conditions": [],
        "lab_thresholds": {"eGFR": {"min_value": 30, "max_value": 60}},
        "required_medications": ["statin"], "excluded_medications": [],
        "residual_criteria": [],
    }
    base.update(overrides)
    return base


def _enriched_patient(**overrides):
    base = {
        "patient_id": "p1",
        "demographics": {"age": 65, "gender": "M"},
        "conditions": [{"code": str(SNOMED["T2DM"]), "description": "Type 2 Diabetes", "active": True}],
        "active_medications": [],
        "observations": [{"code": LOINC["EGFR"], "value": 55.0, "date": "2024-01-01"}],
        "inferred_conditions": [],
        "care_gaps": [{"condition": "T2DM", "missing_drug": "statin", "guideline": "ADA 2025", "reason": "..."}],
        "expanded_indications": [],
        "all_indications": ["Type 2 Diabetes"],
    }
    base.update(overrides)
    return base


def test_candidate_filter_matches_on_required_conditions():
    from agent.match_engine import get_candidates
    patient = _enriched_patient()
    trials = [
        _structured_trial(),
        _structured_trial(nct_id="NCT2", conditions=["Lung Cancer"], required_conditions=["lung_cancer"]),
    ]
    candidates = get_candidates(patient, trials)
    assert any(c["nct_id"] == "NCT00000001" for c in candidates)
    assert not any(c["nct_id"] == "NCT2" for c in candidates)


def test_age_pass():
    from agent.match_engine import match_patient_to_trial
    verdicts = match_patient_to_trial(_enriched_patient(), _structured_trial())
    age_verdicts = [v for v in verdicts if "age" in v["criterion"]]
    assert all(v["verdict"] == "PASS" for v in age_verdicts)


def test_age_fail_outside_range():
    from agent.match_engine import match_patient_to_trial
    patient = _enriched_patient()
    patient["demographics"]["age"] = 80
    verdicts = match_patient_to_trial(patient, _structured_trial())
    age_max = next(v for v in verdicts if "age <=" in v["criterion"])
    assert age_max["verdict"] == "FAIL"


def test_lab_threshold_pass():
    from agent.match_engine import match_patient_to_trial
    verdicts = match_patient_to_trial(_enriched_patient(), _structured_trial())
    egfr = next(v for v in verdicts if "eGFR" in v["criterion"])
    assert egfr["verdict"] == "PASS"


def test_lab_missing_is_unknown():
    from agent.match_engine import match_patient_to_trial
    patient = _enriched_patient(observations=[])
    verdicts = match_patient_to_trial(patient, _structured_trial())
    egfr = next(v for v in verdicts if "eGFR" in v["criterion"])
    assert egfr["verdict"] == "UNKNOWN"


def test_missing_statin_flagged_resolvable():
    from agent.match_engine import match_patient_to_trial, annotate_with_care_gaps
    patient = _enriched_patient()
    verdicts = match_patient_to_trial(patient, _structured_trial())
    verdicts = annotate_with_care_gaps(verdicts, patient)
    statin = next(v for v in verdicts if "statin" in v["criterion"])
    assert statin["verdict"] == "FAIL"
    assert statin["resolvable"] is True
    assert statin["care_gap"] is not None


def test_provenance_documented_full_weight():
    from agent.match_engine import compute_provenance
    patient = _enriched_patient()
    tier, weight = compute_provenance(patient, "T2DM")
    assert tier == "documented"
    assert weight == 1.0


def test_provenance_inferred_reduced_weight():
    from agent.match_engine import compute_provenance
    patient = _enriched_patient(
        inferred_conditions=[{
            "name": "CKD_STAGE3", "description": "Chronic Kidney Disease",
            "snomed": SNOMED["CKD_STAGE3"], "guideline": "KDIGO 2024", "evidence": "..."
        }]
    )
    tier, weight = compute_provenance(patient, "CKD")
    assert tier == "inferred"
    assert weight == 0.7


def test_provenance_expanded_uses_expansion_score():
    from agent.match_engine import compute_provenance
    patient = _enriched_patient(
        expanded_indications=[{
            "name": "CKD", "confidence": "high",
            "expansion_score": 0.356, "recommendation": "..."
        }]
    )
    tier, weight = compute_provenance(patient, "CKD")
    assert tier == "expanded"
    assert weight == 0.356


def test_expanded_match_scores_lower_than_documented():
    from agent.match_engine import process_patient
    documented_patient = _enriched_patient()
    expanded_patient = _enriched_patient(
        conditions=[],
        all_indications=["CKD"],
        expanded_indications=[{
            "name": "CKD", "confidence": "medium",
            "expansion_score": 0.30, "recommendation": "..."
        }],
    )
    doc_trial = _structured_trial(required_medications=[])
    ckd_trial = _structured_trial(
        nct_id="NCT2", required_conditions=["CKD"],
        required_medications=[], lab_thresholds={},
    )
    doc_results = process_patient(documented_patient, [doc_trial], skip_residual=True)
    exp_results = process_patient(expanded_patient, [ckd_trial], skip_residual=True)
    assert len(doc_results) > 0
    assert len(exp_results) > 0
    assert doc_results[0]["score"] > exp_results[0]["score"]
