"""Verify that the pipeline correctly orchestrates Steps 1-4 and produces an enriched dict."""
import json
from agent.clinical_codes import LOINC, SNOMED

EGFR = LOINC["EGFR"]
T2DM_CODE = str(SNOMED["T2DM"])


def _t2dm_bundle():
    """A T2DM patient with borderline eGFR and no medications, so gaps and expansions are expected."""
    return {
        "patient_id": "test-pipe",
        "demographics": {"age": 65, "gender": "M"},
        "conditions": [
            {"code": T2DM_CODE, "description": "Type 2 diabetes", "start": "2020-01-01", "active": True}
        ],
        "medications": [],
        "observations": [
            {"code": EGFR, "value": 65.0, "units": "mL/min", "date": "2024-01-01", "description": "eGFR"},
        ],
    }


def test_enrich_injects_four_keys():
    from agent.pipeline import enrich_patient_dict
    enriched = enrich_patient_dict(_t2dm_bundle())
    assert "inferred_conditions" in enriched
    assert "care_gaps" in enriched
    assert "expanded_indications" in enriched
    assert "all_indications" in enriched


def test_enrich_preserves_original_bundle_fields():
    from agent.pipeline import enrich_patient_dict
    bundle = _t2dm_bundle()
    enriched = enrich_patient_dict(bundle)
    assert enriched["patient_id"] == "test-pipe"
    assert enriched["demographics"] == bundle["demographics"]
    assert enriched["conditions"] == bundle["conditions"]


def test_all_indications_is_nonempty_for_multi_indication_patient():
    from agent.pipeline import enrich_patient_dict
    enriched = enrich_patient_dict(_t2dm_bundle())
    assert len(enriched["all_indications"]) > 0


def test_care_gaps_are_detected_for_t2dm_without_statin():
    from agent.pipeline import enrich_patient_dict
    enriched = enrich_patient_dict(_t2dm_bundle())
    drug_classes = [g["missing_drug"] for g in enriched["care_gaps"]]
    assert "statin" in drug_classes


def test_ckd_expansion_present_for_t2dm_with_borderline_egfr():
    from agent.pipeline import enrich_patient_dict
    enriched = enrich_patient_dict(_t2dm_bundle())
    names = [ei["name"] for ei in enriched["expanded_indications"]]
    assert "CKD" in names


def test_enrich_output_is_json_serializable():
    from agent.pipeline import enrich_patient_dict
    enriched = enrich_patient_dict(_t2dm_bundle())
    json.dumps(enriched, default=str)
