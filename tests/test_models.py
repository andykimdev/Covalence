"""Verify that all pipeline dataclasses can be instantiated with their expected fields."""
from datetime import date


def test_lab_reading_fields():
    from agent.models import LabReading
    r = LabReading(value=57.3, units="mL/min/1.73m2", date=date(2024, 1, 1), description="eGFR")
    assert r.value == 57.3
    assert r.units == "mL/min/1.73m2"
    assert r.date == date(2024, 1, 1)
    assert r.description == "eGFR"


def test_condition_fields():
    from agent.models import Condition
    c = Condition(code="44054006", description="Type 2 diabetes mellitus", start="2020-01-01", active=True)
    assert c.code == "44054006"
    assert c.active is True


def test_parsed_patient_defaults():
    from agent.models import ParsedPatient
    p = ParsedPatient(patient_id="p1", age=65, gender="F", conditions=[], active_medications=[])
    assert p.labs == {}
    assert p.raw_bundle == {}


def test_inferred_condition_fields():
    from agent.models import InferredCondition
    ic = InferredCondition(
        name="CKD_STAGE3",
        description="Chronic kidney disease stage 3",
        snomed=433144002,
        guideline="KDIGO 2024",
        evidence="eGFR 55 on 2024-01-01",
    )
    assert ic.snomed == 433144002


def test_care_gap_fields():
    from agent.models import CareGap
    g = CareGap(condition="T2DM", missing_drug="statin", guideline="ADA 2025", reason="statin recommended for all T2DM patients over 40")
    assert g.missing_drug == "statin"


def test_expanded_indication_fields():
    from agent.models import ExpandedIndication
    ei = ExpandedIndication(name="CKD", confidence="medium", expansion_score=0.24, recommendation="Repeat eGFR in 90 days.")
    assert ei.confidence == "medium"


def test_expanded_patient_fields():
    from agent.models import ExpandedPatient, ParsedPatient
    parsed = ParsedPatient(patient_id="p1", age=65, gender="F", conditions=[], active_medications=[])
    ep = ExpandedPatient(
        parsed=parsed,
        inferred_conditions=[],
        care_gaps=[],
        expanded_indications=[],
        all_indications=["Type 2 diabetes"],
    )
    assert ep.all_indications == ["Type 2 diabetes"]
