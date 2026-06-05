"""Cross-reference patient conditions against guideline treatment tables to find care gaps."""
from agent.clinical_codes import GUIDELINE_TREATMENTS
from agent.models import CareGap, InferredCondition, ParsedPatient


def detect_care_gaps(patient: ParsedPatient, inferred: list[InferredCondition]) -> list[CareGap]:
    """Return gaps where a condition is present but its recommended medication is absent."""
    # Build the active SNOMED code set from both documented and inferred conditions
    # so that gaps are flagged even when the condition was not on the original chart.
    active_snomed = {int(c.code) for c in patient.conditions if c.active and c.code.isdigit()}
    for ic in inferred:
        active_snomed.add(ic.snomed)

    gaps = []
    for condition_name, entry in GUIDELINE_TREATMENTS.items():
        if not active_snomed.intersection(entry["condition_codes"]):
            continue

        for med in entry["required_meds"]:
            if not _patient_has_med(patient.active_medications, med["keywords"]):
                gaps.append(CareGap(
                    condition=condition_name,
                    missing_drug=med["class"],
                    guideline=entry["source"],
                    reason=med["rationale"],
                ))

    return gaps


def _patient_has_med(active_medications: list[str], keywords: list[str]) -> bool:
    """Return True if any active medication description contains any of the drug keywords."""
    for med in active_medications:
        for kw in keywords:
            if kw in med:
                return True
    return False
