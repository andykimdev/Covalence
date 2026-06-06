"""Self-correction tool. Flags PASS verdicts where the patient bundle lacks the relevant data."""
from data.load_fixtures import get_patient
import re





def _has_condition(patient: dict, keyword: str) -> bool:
    """Check if patient has a condition matching keyword (in code description, active conditions)."""
    keyword = keyword.lower()
    for cond in patient.get("conditions", []):
        if cond.get("active") and keyword in cond.get("description", "").lower():
            return True
    for cond in patient.get("summary", {}).get("active_conditions", []):
        if keyword in cond.lower():
            return True
    return False


def _has_lab(patient: dict, keyword: str) -> tuple[bool, any]:
    """Check observations for a lab matching keyword."""
    keyword = keyword.lower()
    for obs in patient.get("observations", []):
        if keyword in obs.get("description", "").lower():
            return True, obs.get("value")
    return False, None


# Keywords in criteria text → check function
CRITERION_CHECKS = {
    "ejection fraction": "ejection fraction",
    "lvef": "ejection fraction",
    "hba1c": "hemoglobin a1c",
    "creatinine": "creatinine",
    "egfr": "glomerular filtration",
    "bilirubin": "bilirubin",
    "ast": "aspartate aminotransferase",
    "alt": "alanine aminotransferase",
    "hemoglobin": "hemoglobin",
    "platelet": "platelet",
}


def validate_verdicts(patient_id: str, verdict_objects) -> dict:
    import json
    if isinstance(verdict_objects, str):
        try:
            verdict_objects = json.loads(verdict_objects)
        except json.JSONDecodeError:
            return {"validated": False, "issues": [{"error": "verdict_objects could not be parsed as JSON"}]}
    patient = get_patient(patient_id)
    if not patient:
        return {"validated": False, "issues": [{"error": f"patient {patient_id} not found"}]}

    issues = []
    for v in verdict_objects:
        trial_id = v.get("trial_id", "unknown")
        for cv in v.get("criteria_verdicts", []):
            if cv.get("verdict") != "PASS":
                continue
            criterion_lower = cv.get("criterion", "").lower()
            for keyword, lab_term in CRITERION_CHECKS.items():
                if re.search(rf'\b{re.escape(keyword)}\b', criterion_lower): 
                    found, value = _has_lab(patient, lab_term)
                    if not found:
                        issues.append({
                            "trial_id": trial_id,
                            "criterion": cv["criterion"],
                            "issue": f"PASS verdict but patient bundle has no {lab_term} observation",
                        })
                    break
    return {"validated": len(issues) == 0, "issues": issues}