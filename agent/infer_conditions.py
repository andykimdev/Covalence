"""Scan patient labs against clinical guideline thresholds to surface undiagnosed conditions."""
from agent.clinical_codes import INFERENCE_RULES
from agent.models import InferredCondition, ParsedPatient


def infer_conditions(patient: ParsedPatient) -> list[InferredCondition]:
    """Return conditions implied by lab values that are not yet on the patient's chart."""
    # Build the set of SNOMED codes already documented so we do not re-flag known conditions.
    documented_snomed = {
        int(c.code) for c in patient.conditions if c.active and c.code.isdigit()
    }

    results = []
    for rule in INFERENCE_RULES:
        if rule["snomed"] in documented_snomed:
            continue

        readings = patient.labs.get(rule["lab"], [])
        if not readings:
            continue

        qualifying = _qualifying_readings(readings, rule["operator"], rule["threshold"])
        if not qualifying:
            continue

        # Some rules require the abnormal readings to occur at least N days apart
        # to distinguish a true trend from a transient abnormality.
        if rule["require_repeat"] and rule["repeat_days_apart"] > 0:
            if not _has_repeat(qualifying, rule["repeat_days_apart"]):
                continue
        elif rule["require_repeat"]:
            if len(qualifying) < 2:
                continue

        results.append(InferredCondition(
            name=rule["condition"],
            description=rule["description"],
            snomed=rule["snomed"],
            guideline=rule["source"],
            evidence=_build_evidence(qualifying, rule),
        ))

    return results


def _qualifying_readings(readings, operator: str, threshold: float) -> list:
    """Return the readings that satisfy the threshold condition for the given operator."""
    ops = {
        "<":  lambda v: v < threshold,
        "<=": lambda v: v <= threshold,
        ">":  lambda v: v > threshold,
        ">=": lambda v: v >= threshold,
    }
    check = ops.get(operator)
    return [r for r in readings if check and check(r.value)]


def _has_repeat(readings, days_apart: int) -> bool:
    """Return True if at least two readings in the list are at least days_apart days apart."""
    for i in range(len(readings)):
        for j in range(i + 1, len(readings)):
            if abs((readings[j].date - readings[i].date).days) >= days_apart:
                return True
    return False


def _build_evidence(readings, rule: dict) -> str:
    """Return a human-readable string naming the lab values and dates that triggered the rule."""
    if len(readings) == 1:
        r = readings[0]
        return f"{rule['description']} flagged by {r.description} {r.value} on {r.date}"
    first, last = readings[0], readings[-1]
    delta = abs((last.date - first.date).days)
    return (
        f"{rule['description']} flagged by {first.description} {first.value} on {first.date}"
        f" and {last.description} {last.value} on {last.date}, readings {delta} days apart"
    )
