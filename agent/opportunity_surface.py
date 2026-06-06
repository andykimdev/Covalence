"""Build a unified opportunity surface combining trial matches and care gaps."""
import re

from data.load_fixtures import get_trial


# Drug class → criteria text keywords for linking
_LINK_KEYWORDS: dict[str, list[str]] = {
    "ACEi/ARB": ["ace inhibitor", "arb", "angiotensin", "raas", "lisinopril",
                  "losartan", "enalapril", "ramipril", "benazepril"],
    "RAS inhibitor (ARNi preferred)": ["ace inhibitor", "arb", "angiotensin",
                                        "raas", "sacubitril", "entresto"],
    "SGLT2i": ["sglt2", "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"],
    "statin": ["statin", "lipid-lowering", "atorvastatin", "rosuvastatin", "simvastatin"],
    "beta blocker": ["beta blocker", "beta-blocker", "metoprolol", "carvedilol", "bisoprolol"],
    "MRA": ["spironolactone", "eplerenone", "mineralocorticoid", "aldosterone"],
    "GLP-1 agonist": ["glp-1", "glp1", "semaglutide", "liraglutide", "dulaglutide", "tirzepatide"],
    "antidiabetic": ["metformin", "insulin", "antidiabetic", "glycemic", "hypoglycemic"],
    "antihypertensive": ["antihypertensive", "blood pressure", "hypertension treatment"],
    "anticoagulant": ["anticoagulant", "warfarin", "apixaban", "rivaroxaban", "dabigatran"],
}

_URGENT_CONDITIONS = {"CHF", "CKD", "AFIB", "Heart Failure", "Kidney Disease"}


def _parse_match_pct(match: dict) -> int:
    """Extract 0-100 integer match score from a ranked_match entry."""
    pct_str = match.get("match_pct", "")
    if isinstance(pct_str, str):
        m = re.search(r"(\d+)%", pct_str)
        if m:
            return min(100, int(m.group(1)))
    score = match.get("score", 0) or 0
    if isinstance(score, (int, float)):
        if score >= 0:
            return min(100, max(0, int(score * 100)))
        return max(0, int((score + 1) * 30))
    return 0


def _criteria_lists(verdict_detail: dict) -> tuple[list[str], list[str]]:
    """Return (criteria_met, criteria_pending) from a verdict_detail dict."""
    met, pending = [], []
    for cv in verdict_detail.get("criteria_verdicts", []):
        text = cv.get("criterion", "")[:120]
        v = cv.get("verdict", "")
        if v == "PASS":
            met.append(text)
        elif v in ("UNKNOWN", "PARTIAL"):
            pending.append(text)
    return met, pending


def _gap_id(gap: dict) -> str:
    key = f"gap_{gap['condition']}_{gap['missing_drug']}".replace(" ", "_").lower()
    return re.sub(r"[^a-z0-9_]", "", key)


def _trial_text(match: dict) -> str:
    """Gather all searchable text for a trial match (criteria + title + summary)."""
    parts = [match.get("title", "").lower(), match.get("summary", "").lower()]
    vd = match.get("verdict_detail", {})
    for cv in vd.get("criteria_verdicts", []):
        parts.append(cv.get("criterion", "").lower())
    return " ".join(parts)


def _is_linked(gap: dict, trial_text: str) -> bool:
    """Return True if any keyword for the gap's drug class appears in the trial text."""
    drug_class = gap.get("missing_drug", "")
    keywords = _LINK_KEYWORDS.get(drug_class, [drug_class.lower()])
    return any(kw in trial_text for kw in keywords)


def _is_urgent(gap: dict) -> bool:
    cond = gap.get("condition", "")
    return any(uc.lower() in cond.lower() for uc in _URGENT_CONDITIONS)


def run_opportunity_surface(patient: dict, enriched: dict, result: dict) -> list[dict]:
    """Build unified list of opportunity objects from trial matches + care gaps.

    Parameters
    ----------
    patient:  raw patient bundle (from fixtures)
    enriched: output of enrich_patient_dict — contains care_gaps, inferred_conditions
    result:   output of run_agent — contains ranked_matches
    """
    care_gaps: list[dict] = enriched.get("care_gaps", [])
    ranked_matches: list[dict] = (result or {}).get("ranked_matches", [])

    # ── Build trial opportunity objects ──
    trial_ops: list[dict] = []
    for match in ranked_matches:
        trial_id = match.get("trial_id", "")
        vd = match.get("verdict_detail", {})
        met, pending = _criteria_lists(vd)

        trial_data = get_trial(trial_id) or {}
        conditions = trial_data.get("conditions", [])
        indication = " · ".join(conditions[:2]) if conditions else trial_id

        op: dict = {
            "id": trial_id,
            "type": "trial",
            "title": match.get("title") or trial_id,
            "indication": indication,
            "match_score": _parse_match_pct(match),
            "criteria_met": met,
            "criteria_pending": pending,
            "status": "Enrolling",
            "urgent": False,
            "linked_id": None,
            "linked_label": None,
            "action": "Review eligibility criteria and contact study coordinator",
            "_match": match,  # keep for rendering extras; stripped before returning
        }
        trial_ops.append(op)

    # ── Build care gap opportunity objects ──
    gap_ops: list[dict] = []
    for gap in care_gaps:
        cond = gap.get("condition", "")
        drug = gap.get("missing_drug", "")
        urgent = _is_urgent(gap)
        op = {
            "id": _gap_id(gap),
            "type": "care_gap",
            "title": f"{cond} — missing {drug}",
            "indication": cond,
            "match_score": 100,
            "criteria_met": [gap.get("reason", "")],
            "criteria_pending": [],
            "status": "Overdue" if urgent else "Active gap",
            "urgent": urgent,
            "linked_id": None,
            "linked_label": None,
            "action": f"Prescribe {drug} per {gap.get('guideline', 'clinical guideline')}",
        }
        gap_ops.append(op)

    # ── Detect linked pairs (deterministic keyword match) ──
    for gap_op, gap in zip(gap_ops, care_gaps):
        for trial_op, match in zip(trial_ops, ranked_matches):
            if gap_op["linked_id"] or trial_op["linked_id"]:
                continue
            if _is_linked(gap, _trial_text(match)):
                drug = gap.get("missing_drug", "")
                cond = gap.get("condition", "")
                gap_op["linked_id"] = trial_op["id"]
                gap_op["linked_label"] = (
                    f"Closing the {drug} care gap for {cond} "
                    f"may improve eligibility for this trial"
                )
                trial_op["linked_id"] = gap_op["id"]
                trial_op["linked_label"] = (
                    f"This trial may require {drug} therapy — "
                    f"patient has an active {cond} care gap as a potential blocker"
                )
                break

    # Merge: trials first, then care gaps; strip internal _match key
    all_ops = trial_ops + gap_ops
    for op in all_ops:
        op.pop("_match", None)
    return all_ops
