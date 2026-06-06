"""Runtime deterministic trial matching with provenance tracking and weighted scoring."""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from agent.clinical_codes import GRAPH_NODE_CODES, LOINC, MED_CLASSES
from agent.prompts import RESIDUAL_CHECK_PROMPT

load_dotenv()

_client = OpenAI(
    base_url=os.getenv("NEBIUS_BASE_URL"),
    api_key=os.getenv("NEBIUS_API_KEY"),
)
_MODEL = os.getenv("MODEL_AGENT", "meta-llama/Llama-3.3-70B-Instruct")

# Reverse lookup: SNOMED int code -> controlled-vocab short code ("T2DM", "CKD", etc.)
_SNOMED_TO_SHORT: dict[int, str] = {
    code: short
    for short, codes in GRAPH_NODE_CODES.items()
    for code in codes
}


# ---------------------------------------------------------------------------
# PATIENT CANONICALIZATION
# ---------------------------------------------------------------------------

def patient_canonical_conditions(patient: dict) -> set[str]:
    """Resolve all patient conditions to controlled-vocab short codes via SNOMED lookup."""
    short_codes: set[str] = set()

    for c in patient.get("conditions", []):
        if c.get("active") and str(c.get("code", "")).isdigit():
            short = _SNOMED_TO_SHORT.get(int(c["code"]))
            if short:
                short_codes.add(short)

    for ic in patient.get("inferred_conditions", []):
        short = _SNOMED_TO_SHORT.get(ic.get("snomed"))
        if short:
            short_codes.add(short)

    for ei in patient.get("expanded_indications", []):
        short_codes.add(ei["name"])

    return short_codes


# ---------------------------------------------------------------------------
# CANDIDATE FILTERING (multi-indication)
# ---------------------------------------------------------------------------

def get_candidates(patient: dict, structured_trials: list[dict]) -> list[dict]:
    """Return trials whose required_conditions overlap the patient's canonical condition codes."""
    patient_codes = patient_canonical_conditions(patient)

    candidates = []
    for trial in structured_trials:
        trial_codes = set(trial.get("required_conditions", []))
        matched = patient_codes & trial_codes
        if matched:
            c = dict(trial)
            c["surfaced_by"] = sorted(matched)
            candidates.append(c)
    return candidates


# ---------------------------------------------------------------------------
# DETERMINISTIC MATCHING
# ---------------------------------------------------------------------------

def _latest_lab(patient: dict, loinc_code: str) -> float | None:
    """Return the most recent numeric value for a given LOINC code, or None if absent."""
    readings = [o for o in patient.get("observations", []) if o.get("code") == loinc_code]
    if not readings:
        return None
    readings.sort(key=lambda r: r.get("date", ""))
    try:
        return float(readings[-1]["value"])
    except (KeyError, TypeError, ValueError):
        return None


def _patient_on_med_class(patient_meds_str: str, med_class: str) -> bool:
    """Return True if any keyword for the given med class appears in the patient's medication strings."""
    keywords = MED_CLASSES.get(med_class, [med_class.lower()])
    return any(kw in patient_meds_str for kw in keywords)


def match_patient_to_trial(patient: dict, trial: dict) -> list[dict]:
    """Return per-criterion verdicts (PASS/FAIL/UNKNOWN) using deterministic boolean logic."""
    verdicts = []

    def add(criterion: str, verdict: str, rationale: str = ""):
        verdicts.append({"criterion": criterion, "verdict": verdict, "rationale": rationale})

    age = (patient.get("demographics") or {}).get("age") or patient.get("age")

    if trial.get("min_age") is not None:
        if age is None:
            add(f"age >= {trial['min_age']}", "UNKNOWN", "Patient age not available")
        else:
            v = "PASS" if age >= trial["min_age"] else "FAIL"
            add(f"age >= {trial['min_age']}", v, f"Patient age is {age}")

    if trial.get("max_age") is not None:
        if age is None:
            add(f"age <= {trial['max_age']}", "UNKNOWN", "Patient age not available")
        else:
            v = "PASS" if age <= trial["max_age"] else "FAIL"
            add(f"age <= {trial['max_age']}", v, f"Patient age is {age}")

    sex = trial.get("sex", "all")
    if sex != "all":
        pg = ((patient.get("demographics") or {}).get("gender") or patient.get("gender", "")).lower()
        v = "PASS" if pg.startswith(sex[0]) else ("UNKNOWN" if not pg else "FAIL")
        add(f"sex = {sex}", v, f"Patient gender: {pg or 'unknown'}")

    patient_codes = patient_canonical_conditions(patient)
    for req in trial.get("required_conditions", []):
        v = "PASS" if req in patient_codes else "FAIL"
        add(f"requires {req}", v, f"Patient codes: {sorted(patient_codes)}")
    for exc in trial.get("excluded_conditions", []):
        if exc in patient_codes:
            add(f"excludes {exc}", "FAIL", f"Patient has {exc}")
        else:
            # Absence from the record is not the same as confirmed absence.
            # A real physician must rule this out — flag UNKNOWN rather than PASS.
            add(f"excludes {exc}", "UNKNOWN", f"No confirmed {exc} in record — requires clinical verification")

    for lab_name, bounds in trial.get("lab_thresholds", {}).items():
        loinc = LOINC.get(lab_name.upper())
        value = _latest_lab(patient, loinc) if loinc else None
        lo = bounds.get("min_value") if isinstance(bounds, dict) else getattr(bounds, "min_value", None)
        hi = bounds.get("max_value") if isinstance(bounds, dict) else getattr(bounds, "max_value", None)
        if value is None:
            add(f"{lab_name} in [{lo},{hi}]", "UNKNOWN", f"No {lab_name} reading in patient record")
        else:
            ok = (lo is None or value >= lo) and (hi is None or value <= hi)
            add(f"{lab_name} in [{lo},{hi}]", "PASS" if ok else "FAIL", f"{lab_name} = {value}")

    meds_str = " ".join(patient.get("active_medications", [])).lower()
    for med in trial.get("required_medications", []):
        v = "PASS" if _patient_on_med_class(meds_str, med) else "FAIL"
        add(f"on {med}", v)
    for med in trial.get("excluded_medications", []):
        if _patient_on_med_class(meds_str, med):
            v = "FAIL"
        else:
            # Not currently on it doesn't mean never on it — leave for physician to confirm.
            v = "UNKNOWN"
        add(f"not on {med}", v)

    return verdicts


# ---------------------------------------------------------------------------
# CARE GAP ANNOTATION
# ---------------------------------------------------------------------------

def annotate_with_care_gaps(verdicts: list[dict], patient: dict) -> list[dict]:
    """Mark medication-requirement FAILs as resolvable if they match a known care gap."""
    gap_lookup = {g["missing_drug"].lower(): g for g in patient.get("care_gaps", [])}
    for v in verdicts:
        v["resolvable"] = False
        v["care_gap"] = None
        if v["verdict"] == "FAIL" and v["criterion"].startswith("on "):
            required = v["criterion"][3:].lower()
            for gap_drug, gap in gap_lookup.items():
                if gap_drug in required or required in gap_drug:
                    v["resolvable"] = True
                    v["care_gap"] = gap
                    break
    return verdicts


# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------

def compute_provenance(patient: dict, short_code: str) -> tuple[str, float]:
    """Return (tier, weight) for how a condition short code was derived.

    documented -> 1.0, inferred -> 0.7, expanded -> its expansion_score.
    """
    documented_codes: set[str] = set()
    for c in patient.get("conditions", []):
        if c.get("active") and str(c.get("code", "")).isdigit():
            s = _SNOMED_TO_SHORT.get(int(c["code"]))
            if s:
                documented_codes.add(s)
    if short_code in documented_codes:
        return "documented", 1.0

    inferred_codes: set[str] = set()
    for ic in patient.get("inferred_conditions", []):
        s = _SNOMED_TO_SHORT.get(ic.get("snomed"))
        if s:
            inferred_codes.add(s)
    if short_code in inferred_codes:
        return "inferred", 0.7

    for ei in patient.get("expanded_indications", []):
        if ei["name"] == short_code:
            return "expanded", ei.get("expansion_score", 0.3)

    return "unknown", 0.5


def best_provenance(patient: dict, trial: dict) -> tuple[str, float]:
    """Across all short codes that surfaced this trial, return the strongest provenance."""
    best_tier, best_weight = "unknown", 0.0
    tier_rank = {"documented": 3, "inferred": 2, "expanded": 1, "unknown": 0}
    for code in trial.get("surfaced_by", []):
        tier, weight = compute_provenance(patient, code)
        if weight > best_weight or tier_rank.get(tier, 0) > tier_rank.get(best_tier, 0):
            best_tier, best_weight = tier, weight
    return best_tier, best_weight


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score_match(verdicts: list[dict], provenance_weight: float) -> tuple[float, float]:
    """Return (score, adjusted_score), both multiplied by provenance weight.

    adjusted_score assumes all resolvable care-gap FAILs are fixed.
    """
    total = len(verdicts)
    if total == 0:
        return 0.0, 0.0
    passes = sum(1 for v in verdicts if v["verdict"] == "PASS")
    unknowns = sum(1 for v in verdicts if v["verdict"] == "UNKNOWN")
    resolvable = sum(1 for v in verdicts if v.get("resolvable"))
    hard_fail = any(v["verdict"] == "FAIL" and not v.get("resolvable") for v in verdicts)

    base = (passes / total) - 0.3 * (unknowns / total)
    base -= 1.0 if hard_fail else 0.0
    base -= 0.1 * (resolvable / total)

    adjusted = ((passes + resolvable) / total) - 0.3 * (unknowns / total)
    adjusted -= 1.0 if hard_fail else 0.0

    return round(base * provenance_weight, 3), round(adjusted * provenance_weight, 3)


# ---------------------------------------------------------------------------
# RESIDUAL LLM CHECK
# ---------------------------------------------------------------------------

def check_residual(
    patient: dict,
    residual_criteria: list[str],
    trial_id: str = "",
    trace_callback=None,
) -> list[dict]:
    """LLM-check free-text criteria that could not be structured. Only call on finalists."""
    if not residual_criteria:
        return []

    if trace_callback:
        trace_callback({
            "type": "residual_check_start",
            "trial_id": trial_id,
            "criteria_count": len(residual_criteria),
            "criteria": residual_criteria,
        })

    age = (patient.get("demographics") or {}).get("age") or patient.get("age")
    user = (
        f"Patient: age {age}, conditions {patient.get('all_indications')}, "
        f"meds {patient.get('active_medications')}.\n\n"
        f"Residual criteria:\n" + "\n".join(f"- {c}" for c in residual_criteria)
    )
    try:
        resp = _client.chat.completions.create(
            model=_MODEL, temperature=0,
            messages=[
                {"role": "system", "content": RESIDUAL_CHECK_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        clean = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        verdicts = json.loads(clean).get("residual_verdicts", [])
    except Exception:
        verdicts = [
            {"criterion": c, "verdict": "UNKNOWN", "rationale": "residual check failed"}
            for c in residual_criteria
        ]

    if trace_callback:
        trace_callback({
            "type": "residual_check_result",
            "trial_id": trial_id,
            "verdicts": verdicts,
        })

    return verdicts


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def _is_cross_indication(patient: dict, trial: dict) -> bool:
    """Return True if the trial was surfaced by an inferred or expanded indication rather than documented."""
    documented_codes: set[str] = set()
    for c in patient.get("conditions", []):
        if c.get("active") and str(c.get("code", "")).isdigit():
            s = _SNOMED_TO_SHORT.get(int(c["code"]))
            if s:
                documented_codes.add(s)
    for code in trial.get("surfaced_by", []):
        if code not in documented_codes:
            return True
    return False


def _combined_score(
    structured_verdicts: list[dict],
    residual_verdicts: list[dict],
    provenance_weight: float,
) -> tuple[float, float]:
    """Two-stage weighted score:
      - structured criteria weighted 0.6, residual weighted 0.4
      - UNKNOWN/Unclear count as soft penalty (-0.3 each per proportion)
      - provenance weight applied last
      - adjusted_score assumes resolvable gaps are fixed
    """
    def _counts(verdicts: list[dict], pass_kw: str = "PASS", fail_kw: str = "FAIL", unk_kw: str = "UNKNOWN"):
        passes = sum(1 for v in verdicts if v.get("verdict") == pass_kw)
        unknowns = sum(1 for v in verdicts if v.get("verdict") == unk_kw)
        resolvable = sum(1 for v in verdicts if v.get("resolvable"))
        return passes, unknowns, resolvable, len(verdicts)

    s_pass, s_unk, s_res, s_total = _counts(structured_verdicts)
    r_pass, r_unk, r_res, r_total = _counts(
        residual_verdicts,
        pass_kw="PASS", fail_kw="FAIL", unk_kw="UNKNOWN",
    )

    s_rate = (s_pass / s_total) if s_total else 0.0
    r_rate = (r_pass / r_total) if r_total else s_rate
    unk_rate = ((s_unk + r_unk) / (s_total + r_total)) if (s_total + r_total) else 0.0
    res_rate = (s_res / s_total) if s_total else 0.0

    base = (0.6 * s_rate + 0.4 * r_rate) - 0.3 * unk_rate - 0.1 * res_rate
    s_adj_rate = ((s_pass + s_res) / s_total) if s_total else 0.0
    adjusted = (0.6 * s_adj_rate + 0.4 * r_rate) - 0.3 * unk_rate

    return round(base * provenance_weight, 3), round(adjusted * provenance_weight, 3)


def process_patient(
    patient: dict,
    structured_trials: list[dict],
    top_n: int = 10,
    skip_residual: bool = False,
    trace_callback=None,
) -> list[dict]:
    """Two-stage matching pipeline.

    Stage 1 (deterministic): drop trials with any hard non-resolvable structured FAIL.
    Stage 2 (residual LLM): check free-text criteria on survivors; drop if hard residual FAIL.
    Final score: weighted combination of structured + residual pass rates × provenance weight.
    """
    candidates = get_candidates(patient, structured_trials)

    # ── Stage 1: deterministic matching + filter ──
    survivors = []
    for trial in candidates:
        verdicts = match_patient_to_trial(patient, trial)
        verdicts = annotate_with_care_gaps(verdicts, patient)

        hard_fail = any(
            v["verdict"] == "FAIL" and not v.get("resolvable")
            for v in verdicts
        )
        if hard_fail:
            continue

        tier, weight = best_provenance(patient, trial)
        survivors.append({
            "trial": trial,
            "verdicts": verdicts,
            "provenance": tier,
            "provenance_weight": weight,
        })

    # ── Stage 2: residual LLM check on survivors ──
    results = []
    for s in survivors:
        trial = s["trial"]
        verdicts = s["verdicts"]
        tier = s["provenance"]
        weight = s["provenance_weight"]
        residual_verdicts: list[dict] = []

        if not skip_residual and trial.get("residual_criteria"):
            residual_verdicts = check_residual(
                patient,
                trial["residual_criteria"],
                trial_id=trial["nct_id"],
                trace_callback=trace_callback,
            )
            # Drop if any residual FAIL that is not resolvable (hard requirement)
            residual_hard_fail = any(
                v.get("verdict") == "FAIL" and not v.get("resolvable")
                for v in residual_verdicts
            )
            if residual_hard_fail:
                continue

        score, adjusted = _combined_score(verdicts, residual_verdicts, weight)

        s_pass = sum(1 for v in verdicts if v["verdict"] == "PASS")
        s_total = len(verdicts)
        r_pass = sum(1 for v in residual_verdicts if v.get("verdict") == "PASS")
        r_total = len(residual_verdicts)
        resolvable = sum(1 for v in verdicts if v.get("resolvable"))

        all_verdicts = verdicts + [
            {**rv, "source": "residual"} for rv in residual_verdicts
        ]
        total_pass = s_pass + r_pass
        total_crit = s_total + r_total

        results.append({
            "nct_id": trial["nct_id"],
            "title": trial.get("title", ""),
            "surfaced_by": trial.get("surfaced_by", []),
            "provenance": tier,
            "provenance_weight": weight,
            "needs_double_check": tier in ("inferred", "expanded"),
            "cross_indication": _is_cross_indication(patient, trial),
            "score": score,
            "adjusted_score": adjusted,
            "match_pct": f"{total_pass}/{total_crit}" if total_crit else "0/0",
            "adjusted_match_pct": f"{total_pass + resolvable}/{total_crit}" if total_crit else "0/0",
            "verdicts": all_verdicts,
            "residual_criteria": trial.get("residual_criteria", []),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]
