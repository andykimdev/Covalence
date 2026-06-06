"""Walk the disease graph from known conditions and score candidate expansions with lab evidence."""
import networkx as nx

from agent.clinical_codes import (
    DISEASE_GRAPH_EDGES,
    EXPANSION_THRESHOLD,
    GRAPH_NODE_CODES,
    LAB_EVIDENCE_RANGES,
)
from agent.models import ExpandedIndication, InferredCondition, ParsedPatient

# Build the directed graph once at import time so it is not reconstructed for every patient.
_GRAPH = nx.DiGraph()
for _source, _target, _weight in DISEASE_GRAPH_EDGES:
    _GRAPH.add_edge(_source, _target, weight=_weight)


def expand_indications(patient: ParsedPatient, inferred: list[InferredCondition]) -> list[ExpandedIndication]:
    """Return candidate indications the patient may be developing based on comorbidity graph evidence."""
    known_nodes = _resolve_known_nodes(patient, inferred)

    # Walk one hop of outgoing edges from every known condition and score each candidate.
    candidates: dict[str, ExpandedIndication] = {}
    for node in known_nodes:
        if node not in _GRAPH:
            continue
        for target, edge_data in _GRAPH[node].items():
            if target in known_nodes:
                continue
            score = round(edge_data["weight"] * _lab_evidence_score(patient, target), 3)
            if score <= EXPANSION_THRESHOLD:
                continue
            # When multiple paths lead to the same candidate, keep the highest-scoring entry.
            if target not in candidates or score > candidates[target].expansion_score:
                candidates[target] = ExpandedIndication(
                    name=target,
                    confidence=_confidence_tier(score),
                    expansion_score=score,
                    recommendation=_recommendation(target),
                )

    return list(candidates.values())


def _resolve_known_nodes(patient: ParsedPatient, inferred: list[InferredCondition]) -> set[str]:
    """Map patient SNOMED codes to graph node names using GRAPH_NODE_CODES."""
    patient_codes = {int(c.code) for c in patient.conditions if c.active and c.code.isdigit()}
    for ic in inferred:
        patient_codes.add(ic.snomed)
    return {name for name, codes in GRAPH_NODE_CODES.items() if patient_codes.intersection(codes)}


def _lab_evidence_score(patient: ParsedPatient, target: str) -> float:
    """Return a 0.0 to 1.0 score for how close the patient's labs are to the diagnostic threshold."""
    if target not in LAB_EVIDENCE_RANGES:
        return 0.3
    config = LAB_EVIDENCE_RANGES[target]
    readings = patient.labs.get(config["lab"], [])
    if not readings:
        return config["missing_data_score"]
    latest = readings[-1].value
    for min_val, max_val, score in config["ranges"]:
        above_min = min_val is None or latest >= min_val
        below_max = max_val is None or latest < max_val
        if above_min and below_max:
            return score
    return config["missing_data_score"]


def _confidence_tier(score: float) -> str:
    """Map a numeric expansion score to a confidence label."""
    if score > 0.35:
        return "high"
    if score > 0.20:
        return "medium"
    return "low"


def _recommendation(target: str) -> str:
    """Return a follow-up recommendation for the given expanded indication."""
    recommendations = {
        "CKD":            "Repeat eGFR in 90 days to confirm trend.",
        "T2DM":           "Order fasting glucose and A1c to rule out diabetes.",
        "HYPERTENSION":   "Repeat blood pressure on two separate visits.",
        "HYPERLIPIDEMIA": "Order fasting lipid panel.",
        "CHF":            "Order NT-proBNP and echocardiogram if elevated.",
        "ANEMIA":         "Order complete blood count.",
        "OBESITY":        "Measure BMI at next visit.",
        "AFIB":           "Order ECG if palpitations are reported.",
    }
    return recommendations.get(target, f"Baseline screening recommended for {target}.")
