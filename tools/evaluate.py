"""Evaluate agent output against ground truth. Use for demo metrics."""
from data.load_fixtures import get_ground_truth


def evaluate_against_ground_truth(patient_id: str, agent_output: dict) -> dict:
    gt = get_ground_truth(patient_id)
    if not gt:
        return {"error": f"no ground truth for {patient_id}"}

    expected_actions = set(gt.get("expected_actions", []))
    gt_indications = set(i.lower() for i in gt.get("indications", []))
    gt_is_multi = gt.get("is_multi_indication", False)

    ranked = agent_output.get("ranked_matches", [])
    cross_alerts = agent_output.get("cross_indication_alerts", [])

    # Did we trigger cross-indication search for a multi-indication patient?
    detected_multi = len(cross_alerts) > 0
    multi_correct = detected_multi == gt_is_multi

    return {
        "patient_id": patient_id,
        "ground_truth_priority": gt.get("priority"),
        "ground_truth_indications": list(gt_indications),
        "ground_truth_is_multi_indication": gt_is_multi,
        "agent_detected_multi_indication": detected_multi,
        "multi_indication_correct": multi_correct,
        "ranked_count": len(ranked),
        "expected_actions": list(expected_actions),
    }