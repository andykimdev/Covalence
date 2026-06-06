"""End-to-end smoke test. Picks a high-priority multi-indication patient and runs the match engine."""
import json
from data.load_fixtures import load_all, all_patients, get_ground_truth
from agent.pipeline import match_patient_end_to_end


def pick_test_patient():
    """Find a high-priority, multi-indication patient for the best demo case."""
    for p in all_patients():
        gt = get_ground_truth(p["patient_id"])
        if gt and gt.get("priority") == "high" and gt.get("is_multi_indication"):
            return p, gt
    for p in all_patients():
        gt = get_ground_truth(p["patient_id"])
        if gt and gt.get("is_multi_indication"):
            return p, gt
    p = all_patients()[0]
    return p, get_ground_truth(p["patient_id"])


def main():
    print("Loading fixtures...")
    load_all("v1")
    print(f"Loaded {len(all_patients())} patients.")

    patient, gt = pick_test_patient()
    print("\n" + "=" * 70)
    print(f"Testing on patient {patient['patient_id']}")
    print(f"Indications: {gt.get('indications') if gt else 'unknown'}")
    print(f"Priority: {gt.get('priority') if gt else 'unknown'}")
    print(f"Multi-indication: {gt.get('is_multi_indication') if gt else 'unknown'}")
    print(f"Expected actions: {gt.get('expected_actions') if gt else 'unknown'}")
    print("=" * 70)

    results = match_patient_end_to_end(patient)

    print("\n" + "=" * 70)
    print("RANKED MATCHES")
    print("=" * 70)
    print(json.dumps(results[:3], indent=2, default=str)[:3000])

    print("\n" + "=" * 70)
    print("EVAL SIGNALS vs GROUND TRUTH")
    print("=" * 70)
    cross = [r for r in results if r.get("cross_indication")]
    needs_check = [r for r in results if r.get("needs_double_check")]
    print(f"Ranked matches returned:        {len(results)}")
    print(f"Cross-indication matches:       {len(cross)}")
    print(f"Needs double-check (inferred/expanded): {len(needs_check)}")
    print(f"GT multi-indication:            {gt.get('is_multi_indication') if gt else 'unknown'}")
    print(f"Engine detected cross-indication: {len(cross) > 0}")

    if results:
        top = results[0]
        print(f"\nTop match: {top['nct_id']} score={top['score']} provenance={top['provenance']}")
        print(f"  Match: {top['match_pct']}  Adjusted: {top['adjusted_match_pct']}")


if __name__ == "__main__":
    main()
