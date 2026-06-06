"""End-to-end smoke test. Picks a high-priority multi-indication patient and runs the agent."""
import json
from data.load_fixtures import load_all, all_patients, get_ground_truth
from tools.trial_search import build_index
from agent.loop import run_agent
from agent.pipeline import enrich_patient_dict


def pick_test_patient():
    """Find a high-priority, multi-indication patient for the best demo case."""
    for p in all_patients():
        gt = get_ground_truth(p["patient_id"])
        if gt and gt.get("priority") == "high" and gt.get("is_multi_indication"):
            return p, gt
    # Fallback: any multi-indication patient
    for p in all_patients():
        gt = get_ground_truth(p["patient_id"])
        if gt and gt.get("is_multi_indication"):
            return p, gt
    # Last resort: first patient
    p = all_patients()[0]
    return p, get_ground_truth(p["patient_id"])


def make_trace_printer():
    """Console-friendly trace renderer."""
    def trace(event):
        t = event["type"]
        if t == "agent_thinking":
            content = event.get("content") or ""
            print(f"\n[THINK iter={event['iter']}]")
            print(content[:400])
        elif t == "tool_call":
            args_str = json.dumps(event["args"], default=str)[:150]
            print(f"\n[CALL  iter={event['iter']}] → {event['name']}({args_str})")
        elif t == "tool_result":
            result_str = json.dumps(event["result"], default=str)[:250]
            print(f"[RESULT iter={event['iter']}] ← {result_str}")
        elif t == "final":
            print(f"\n[FINAL] {(event.get('content') or '')[:300]}")
    return trace


def main():
    print("Loading fixtures...")
    load_all("v1")
    build_index()
    print(f"Loaded {len(all_patients())} patients.")

    patient, gt = pick_test_patient()
    print("\n" + "=" * 70)
    print(f"Testing on patient {patient['patient_id']}")
    print(f"Indications: {gt.get('indications') if gt else 'unknown'}")
    print(f"Priority: {gt.get('priority') if gt else 'unknown'}")
    print(f"Multi-indication: {gt.get('is_multi_indication') if gt else 'unknown'}")
    print(f"Expected actions: {gt.get('expected_actions') if gt else 'unknown'}")
    print("=" * 70)

    patient = enrich_patient_dict(patient)
    result = run_agent(patient, trace_callback=make_trace_printer())

    print("\n" + "=" * 70)
    print("FINAL RANKED OUTPUT")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str)[:3000])

    print("\n" + "=" * 70)
    print("EVAL SIGNALS vs GROUND TRUTH")
    print("=" * 70)
    ranked = result.get("ranked_matches", [])
    cross = result.get("cross_indication_alerts", [])
    print(f"Ranked matches returned:        {len(ranked)}")
    print(f"Cross-indication alerts:        {len(cross)}")
    print(f"GT multi-indication:            {gt.get('is_multi_indication') if gt else 'unknown'}")
    print(f"Agent detected multi-indication: {len(cross) > 0}")


if __name__ == "__main__":
    main()