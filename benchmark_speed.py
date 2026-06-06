"""Benchmark agent speed across a sample of patients.

Run with:
    python3 benchmark_speed.py
    python3 benchmark_speed.py --n 10        # test 10 patients
    python3 benchmark_speed.py --priority high  # only high-priority patients
"""
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.load_fixtures import load_all, all_patients, get_ground_truth
from tools.trial_search import build_index
from agent.pipeline import enrich_patient_dict
from agent.loop import run_agent


def pick_patients(n: int, priority: str | None) -> list[dict]:
    """Select up to n patients, optionally filtered by ground truth priority."""
    all_p = all_patients()
    if priority:
        all_p = [p for p in all_p if (get_ground_truth(p["patient_id"]) or {}).get("priority") == priority]
    return all_p[:n]


def run_benchmark(patients: list[dict]) -> None:
    total_start = time.time()
    results = []

    for i, patient in enumerate(patients):
        pid = patient["patient_id"]
        gt = get_ground_truth(pid) or {}
        indications = ", ".join(gt.get("indications", []))[:50] or "unknown"

        print(f"\n[{i+1}/{len(patients)}] {pid[:8]}...  {indications}")

        enriched = enrich_patient_dict(patient)

        tool_calls = []
        def trace(event):
            if event.get("type") == "tool_call":
                tool_calls.append(event.get("name"))

        start = time.time()
        result = run_agent(enriched, trace_callback=trace)
        elapsed = time.time() - start

        if "ranked_matches" in result:
            outcome = f"success  {len(result['ranked_matches'])} match(es)"
        else:
            outcome = result.get("outcome", "unknown")

        from collections import Counter
        call_summary = ", ".join(f"{k}×{v}" for k, v in Counter(tool_calls).items())
        print(f"  time: {elapsed:.1f}s  outcome: {outcome}  calls: [{call_summary}]")
        results.append({"pid": pid[:8], "elapsed": elapsed, "outcome": outcome})

    total_elapsed = time.time() - total_start
    avg = sum(r["elapsed"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"Patients tested:  {len(results)}")
    print(f"Total time:       {total_elapsed:.1f}s")
    print(f"Average per patient: {avg:.1f}s")
    print(f"Fastest:          {min(r['elapsed'] for r in results):.1f}s")
    print(f"Slowest:          {max(r['elapsed'] for r in results):.1f}s")
    successes = sum(1 for r in results if r["outcome"].startswith("success"))
    print(f"Success rate:     {successes}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of patients to test (default 5)")
    parser.add_argument("--priority", type=str, default=None, help="Filter by GT priority: high, medium, low")
    args = parser.parse_args()

    print("Loading fixtures...")
    load_all("v1")
    build_index()

    patients = pick_patients(args.n, args.priority)
    if not patients:
        print(f"No patients found with priority={args.priority}")
        sys.exit(1)

    print(f"Benchmarking {len(patients)} patient(s)...\n")
    run_benchmark(patients)
