"""Pre-parse all trials in the fixture corpus and save to disk cache.

Run this once before demos or benchmarks to eliminate parse_criteria latency.
Subsequent agent runs will load criteria from cache instead of calling the LLM.

Run with:
    python3 warmup_cache.py
    python3 warmup_cache.py --workers 8   # increase parallelism
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.load_fixtures import load_all, all_trials
from tools.parse_criteria import parse_criteria, _parse_cache, save_cache


def warmup(workers: int) -> None:
    load_all("v1")
    trials = all_trials()
    total = len(trials)

    already_cached = [t for t in trials if t["nct_id"] in _parse_cache]
    to_parse = [t for t in trials if t["nct_id"] not in _parse_cache]

    print(f"Total trials:     {total}")
    print(f"Already cached:   {len(already_cached)}")
    print(f"To parse:         {len(to_parse)}")

    if not to_parse:
        print("\nAll trials already cached. Nothing to do.")
        return

    print(f"\nParsing {len(to_parse)} trials with {workers} workers...\n")

    start = time.time()
    completed = 0
    errors = 0

    times = []

    def parse_one(trial: dict) -> tuple[str, bool, float]:
        print(f"  → starting {trial['nct_id']}", flush=True)
        t0 = time.time()
        result = parse_criteria(trial["nct_id"])
        elapsed = time.time() - t0
        if "error" in result:
            print(f"\n  [error] {trial['nct_id']}: {result.get('error', '')}")
        return trial["nct_id"], "error" not in result, elapsed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_one, t): t for t in to_parse}
        for future in as_completed(futures):
            nct_id, ok, trial_time = future.result()
            times.append(trial_time)
            completed += 1
            if not ok:
                errors += 1
            status = "ok" if ok else "err"
            total_elapsed = time.time() - start
            rate = completed / total_elapsed
            eta = (len(to_parse) - completed) / rate if rate > 0 else 0
            print(f"  [{completed:>3}/{len(to_parse)}] {nct_id}  {status}  "
                  f"{trial_time:.1f}s/trial  (eta {eta:.0f}s)", end="\r")

    save_cache()
    elapsed = time.time() - start
    print(f"\n\nDone in {elapsed:.1f}s")
    print(f"Parsed:  {completed - errors}/{len(to_parse)}")
    print(f"Errors:  {errors}")
    if times:
        print(f"Avg per trial:   {sum(times)/len(times):.1f}s")
        print(f"Fastest trial:   {min(times):.1f}s")
        print(f"Slowest trial:   {max(times):.1f}s")
    print(f"Cache saved to fixtures/parse_criteria_cache.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers (default 5)")
    args = parser.parse_args()
    warmup(args.workers)
