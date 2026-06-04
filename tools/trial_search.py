"""BM25 retrieval over the local trial corpus. No LLM, no API calls."""
import json
import os
from pathlib import Path
from rank_bm25 import BM25Okapi

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Module-level state: load once at import, used by all calls
_trials: list[dict] = []
_documents: list[list[str]] = []
_bm25: BM25Okapi | None = None
_active_snapshot: str = "v1"


def load_trials(snapshot: str = "v1"):
    """Load trial fixtures from JSON and rebuild the BM25 index. Call this at app startup."""
    global _trials, _documents, _bm25, _active_snapshot
    path = FIXTURES_DIR / f"trials_{snapshot}.json"
    with open(path) as f:
        _trials = json.load(f)
    _documents = []
    for trial in _trials:
        doc = " ".join([
            trial.get("title", ""),
            " ".join(trial.get("inclusion_criteria", [])),
            " ".join(trial.get("exclusion_criteria", [])),
            " ".join(trial.get("indication_tags", [])),
        ])
        _documents.append(doc.lower().split())
    _bm25 = BM25Okapi(_documents)
    _active_snapshot = snapshot


def get_trial(trial_id: str) -> dict | None:
    """Look up one trial by NCT ID."""
    for trial in _trials:
        if trial["nct_id"] == trial_id:
            return trial
    return None


def trial_search(query: str, top_n: int = 14) -> dict:
    """Return the top N trial IDs by BM25 score against the query."""
    if _bm25 is None:
        load_trials(_active_snapshot)
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    results = []
    for i in top_indices:
        if scores[i] > 0:
            results.append({
                "nct_id": _trials[i]["nct_id"],
                "title": _trials[i].get("title", ""),
                "indication": _trials[i].get("indication", ""),
                "score": float(scores[i]),
            })
    return {"query": query, "snapshot": _active_snapshot, "trials": results}