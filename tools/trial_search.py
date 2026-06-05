"""BM25 retrieval over the local trial corpus."""
from rank_bm25 import BM25Okapi
from data.load_fixtures import all_trials, get_trial as _get_trial

_documents: list[list[str]] = []
_trial_ids: list[str] = []
_bm25: BM25Okapi | None = None


def build_index():
    """Build BM25 index from currently loaded trials. Call after load_all()."""
    global _documents, _trial_ids, _bm25
    _documents = []
    _trial_ids = []
    for trial in all_trials():
        doc_parts = [
            trial.get("title", ""),
            trial.get("official_title", ""),
            " ".join(trial.get("conditions", [])),
            " ".join(trial.get("keywords", [])),
            trial.get("brief_summary", ""),
            trial.get("eligibility_text", ""),
        ]
        doc = " ".join(doc_parts).lower().split()
        _documents.append(doc)
        _trial_ids.append(trial["nct_id"])
    _bm25 = BM25Okapi(_documents)


_MAX_RESULTS = 10

def trial_search(query: str, top_n: int = _MAX_RESULTS) -> dict:
    if _bm25 is None:
        build_index()
    top_n = min(int(top_n), _MAX_RESULTS)
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    results = []
    for i in top_indices:
        if scores[i] > 0:
            trial = _get_trial(_trial_ids[i])
            results.append({
                "nct_id": _trial_ids[i],
                "title": trial.get("title", ""),
                "conditions": trial.get("conditions", []),
                "phase": trial.get("phase", []),
                "score": float(scores[i]),
            })
    return {"query": query, "trials": results}


def get_trial(trial_id: str):
    return _get_trial(trial_id)