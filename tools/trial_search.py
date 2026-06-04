"""BM25 retrieval over the local trial corpus. No LLM, no API calls."""
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Module-level state: load once at import, used by all calls

#initialize module-level state variables
#stores all trial records loaded from the fixtures file, each trial is a dictionary
_trials: list[dict] = []
#Stores the tokenized version of each trial's title and criteria for BM25 scoring
# each document is a list of strings
_documents: list[list[str]] = []
# Stores the BM25 model (inverted index + scoring logic)
_bm25: BM25Okapi | None = None
# Tracks which trials dataset is currently loaded.
_active_snapshot: str = "v1"


def load_trials(snapshot: str = "v1"):
    """Load trial fixtures from JSON and rebuild the BM25 index. Call this at app startup."""
    #tells Python we're modifying the module-level state variables not creating function-scoped local variables
    global _trials, _documents, _bm25, _active_snapshot
    #builds path for a given set of trials snapshot and opens it as a file object
    path = FIXTURES_DIR / f"trials_{snapshot}.json"
    with open(path) as f:
        _trials = json.load(f)
    _documents = [] #reset the documents list to empty
    #iterate over each trial in the trials list and tokenize the title and criteria for BM25 scoring
    #each document is a list of strings that are the title and criteria of the trial
    for trial in _trials:
        doc = " ".join([
            trial.get("title", ""),
            " ".join(trial.get("inclusion_criteria", [])),
            " ".join(trial.get("exclusion_criteria", [])),
            " ".join(trial.get("indication_tags", [])),
        ])
        _documents.append(doc.lower().split())

    #Builds BM25 model over tokenized documents (enables scoring queries against all trials)
    _bm25 = BM25Okapi(_documents)
    #update the active snapshot to the new snapshot
    _active_snapshot = snapshot


def get_trial(trial_id: str) -> dict | None:
    """Look up one trial by NCT ID in the trials list of the current active snapshot"""
    for trial in _trials:
        if trial["nct_id"] == trial_id:
            return trial
    return None


def trial_search(query: str, top_n: int = 14) -> dict:
    """Return the top N trial IDs by BM25 score against the query."""
    if _bm25 is None: #if the BM25 index is not loaded, load the trials and create the index
        load_trials(_active_snapshot)
    tokens = query.lower().split() #tokenize the query for BM25 scoring (create a list of lowercased strings split by whitespace from the query)
    #get the BM25 scores for the query, scores is a list of floats that are the BM25 scores for the query
    # essentially compares the query to each document in the documents list and each document is correlated to a trial in the trials list and scores how well each trial matches the query based on word overlap and weighting
    scores = _bm25.get_scores(tokens) #get the BM25 scores for the query
    #sort the indices of the scores in descending order and get the top N indices
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    results = [] #initialize an empty list to store the results
    #iterate over the top N indices and add the trial data for top N trials to the results list
    for i in top_indices:
        if scores[i] > 0: #if the score is greater than 0, add the trial data to the results list
            results.append({
                "nct_id": _trials[i]["nct_id"],
                "title": _trials[i].get("title", ""),
                "indication": _trials[i].get("indication", ""),
                "score": float(scores[i]),
            })

    #return the results as a dictionary
    return {"query": query, "snapshot": _active_snapshot, "trials": results} 