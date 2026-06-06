"""Load team fixtures into a normalized in-memory store."""
import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Module-level store
_patients: list[dict] = []
_patients_by_id: dict[str, dict] = {}
_trials: list[dict] = []
_trials_by_nct: dict[str, dict] = {}
_ground_truth: dict[str, dict] = {}


def load_all(snapshot: str = "v1"):
    """Call once at startup. Loads patients, trials (snapshot v1 or v2), and ground truth."""
    global _patients, _patients_by_id, _trials, _trials_by_nct, _ground_truth

    with open(FIXTURES / "patient_bundles.json") as f:
        _patients = json.load(f)
    _patients_by_id = {p["patient_id"]: p for p in _patients}

    trials_file = f"clinical_trials_{snapshot}.json" if snapshot != "v1" else "clinical_trials_cleaned.json"
    with open(FIXTURES / trials_file) as f:
        _trials = json.load(f)
    _trials_by_nct = {t["nct_id"]: t for t in _trials}

    with open(FIXTURES / "ground_truth.json") as f:
        gt_list = json.load(f)
    _ground_truth = {g["patient_id"]: g for g in gt_list}


def get_patient(patient_id: str) -> dict | None:
    return _patients_by_id.get(patient_id)


def get_trial(nct_id: str) -> dict | None:
    return _trials_by_nct.get(nct_id)


def get_ground_truth(patient_id: str) -> dict | None:
    return _ground_truth.get(patient_id)


def all_patients() -> list[dict]:
    return _patients


def all_trials() -> list[dict]:
    return _trials