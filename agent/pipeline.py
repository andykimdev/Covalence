"""Orchestrate Steps 1-4 and produce the enriched patient dict for the LLM agent."""
import dataclasses
import json
from pathlib import Path

from agent.detect_care_gaps import detect_care_gaps
from agent.expand_indications import expand_indications
from agent.infer_conditions import infer_conditions
from agent.ingest_patient import load_bundles, parse_bundle
from agent.models import ExpandedPatient

# Map graph node codes to readable names so all_indications uses consistent full descriptions
# rather than mixing "Type 2 diabetes mellitus" with short codes like "CKD".
_DISPLAY_NAMES = {
    "T2DM":           "Type 2 Diabetes",
    "CKD":            "Chronic Kidney Disease",
    "CHF":            "Chronic Heart Failure",
    "HYPERTENSION":   "Hypertension",
    "HYPERLIPIDEMIA": "Hyperlipidemia",
    "OBESITY":        "Obesity",
    "AFIB":           "Atrial Fibrillation",
    "ANEMIA":         "Anemia",
    "COPD":           "COPD",
    "MDD":            "Major Depressive Disorder",
}


def enrich_patient_dict(bundle: dict) -> dict:
    """Run Steps 1-4 on one raw bundle and return it with pipeline results injected as new keys."""
    parsed = parse_bundle(bundle)
    inferred = infer_conditions(parsed)
    gaps = detect_care_gaps(parsed, inferred)
    expanded = expand_indications(parsed, inferred)
    all_indications = _merge_indications(parsed, inferred, expanded)

    enriched = _slim_bundle(bundle)
    enriched["inferred_conditions"] = [dataclasses.asdict(ic) for ic in inferred]
    enriched["care_gaps"] = [dataclasses.asdict(g) for g in gaps]
    enriched["expanded_indications"] = [dataclasses.asdict(ei) for ei in expanded]
    enriched["all_indications"] = all_indications
    return enriched


def run_pipeline(bundles_path: str | Path) -> list[ExpandedPatient]:
    """Run Steps 1-4 on every patient in the JSON file and write results to fixtures/pipeline_output.json."""
    bundles_path = Path(bundles_path)
    patients = load_bundles(bundles_path)
    results = []

    for parsed in patients:
        inferred = infer_conditions(parsed)
        gaps = detect_care_gaps(parsed, inferred)
        expanded = expand_indications(parsed, inferred)
        all_indications = _merge_indications(parsed, inferred, expanded)
        results.append(ExpandedPatient(
            parsed=parsed,
            inferred_conditions=inferred,
            care_gaps=gaps,
            expanded_indications=expanded,
            all_indications=all_indications,
        ))

    output_path = bundles_path.parent.parent / "fixtures" / "pipeline_output.json"
    _write_output(results, output_path)
    print(f"Pipeline output written to {output_path}")
    return results


def _slim_bundle(bundle: dict) -> dict:
    """Return a copy of the bundle with PRAPARE observations removed and one reading per code kept."""
    slimmed = dict(bundle)

    # PRAPARE social-determinants codes all begin with "93" and are not used for trial eligibility.
    seen_codes: set[str] = set()
    kept = []
    for obs in sorted(bundle.get("observations", []), key=lambda o: o.get("date", ""), reverse=True):
        code = obs.get("code", "")
        if code.startswith("93"):
            continue
        if code not in seen_codes:
            seen_codes.add(code)
            kept.append(obs)

    slimmed["observations"] = kept
    slimmed.pop("summary", None)
    return slimmed


def _merge_indications(parsed, inferred, expanded) -> list[str]:
    """Combine documented, inferred, and expanded condition names into a sorted deduplicated list."""
    names = {c.description for c in parsed.conditions if c.active}
    names.update(ic.description for ic in inferred)
    # Expanded indications use short graph node codes so we map them to readable names
    # before merging, so the final list has consistent full descriptions throughout.
    names.update(_DISPLAY_NAMES.get(ei.name, ei.name) for ei in expanded)
    return sorted(names)


def _write_output(results: list[ExpandedPatient], path: Path) -> None:
    """Serialize pipeline results to JSON so the partner can inspect them offline."""
    output = [
        {
            "patient_id": ep.parsed.patient_id,
            "all_indications": ep.all_indications,
            "inferred_conditions": [dataclasses.asdict(ic) for ic in ep.inferred_conditions],
            "care_gaps": [dataclasses.asdict(g) for g in ep.care_gaps],
            "expanded_indications": [dataclasses.asdict(ei) for ei in ep.expanded_indications],
        }
        for ep in results
    ]
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
