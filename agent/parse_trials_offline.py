"""Offline: parse raw trials into StructuredTrial objects using the LLM. Run once."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent.prompts import STRUCTURE_TRIAL_PROMPT
from agent.trial_schema import LabThreshold, StructuredTrial

load_dotenv()

_client = OpenAI(
    base_url=os.getenv("NEBIUS_BASE_URL"),
    api_key=os.getenv("NEBIUS_API_KEY"),
)
_MODEL = os.getenv("MODEL_AGENT", "meta-llama/Llama-3.3-70B-Instruct")


def _call_llm(eligibility_text: str, conditions: list[str]) -> str:
    """Send one trial's eligibility text to the LLM and return the raw response string."""
    user = f"Conditions: {', '.join(conditions)}\n\nEligibility:\n{eligibility_text}"
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": STRUCTURE_TRIAL_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def parse_one_trial(raw_trial: dict) -> StructuredTrial | None:
    """Parse a single raw trial into a StructuredTrial, or None if parsing fails."""
    try:
        text = _call_llm(
            raw_trial.get("eligibility_text", ""),
            raw_trial.get("conditions", []),
        )
        clean = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)

        lab_thresholds = {
            name: LabThreshold(
                min_value=bounds.get("min") if isinstance(bounds, dict) else None,
                max_value=bounds.get("max") if isinstance(bounds, dict) else None,
            )
            for name, bounds in data.get("lab_thresholds", {}).items()
            if bounds is not None
        }

        return StructuredTrial(
            nct_id=raw_trial["nct_id"],
            title=raw_trial.get("title", ""),
            conditions=raw_trial.get("conditions", []),
            min_age=data.get("min_age"),
            max_age=data.get("max_age"),
            sex=data.get("sex", "all"),
            required_conditions=data.get("required_conditions", []),
            excluded_conditions=data.get("excluded_conditions", []),
            lab_thresholds=lab_thresholds,
            required_medications=data.get("required_medications", []),
            excluded_medications=data.get("excluded_medications", []),
            residual_criteria=data.get("residual_criteria", []),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def parse_all_trials(input_path: str | Path, output_path: str | Path) -> int:
    """Parse every trial in the corpus and write structured_trials.json. Returns count of successes."""
    import dataclasses
    import time

    input_path, output_path = Path(input_path), Path(output_path)
    with open(input_path) as f:
        raw_trials = json.load(f)

    already_parsed: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            for entry in json.load(f):
                already_parsed[entry["nct_id"]] = entry
        print(f"Resuming — {len(already_parsed)} already parsed, skipping those.")

    structured = list(already_parsed.values())
    errors = 0
    total_start = time.time()

    for i, raw in enumerate(raw_trials):
        if raw["nct_id"] in already_parsed:
            continue
        trial_start = time.time()
        result = parse_one_trial(raw)
        trial_elapsed = time.time() - trial_start

        if result:
            structured.append(dataclasses.asdict(result))
            with open(output_path, "w") as f:
                json.dump(structured, f, indent=2)
        else:
            errors += 1

        completed = i + 1
        total_elapsed = time.time() - total_start
        avg = total_elapsed / completed
        eta = avg * (len(raw_trials) - completed)
        status = "ok" if result else "err"
        print(
            f"  [{completed:>3}/{len(raw_trials)}] {raw['nct_id']}  {status}  "
            f"{trial_elapsed:.1f}s  (avg {avg:.1f}s  eta {eta:.0f}s)",
            flush=True,
        )

    print(f"\nDone in {time.time() - total_start:.1f}s")
    print(f"Parsed {len(structured)}/{len(raw_trials)} trials ({errors} errors)")
    print(f"Wrote to {output_path}")
    return len(structured)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    parse_all_trials(
        base / "fixtures" / "clinical_trials_cleaned.json",
        base / "fixtures" / "structured_trials.json",
    )
