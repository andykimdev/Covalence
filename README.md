# bio-agent-hackathon

Agentic system that matches patients to clinical trials. The agent reasons over a FHIR-style patient bundle, searches a local trial corpus, parses eligibility criteria, evaluates criterion-by-criterion verdicts (PASS / FAIL / UNKNOWN), validates results against missing patient data, and returns a ranked top-3 shortlist with clinician-facing rationales.

**Track 1 · Clinical Decision Support · Pfizer Medical Intelligence sub-track**

## Prerequisites

- Python 3.10+ (3.12 recommended)
- [Nebius Token Factory](https://nebius.com) API key
- Fixtures (provided): `patient_bundles.json`, `clinical_trials_cleaned.json`, `ground_truth.json`

## Setup

```bash
git clone <repo-url>
cd bio-agent-hckthon

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root (gitignored):

```env
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
NEBIUS_API_KEY=your_api_key_here

MODEL_AGENT=Qwen/Qwen3.5-397B-A17B-fast
MODEL_PARSE=meta-llama/Llama-3.3-70B-Instruct
MODEL_CHECK=meta-llama/Llama-3.3-70B-Instruct
MODEL_RANK=meta-llama/Llama-3.3-70B-Instruct
```

Place the fixture files under `fixtures/`:

```
fixtures/
  patient_bundles.json
  clinical_trials_cleaned.json
  ground_truth.json
```

## Run the end-to-end test

```bash
python test_end_to_end.py
```

This script automatically:
1. Loads all 61 patients, 167 trials, and ground truth labels
2. Builds the BM25 retrieval index
3. Picks a high-priority multi-indication patient
4. Runs the full agent loop (search → parse → check → validate → rank)
5. Prints a streaming trace of every tool call and result
6. Reports eval signals against ground truth

Expected output ends with `Ranked matches returned: 3` (or similar).

## Run the Streamlit demo

```bash
streamlit run streamlit_app/app.py
```

Open the URL printed in your terminal. Pick a patient, click Find Matches, watch the reasoning trace render alongside the ranked output.

## Use the agent in your own script

```python
from data.load_fixtures import load_all, get_patient
from tools.trial_search import build_index
from agent.loop import run_agent

# Always do these two before run_agent
load_all("v1")
build_index()

patient = get_patient("4f083ce3-f12b-bb4b-7353-e17f0cd55b0a")

def trace(event):
    print(event["type"], event.get("name") or event.get("content", "")[:80])

result = run_agent(patient, trace_callback=trace)
print(result)
```

The agent loop calls Nebius chat completions with tool calling, runs up to 25 iterations, and returns a JSON object with `ranked_matches`, `missing_data_summary`, and `cross_indication_alerts`.

## Test trial search alone

BM25 search over the local corpus (no LLM call):

```python
from data.load_fixtures import load_all
from tools.trial_search import build_index, trial_search

load_all("v1")
build_index()

results = trial_search("HER2 positive metastatic breast cancer", top_n=5)
print(results)
```

## Project structure

```
agent/
  loop.py                # Agent loop (LLM orchestration + tool execution)
  prompts.py             # System prompts for agent and each tool
data/
  load_fixtures.py       # Loads patient_bundles, trials, and ground_truth
tools/
  __init__.py            # Tool schemas + execute_tool dispatcher
  trial_search.py        # BM25 retrieval
  parse_criteria.py      # Structured inclusion/exclusion extraction
  check_eligibility.py   # Per-criterion PASS/FAIL/UNKNOWN verdicts
  validate_verdicts.py   # Self-correction for PASS-on-null-field hallucinations
  rank_rationale.py      # Deterministic scoring + LLM rationale
streamlit_app/
  app.py                 # Demo UI with live reasoning trace
fixtures/
  patient_bundles.json   # 61 FHIR-style synthetic patients
  clinical_trials_cleaned.json  # 167 RECRUITING trials
  ground_truth.json      # Eval labels per patient
test_end_to_end.py       # Smoke test entry point
```

## Agent pipeline

1. **trial_search** — BM25 keyword retrieval over local trial corpus (no LLM). The agent can call this multiple times for cross-indication search.
2. **parse_criteria** — extract structured inclusion/exclusion from trial protocol text (Llama 3.3 70B).
3. **check_eligibility** — evaluate patient against each criterion. Returns PASS, FAIL, or UNKNOWN with rationale per criterion (Llama 3.3 70B).
4. **validate_verdicts** — pure Python self-check. Flags PASS verdicts where the patient bundle lacks the relevant lab or observation.
5. **rank_with_rationale** — deterministic scoring (pass rate, UNKNOWN penalty, FAIL penalty) + LLM-generated rationale paragraph per top-3 trial (Llama 3.3 70B).

Tool schemas are defined in `tools/__init__.py`; the agent sees them as OpenAI-style function call definitions.

## Three differentiators vs. published literature

1. **Diagnosis Prediction** Research-backed prediction of undiagnosed illnesses and early searching for potential matching to those trials
2. **Proactive missing-data flagging** instead of silent failure or LLM guessing
3. **Multi-indication surfacing** across therapy areas — the agent autonomously decides whether to search cross-indication
