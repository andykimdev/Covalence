# bio-agent-hackathon

LLM agent that matches patients to clinical trials. The agent searches a local trial corpus, parses eligibility criteria, checks criterion-by-criterion verdicts (PASS / FAIL / UNKNOWN), validates results, and returns a ranked top-3 shortlist with rationales.

## Prerequisites

- Python 3.10+ (3.12 recommended)
- A [Nebius Token Factory](https://nebius.com) API key (OpenAI-compatible endpoint)

## Setup

```bash
git clone <repo-url>
cd bio-agent-hckthon

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root (already gitignored):

```env
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
NEBIUS_API_KEY=your_api_key_here

# optional — defaults to meta-llama/Llama-3.3-70B-Instruct
MODEL_AGENT=meta-llama/Llama-3.3-70B-Instruct
```

You'll also need `fixtures/trials_v1.json` and the remaining tool modules under `tools/` — see below for the expected layout.

## Run the agent

From the project root with your venv activated:

```python
from agent.loop import run_agent

patient = {
    "patient_id": "P001",
    "diagnosis": "HER2+ metastatic breast cancer",
    "biomarkers": {"HER2": "positive", "ER": "negative", "PR": "negative"},
    "stage": "IV",
    "prior_treatments": ["trastuzumab", "pertuzumab"],
    "ecog": None,
}

def trace(event):
    print(event["type"], event.get("name") or event.get("content", "")[:80])

result = run_agent(patient, trace_callback=trace)
print(result)
```

Or as a one-liner:

```bash
python -c "
from agent.loop import run_agent
print(run_agent({'patient_id': 'P001', 'diagnosis': 'HER2+ metastatic breast cancer'}))
"
```

The agent loop (`agent/loop.py`) calls Nebius chat completions with tool calling enabled, runs up to 12 iterations, and returns JSON ranked matches when the model finishes.

## Test trial search alone

BM25 search over the local corpus does not call the LLM:

```python
from tools.trial_search import trial_search

results = trial_search("HER2 positive metastatic breast cancer trastuzumab", top_n=5)
print(results)
```

## Trial fixtures

`tools/trial_search.py` reads `fixtures/trials_{snapshot}.json` (default snapshot: `v1`). Each trial should be a JSON object with at least:

```json
[
  {
    "nct_id": "NCT01234567",
    "title": "Phase 2 study of ...",
    "indication": "Breast cancer",
    "inclusion_criteria": ["Age >= 18", "HER2 positive"],
    "exclusion_criteria": ["Prior anthracycline"],
    "indication_tags": ["breast", "HER2", "metastatic"]
  }
]
```

To load a different snapshot at startup:

```python
from tools.trial_search import load_trials

load_trials("v2")  # loads fixtures/trials_v2.json
```

## Project structure

```
agent/
  loop.py       # Agent loop (LLM + tool calls)
  prompts.py    # System prompts for agent and tools
tools/
  __init__.py        # Tool schemas + execute_tool dispatcher
  trial_search.py
  parse_criteria.py
  check_eligibility.py
  validate_verdicts.py
  rank_rationale.py
fixtures/
  trials_v1.json
```

## Agent pipeline

1. **trial_search** — BM25 retrieval over local trials (no LLM)
2. **parse_criteria** — structured inclusion/exclusion per trial
3. **check_eligibility** — per-criterion PASS / FAIL / UNKNOWN
4. **validate_verdicts** — catch PASS when patient data is missing
5. **rank_with_rationale** — top 3 with scoring and clinician-facing rationale

Tool definitions live in `tools/__init__.py`; the agent sees them as OpenAI-style function schemas.
