# Research Experiment Manager

An agentic assistant for running ML research experiments. It keeps an SQLite registry of
experiments, analyses their result files, fills in missing metrics from saved predictions,
compares against a baseline, and proposes what to try next.

**GPT-OSS served by Groq** (`openai/gpt-oss-20b` by default) does the language work:
interpreting requests, explaining evidence, prioritising next steps and answering questions.
Every number, file write and status change is done by deterministic Python. GPT-OSS runs on
Groq's servers in this setup, not locally, and the app stays fully usable when Groq is unavailable.

## Setup

1. **Create or sign in to a Groq account** at <https://console.groq.com>.
2. **Generate an API key** in the Groq Console under _API Keys_.
3. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```
4. **Add your API key to `.env`** (open it in an editor and replace `your_groq_api_key_here`).
   `.env` is listed in `.gitignore`. Never put the key in source code or commit it.
5. **Install the dependencies** (Python 3.10+):
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
6. **Run the application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Click **Load demo experiments** in the sidebar to start with sample data.

The sidebar shows whether Groq is active. Without a key, the app runs in deterministic mode
and tells you how to configure it.

### Configuration

| Variable                | Default              | Purpose                                                 |
| ----------------------- | -------------------- | ------------------------------------------------------- |
| `GROQ_API_KEY`          | —                    | Your Groq key (read only by `llm/groq_client.py`)       |
| `GROQ_MODEL`            | `openai/gpt-oss-20b` | Any chat model your Groq account can access             |
| `LLM_PROVIDER`          | `groq`               | Provider selector                                       |
| `LLM_ENABLED`           | `true`               | `false` forces deterministic mode                       |
| `GROQ_TIMEOUT_SECONDS`  | `30`                 | Per-request timeout                                     |
| `GROQ_MAX_RETRIES`      | `3`                  | Retries for transient failures                          |
| `GROQ_REASONING_EFFORT` | `low`                | GPT-OSS reasoning effort (`low`/`medium`/`high`/`none`) |
| `REM_DATA_DIR`          | `data`               | Where the registry, results, task files and cache live  |

## Three-cycle demo

```bash
python -m demo.run_demo               # deterministic, no API calls
python -m demo.run_demo --llm mock    # full LLM code path with a mocked Groq, no API calls
python -m demo.run_demo --llm groq    # real Groq GPT-OSS (uses free-tier quota)
```

Starting state: a baseline (`EXP-001`), an experiment whose evaluation crashed before writing
F1 (`EXP-002`, predictions saved), and one still running (`EXP-003`).

1. **Cycle 1** recomputes EXP-002's F1 from its predictions, compares to the baseline, and
   proposes a follow-up (lower decision threshold) which is registered with a task file.
2. **Cycle 2** runs that experiment (a deterministic simulator stands in for training),
   analyses it, and proposes another.
3. **Cycle 3** analyses everything, recommends and registers the next planned experiment, and
   the assistant answers "which experiment is best?".

The demo auto-approves gated steps so it can run unattended; the UI asks you instead.

## How the LLM is constrained

- **Allowlisted tools only.** The model proposes actions by name from `agent/tools.py`
  (`validate_metrics`, `calculate_missing_metric`, `compare_to_baseline`,
  `propose_experiment`, `update_status`, …). Nothing else can run: no shell, no `eval`.
- **Structured, validated output.** Plans, decisions and answers are JSON, validated with
  Pydantic. Unknown actions, invalid arguments and experiment IDs not in SQLite are rejected
  step by step. If nothing valid remains, the deterministic planner takes over.
- **Approval is enforced by the registry, not the model.** `propose_experiment` and
  `update_status` always need approval; the model can only make a step stricter.
- **No invented numbers.** The decision engine generates candidates and evidence in Python;
  the model can only rank them. Cited metrics are checked against the registry, and decimal
  numbers in assistant answers must appear in the context that was sent. Anything else is
  discarded in favour of the deterministic answer, with a visible warning.
- **Final verification** after every execution re-derives computed metrics from the
  prediction files, checks value ranges and status validity, and confirms task files exist.

### Free-tier strategy

- Prompts carry compact per-experiment summaries (IDs, params, rounded metrics, flags), only
  for the experiments a request mentions plus the baseline. Raw CSVs and logs are never sent.
- Identical requests are served from a local SQLite cache (`data/llm_cache.sqlite`,
  separate from the registry).
- Rate limits (429) are retried with backoff, honouring `Retry-After`, up to
  `GROQ_MAX_RETRIES`, then surface as a clear message while deterministic results still show.
- Auth errors, unknown models and bad requests are never retried.
- Groq's free-tier limits and model list change over time. If `openai/gpt-oss-20b` is not
  available to your account, set `GROQ_MODEL` to one that is.

## Data formats

- **Result summary** `results/EXP-###_results.csv`: columns `metric,value`. Blank or
  non-numeric values count as missing, not zero.
- **Predictions** `results/EXP-###_predictions.csv`: columns `y_true,y_pred` (binary 0/1).

Required metrics are `accuracy`, `precision`, `recall`, `f1` (primary: `f1`). Upload files from
the _Experiments_ tab or point the registry at them in code.

## Project structure

```text
research-experiment-manager/
├── llm/
│   ├── __init__.py
│   ├── groq_client.py      # Groq SDK, retries/backoff, error mapping, key redaction
│   └── provider.py         # provider-neutral interface, errors, cache, get_provider()
├── agent/
│   ├── planner.py          # request -> validated plan (LLM or deterministic)
│   ├── executor.py         # runs allowlisted tools, approval gate, final verification
│   ├── decision_engine.py  # evidence + candidates (Python), explanation/ranking (LLM)
│   ├── assistant.py        # grounded Q&A for the Streamlit chat
│   ├── prompts.py          # system instructions and prompt templates
│   ├── schemas.py          # Pydantic models for structured output
│   ├── context.py          # compact context builder, cited-number verification
│   └── tools.py            # the allowlisted tool registry
├── core/
│   ├── registry.py         # SQLite registry, status state machine, transactions
│   ├── analysis.py         # CSV parsing, metrics, missing-metric checks, baseline deltas
│   ├── tasks.py            # Markdown task files
│   ├── paths.py            # path validation
│   └── config.py           # settings from env / .env
├── app/streamlit_app.py
├── demo/                   # seed data, run simulator, mock LLM, three-cycle script
├── tests/
├── .env.example
├── .gitignore
└── requirements.txt
```

### Adding another provider

Implement a class with `name`, `model` and
`chat(messages, *, json_mode, temperature, max_tokens) -> LLMResponse`, raising the error types
in `llm/provider.py`, then register it in `get_provider()`. The registry, analysis tools and
agent code don't change.

## run

`python -m streamlit run app/streamlit_app.py`

## Tests

```bash
pytest
```

The suite needs no Groq key and makes no network calls: it removes `GROQ_API_KEY`, skips
`.env`, and makes constructing a real Groq client fail. It covers the setup message for a
missing key, configurable model names, response handling, retry/no-retry behaviour, key
redaction, malformed and malicious model output, registry integrity under API failures,
a mocked Groq response driving the planning workflow, the three-cycle demo in offline and
mock modes, and a headless Streamlit smoke test.
