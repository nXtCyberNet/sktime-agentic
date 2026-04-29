# sktime-agentic

> **Agent-driven time-series forecasting system.**

[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/nXtCyberNet/sktime-forge/pulls)

> ⚠️ **Experimental — not production-ready.** APIs may change, several known issues exist (documented below), and not all features are stable. The end-to-end demo on the airline dataset is verified and working.

A companion project to [`sktime`](https://github.com/sktime/sktime) and [`sktime-mcp`](https://github.com/sktime/sktime-mcp).

---

## What This Is

Most AutoML systems make model selection a deterministic algorithm. This project makes it a reasoning problem — Model selection is initiated by an LLM agent, while final outcomes are constrained by system-level validation, availability, and performance evaluation.

The dataset profile is constructed using MCP tools such as detect_seasonality, run_stationarity_test, and check_structural_break. The agent receives this data profile (along with past failures) and ranks candidate models based on dataset diagnostics and prior outcomes. In addition to LLM-ranked candidates, the system includes baseline estimators (e.g., AutoETS) to ensure fallback coverage. It then fits all candidates, evaluates them on a held-out validation split, and promotes the winner to a model registry. If the promoted model later degrades in production, a watchdog queues a retrain automatically.

The goal of this project was to implement that full ReAct loop end-to-end and validate it on real data. The airline dataset demo below is captured from a full end-to-end run.

---

## Decision Hierarchy

The system separates decision-making into two layers:

1. **LLM Layer**
   - proposes and ranks candidate models based on dataset diagnostics and history

2. **System Layer**
   - enforces constraints (dependencies, availability)
   - evaluates candidates on validation data
   - selects the final model based on performance

This ensures that LLM reasoning is advisory, while final outcomes remain grounded in empirical evaluation. 

---

## How It Works

```
Production event (drift / cold start / human request)
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                  LLM Agent Loop (ReAct)                  │
│                                                          │
│       observe → reason → call tool → observe → ...      │
│                                                          │
│  Has access to: stored production history for the        │
│  dataset, past model failures, drift patterns,           │
│  dataset characteristics derived from profiling          │
│  (seasonality, stationarity, structural breaks)          │
└──────────────────────────┬───────────────────────────────┘
                           │  MCP tool calls
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  sktime-mcp Tool Layer                   │
│                                                          │
│  check_structural_break     get_dataset_history          │
│  detect_seasonality         get_model_complexity_budget  │
│  estimate_training_cost     run_stationarity_test        │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           Production Infrastructure             │
│                                                 │
│  sktime pipelines   MLflow model registry       │
│  Valkey             FastAPI serving layer       │
│  Go API gateway     S3 / GCS model storage      │
└─────────────────────────────────────────────────┘
```

---

## Failure Handling & Resilience

This is a core design concern. The system is built to always produce a forecast — it should never hard-fail because a single model or dependency is unavailable.

### Dependency Fallback Chain

The candidate list is ordered by preference. If a model's optional dependency is missing at training time, it is skipped and the next candidate is tried. In a fresh environment, the following packages are **not** in `requirements.txt` and will be skipped if not separately installed:

| Package | Required by | Install |
|---|---|---|
| `prophet` | Prophet | `pip install prophet` |
| `tbats` | TBATS, BATS | `pip install tbats` |
| `pmdarima` | AutoARIMA | `pip install pmdarima` |

`NaiveForecaster` has no optional dependencies and is always the guaranteed last-resort fallback. If every other candidate fails, `NaiveForecaster` will be trained and promoted. The system will not crash due to a missing package.

### Fit Failure Fallback

If a model is installed but throws during `.fit()`, the error is caught, logged at `ERROR` level, and the training run skips to the next candidate. The failed model is recorded in Valkey so the agent can factor it into future ranking decisions.

### LLM Failure

If the LLM call fails or returns malformed output, `ModelSelectorAgent` falls back to a hardcoded default candidate order rather than halting the pipeline.

### Watchdog & Retrain

After every promotion, a `Watchdog` monitors live MAE against the training baseline. If degradation exceeds the configured threshold, it queues a retrain job. If the retrain also produces a worse model than the current production model, the current version is kept — the system does not regress.

---

## Verified End-to-End: Airline Dataset

> Full unedited log: [`docs/full_end_log.md`](docs/full_end_log.md)

The following is a condensed trace of the system running the `airline` dataset from a cold start — no model in the registry, no cached state.

### Step 1 — Cold Start Detected

```
WARNING: failed MLflow version fallback for airline:
  RESOURCE_DOES_NOT_EXIST: Registered Model with name=ts-forecaster-airline not found
INFO: cold start flow for dataset_id=airline
```

### Step 2 — Data Profiled, LLM Reasons Over It

The `PipelineArchitectAgent` profiles the dataset. The `ModelSelectorAgent` sends that profile to the LLM, which generates a reasoning trace based on the dataset profile:

> *"Dataset: non-stationary, strong seasonality, structural break detected.
> Prefer models that handle changepoints natively (Prophet) or are robust to level shifts
> (ExponentialSmoothing) over ARIMA-family models. Always include NaiveForecaster
> as a last-resort fallback."*

```
LLM ranked output: ["Prophet", "ExponentialSmoothing", "TBATS", "NaiveForecaster"]
```

### Step 3 — Dependency Fallbacks Fire

Prophet, TBATS, BATS, and AutoARIMA are skipped — none of their optional packages are installed. The system logs each skip at `ERROR` and continues:

```
ERROR: cannot instantiate Prophet   → pip install prophet
ERROR: cannot instantiate TBATS     → pip install tbats
ERROR: cannot instantiate BATS      → pip install tbats
ERROR: cannot instantiate AutoARIMA → pip install pmdarima
```

The remaining candidates are fitted normally.

### Step 4 — All Available Candidates Evaluated

| Model | val_mae | val_rmse | fit_seconds |
|---|---|---|---|
| **AutoETS ✅** | **25.1520** | **29.7411** | 2.1 |
| PolynomialTrendForecaster | 34.5551 | 48.1882 | 0.0 |
| ExponentialSmoothing | 43.4961 | 51.9158 | 0.2 |
| NaiveForecaster | 81.4483 | 93.1339 | 0.0 |
| ThetaForecaster | 91.3702 | 102.4195 | 0.0 |

### Step 5 — Winner Promoted, Forecast Served

```
INFO: best model for airline is AutoETS (val_mae=25.1520)
INFO: promoted model version 1 for airline
INFO: served 6-step forecast for airline v1 in 25.8 ms (cache_hit=False)
```

```json
{
  "dataset_id": "airline",
  "predictions": [483.756, 429.041, 373.516, 326.012, 370.046, 375.170],
  "prediction_intervals": {
    "lower": [457.318, 399.325, 344.051, 293.409, 330.630, 333.390],
    "upper": [510.689, 458.569, 405.015, 357.596, 411.413, 417.687]
  },
  "model_version": "1",
  "model_class": "TransformedTargetForecaster",
  "model_status": "ok",
  "cache_hit": false
}
```

---

## Architecture

### Project Layout

```
sktime-agentic/
├── python/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── chat_router.py        # NL query → structured ForecastRequest
│   │   │   ├── model_selector.py     # ReAct loop: LLM calls MCP tools to rank models
│   │   │   ├── pipeline_architect.py # Profiles data, writes DataProfile to Valkey
│   │   │   ├── prediction.py         # Loads model from MLflow, runs inference
│   │   │   ├── training.py           # Fits + evaluates all candidates, promotes winner
│   │   │   └── watchdog.py           # Post-promotion MAE monitoring, queues retrain
│   │   ├── mcp/
│   │   │   ├── client.py             # MCPClient: dispatches tool calls to implementations
│   │   │   ├── check_structural_break.py
│   │   │   ├── detect_seasonality.py
│   │   │   ├── estimate_training_cost.py
│   │   │   ├── get_dataset_history.py
│   │   │   ├── get_model_complexity_budget.py
│   │   │   └── run_stationarity_test.py
│   │   ├── memory/
│   │   │   └── memory.py             # Per-dataset history in Valkey
│   │   ├── monitoring/
│   │   │   └── drift_monitor.py      # CUSUM + ADWIN detection, publishes signal only
│   │   ├── registry/
│   │   │   ├── registry.py           # CANDIDATE_ESTIMATORS, profile-based filtering
│   │   │   └── data_registry.py      # Dataset record store in Valkey
│   │   ├── data/
│   │   │   └── local_loader.py       # CSV loader for local fixture datasets
│   │   ├── config.py                 # Pydantic Settings — reads from .env
│   │   ├── contracts.py              # Protocol interfaces (AgentMemory, Watchdog)
│   │   ├── main.py                   # FastAPI app: /forecast, /chat, /admin/*, /metrics
│   │   ├── orchestrator.py           # Coordinates cold-start and retrain flows
│   │   ├── prompts/prompts.py        # System prompts for each LLM agent
│   │   ├── retrain_worker.py         # Valkey stream consumer for retrain:jobs
│   │   └── schemas.py                # Pydantic request/response models
│   ├── scripts/
│   │   ├── run_demo.py               # Local end-to-end demo runner
│   │   ├── cold_start_aeroplane.py   # Seed + cold-start for aeroplane dataset
│   │   ├── ingest_data.py            # Push a CSV dataset into Valkey
│   │   └── start_local_mlflow.py     # Start MLflow tracking server locally
│   ├── tests/
│   │   ├── fixtures/sample_datasets/ # Curated CSVs for testing
│   │   └── unit/                     # Agent unit tests
│   └── requirements.txt
│
├── go/                               # Go API gateway (request routing, Valkey bridge)
├── k8s/                              # Kubernetes manifests for all services
├── docs/                             # Architecture docs, problem log, full run logs
├── data_cache/                       # Airline CSV (built-in fallback dataset)
├── docker-compose.yml                # Valkey + MLflow + Python/Go workers
└── .env.example                      # All required environment variables
```

### What Each Layer Does

**`ModelSelectorAgent` (`agents/model_selector.py`)**
The core intelligence. Runs a ReAct loop: calls `PipelineArchitectAgent` to build a data profile, then calls the LLM with that profile and the list of available estimators. The LLM reasons through seasonality, stationarity, structural breaks, and past failures to return a ranked candidate list. Tool calls are dispatched via `MCPClient`. The ranked list is written to Valkey for `TrainingAgent` to consume.

**`TrainingAgent` (`agents/training.py`)**
Reads the ranked candidate list, fits each estimator as a `TransformedTargetForecaster` sktime pipeline in a thread executor, evaluates on a validation split, logs every run to MLflow, picks the lowest `val_mae` winner, and registers it in the MLflow model registry.

**`PredictionAgent` (`agents/prediction.py`)**
Resolves the active model version from Valkey (falls back to MLflow registry), loads the model with in-process caching, and runs inference off the event loop in an executor. Returns point forecasts, prediction intervals, and an LLM-generated rationale string.

**`Watchdog` (`agents/watchdog.py`)**
Spawned after every model promotion. Polls residuals from Valkey, computes live MAE, compares against the baseline MAE from training, and queues a retrain job if degradation exceeds the threshold.

**`Orchestrator` (`orchestrator.py`)**
Top-level coordinator. Detects cold-start vs warm-start, chains the full agent pipeline, and manages Valkey stream workers.

**`DriftMonitor` (`monitoring/drift_monitor.py`)**
CUSUM + ADWIN statistical detection. Publishes a signal only — it makes no decisions. The agent decides what to do in response.

**Go layer (`go/`)**
API gateway for routing external forecast requests. Stateless — all state lives in Valkey and MLflow.

---

## MCP Tool Reference

All tools are implemented under `python/app/mcp/` and dispatched by `MCPClient`.

| Tool | Description | Returns |
|---|---|---|
| `check_structural_break` | CUSUM-based break detection | break_detected, location, confidence |
| `detect_seasonality` | Seasonal period and strength detection | period, strength, method |
| `estimate_training_cost` | Cost estimate before fitting | estimated_seconds, complexity |
| `get_dataset_history` | Past models, scores, and failures for a dataset | history list |
| `get_model_complexity_budget` | Budget constraints for model selection | max_params, time_budget |
| `run_stationarity_test` | ADF + KPSS stationarity tests | p-values, conclusion |

---

## Prerequisites

Before running, make sure you have:

- Python 3.10+
- Docker + Docker Compose (for Valkey and MLflow)
- Go 1.21+ (only if running the Go gateway)
- An LLM API key — the system uses an OpenAI-compatible API format. The demo uses [Hack Club AI](https://ai.hackclub.com) as a free proxy, but any OpenAI-compatible endpoint works (OpenAI, Together, Groq, local Ollama, etc.)

---

## Quickstart

```bash
# Clone
git clone https://github.com/nXtCyberNet/sktime-forge
cd sktime-forge

# Set up Python environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r python/requirements.txt

# Configure
cp .env.example .env
# Edit .env — see Environment Variables section below

# Option A: run with Docker (recommended)
docker compose up -d valkey mlflow
python python/scripts/run_demo.py --dataset_id airline --valkey_url valkey://localhost:6379

# Option B: run MLflow locally without Docker
python python/scripts/start_local_mlflow.py   # separate terminal
python python/scripts/run_demo.py --dataset_id airline --valkey_url redis://localhost:6379

# Run against a local CSV fixture
python python/scripts/run_demo.py \
  --dataset_id yahoo_s5_like_drift.csv \
  --local_dataset_dir python/tests/fixtures/sample_datasets

# Start the FastAPI server
uvicorn python.app.main:app --reload
# POST /forecast   {"dataset_id": "airline", "fh": [1,2,3,4,5,6]}
# POST /chat       {"query": "forecast airline next 6 months"}
```

### Optional dependencies (for full candidate list)

```bash
pip install prophet      # enables Prophet
pip install tbats        # enables TBATS and BATS
pip install pmdarima     # enables AutoARIMA
```

Without these, the system falls back to AutoETS, ExponentialSmoothing, ThetaForecaster, PolynomialTrendForecaster, and NaiveForecaster — which is sufficient for the demo.

---

## Environment Variables

All variables are read from `.env`. See `.env.example` for a template.

| Variable | Required | Description | Example |
|---|---|---|---|
| `LLM_API_KEY` | Yes | API key for your LLM provider | `sk-...` |
| `LLM_BASE_URL` | Yes | OpenAI-compatible base URL | `https://ai.hackclub.com/proxy/v1` |
| `LLM_MODEL` | Yes | Model name to use | `gpt-4o-mini` |
| `MLFLOW_TRACKING_URI` | Yes | MLflow tracking server URI | `http://localhost:5000` |
| `VALKEY_URL` | Yes | Valkey connection URL | `valkey://localhost:6379` |
| `RETRAIN_MAE_THRESHOLD` | No | Degradation ratio to trigger retrain | `0.15` (default) |
| `PROFILE_TTL_SECONDS` | No | Dataset profile cache TTL in Valkey | `3600` (default) |
| `WATCHDOG_TTL_SECONDS` | No | Watchdog monitoring window per version | `3600` (default) |

---

## Known Issues

These are real bugs and limitations observed in the current implementation.

### 1. Async event loop conflict in `TrainingAgent`

**Symptom:**
```
WARNING: failed to load profile for airline:
  Task got Future attached to a different loop
WARNING: failed to load profile for airline: Event loop is closed
```

**What happens:** `TrainingAgent` runs model fitting in a thread executor (sync context). When it attempts to re-fetch the dataset profile via async Valkey reads inside that thread, it conflicts with the main event loop. Profile loading silently fails for those candidates — training continues but without profile data.

**Impact:** Models fitted mid-run operate without the full data profile context. Results are still valid but the agent has less information than intended.

**Fix (planned):** Pass the pre-fetched `DataProfile` object as a direct argument to the training run rather than re-fetching it async inside the executor thread.

---

### 2. LLM reasons without knowing available dependencies

**Symptom:** The LLM recommends Prophet and AutoARIMA, but both are unavailable and silently skipped.

**What happens:** The LLM receives a list of *all registered candidate estimators*, not a list of *actually installable* ones. It can recommend models that will fail immediately at instantiation.

**Impact:** The LLM's ranked reasoning is partially blind. Its top picks are often the first to be skipped.

**Fix (planned):** Pre-filter the candidate list against installed packages before passing it to the LLM prompt, so the LLM only reasons about models it can actually use.

---

### 3. MLflow artifact path fallback

**Symptom:**
```
Run has no artifacts at artifact path 'model',
registering model based on models:/m-cd05f96f29a747ac82060574a5d21c51 instead
```

**What happens:** The model artifact isn't saved to the expected path. MLflow silently falls back to an internal URI. Promotion succeeds, but loading a specific run's artifact directly by path is unreliable.

**Impact:** Low for normal usage. Higher if you try to inspect or replay a specific run artifact.

---

### 4. MLflow API deprecations

The following deprecated APIs are in use and will need updating for MLflow 3.x:

- `MlflowClient.get_latest_versions` — deprecated since MLflow 2.9.0
- `artifact_path` parameter in model logging — deprecated, use `name`
- `valkey.close()` — deprecated since valkey-py 5.0.1, use `aclose()`

None are blocking in the current version.

---

## Roadmap

- [ ] Fix async profile re-fetch in `TrainingAgent`
- [ ] Pass available-only estimators to LLM prompt
- [ ] Resolve MLflow artifact path inconsistency
- [ ] Replace deprecated MLflow APIs
- [ ] Multi-dataset concurrent scheduling
- [ ] Full Go gateway implementation (currently a stub)
- [ ] `skops` serialization replacing pickle for MLflow model storage

---

## Contributing

Issues and PRs are welcome. If you hit a bug not listed above, please open an issue with the full log output — the structured logging makes it easy to diagnose.

For larger changes, open an issue first to discuss the approach.

---

## License

BSD 3-Clause — same as sktime.