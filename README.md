# Token-Budgeted Heterogeneous Multi-Agent AI System

A laptop-friendly multi-agent orchestration system that decomposes a single user prompt into subtasks, routes each subtask to the most efficient model, enforces token budgets, and streams live execution telemetry to a dashboard.

Current default setup:
- Small model: `qwen2:1.5b` via Ollama (local)
- Large model: `llama-3.1-8b-instant` via Groq
- Orchestration: LangGraph
- API/server: FastAPI
- Frontend: plain HTML/CSS/JS + SSE

## What This Project Solves

Most AI apps do one of two inefficient things:
- Send the entire prompt to a large model every time (expensive and wasteful)
- Force everything through a small model (cheap but incomplete on complex tasks)

This project solves that by:
- Breaking one prompt into dependency-aware subtasks
- Classifying each subtask independently
- Routing each subtask to the smallest sufficient model
- Sharing outputs between agents/subtasks
- Enforcing a strict token budget with graceful degradation
- Preserving completeness, especially for code-required subtasks

## Why It Is Unique

This workflow is not only model routing. It is **budget-aware, subtask-aware, and completion-aware**:

1. **Subtask-level heterogeneity**
- One user prompt can use both small and large models in a single run.

2. **Completion guarantees**
- If a subtask asks for code and model output is prose-only, the system retries with strict code instructions.
- If providers fail or budget is depleted, deterministic fallback fills missing subtasks so final output is still complete.

3. **Practical token governance**
- Shared task budget + per-model usage + savings metrics.
- Planner can switch to heuristic mode to save tokens for execution-critical steps.

4. **Transparent orchestration**
- Full SSE telemetry: route decisions, token updates, per-step logs, subtask results, and agent communication.

## End-to-End Workflow

The LangGraph flow is:

`classify_task -> init_budget -> decompose_task -> execute_subtasks -> budget_guard -> (summarizer | finalize)`

### 1) `classify_task`
- Rule-based classifier labels prompt as `simple | moderate | complex`.
- This sets initial task-level budget and expected complexity.

### 2) `init_budget`
- Initializes task token budget:
  - `simple`: 200
  - `moderate`: 800
  - `complex`: 3000 (updated default)
- Initializes per-model usage/cost tracking and savings baselines.

### 3) `decompose_task`
- Splits prompt into actionable subtasks (supports `then`, `and`, punctuation-aware splitting).
- Each subtask is independently classified.
- Dependency chain is retained (`depends_on`) for sequential context flow.

### 4) `execute_subtasks` (core orchestration loop)
For each subtask:
- Build shared context from prior subtask outputs.
- Planner stage:
  - Uses low-cost heuristic planner for budget-preserving cases.
  - Otherwise calls planner model with token-aware prompt.
  - If planner provider fails, fallback to alternate provider or heuristic planner.
- Router stage:
  - `simple` -> small
  - `moderate` -> small first, escalate if shallow/low-confidence
  - `complex` -> large
- Executor stage:
  - Runs tools when needed (calculator/search/formatter).
  - Generates subtask output with budget-aware prompting.
  - Fallback across providers if one fails.
- **Code completeness enforcement**:
  - If subtask requires code and output has no code artifact, run strict retry with mandatory code directive.
  - If still missing, append deterministic code template fallback.
- Persist:
  - subtask result
  - tool call log
  - agent communication message
  - token/cost/savings updates

If budget is exhausted before all subtasks finish:
- Remaining subtasks are completed through deterministic fallback so final output is not missing sections.

### 5) `budget_guard`
- Decides whether summarization is needed near budget limit.
- If run contains code subtasks and executor output already includes code, summarization is avoided to prevent code loss.

### 6) `summarizer` (conditional)
- Compresses output only when needed.
- Preserves multi-subtask structure.
- Has fallback across providers, then safe truncation if both fail.

### 7) `finalize`
- Emits final output + status + full trace metadata.

## Agents

- **Planner Agent**
  - Produces minimal actionable guidance for each subtask.
  - Can be replaced by heuristic planning under token pressure.

- **Executor Agent**
  - Performs subtask execution.
  - Uses tool outputs + shared context + planner instructions.
  - Enforces complete deliverables (including code when required).

- **Summarizer Agent**
  - Compresses output when budget guard requires it.
  - Preserves multi-section structure.

## Built-in Tools (No External APIs)

- `mock_search`: local KB search (`backend/data/local_kb.txt`)
- `calculator`: arithmetic expression evaluation
- `formatter`: JSON/text formatting helpers

## Routing + Fallback Logic

Primary routing:
- `simple`: small model
- `moderate`: small first, escalate if needed
- `complex`: large model

Fallback strategy:
- Planner small failure -> large planner fallback -> heuristic planner fallback
- Executor failure on chosen model -> alternate model -> tool-only deterministic fallback
- Missing code artifact -> strict retry -> deterministic code append fallback
- Incomplete run due budget -> deterministic completion of pending subtasks

## Token, Cost, and Savings Tracking

Tracked live per run:
- Task budget: used/remaining
- Model budgets:
  - small used/cap
  - large used/cap
- Estimated cost:
  - actual orchestrated cost
  - baseline large-only cost
  - USD saved vs large-only
- Large tokens avoided

## API

### `POST /api/run`
Request:
```json
{ "task": "your prompt" }
```
Response:
```json
{ "run_id": "<uuid>" }
```

### `GET /api/stream/{run_id}` (SSE)
Event types:
- `state_update`
- `step`
- `route_decision`
- `token_update`
- `done`
- `error`

### `GET /api/health`
Reports:
- Ollama reachability + available local models
- Groq configuration/reachability + available Groq models
- Missing configured models

## Dashboard

The dashboard displays:
- Input task
- Classification
- Selected model (live)
- Token usage/remaining with progress bar
- Per-model budgets and estimated cost
- Savings per prompt
- Routing decisions timeline
- Execution steps timeline
- Subtask results
- Agent communication log
- Final output

## Project Structure

```text
backend/
  main.py
  graph.py
  config.py
  schemas.py
  classifier.py
  budget.py
  router.py
  data/local_kb.txt
  agents/
    planner_agent.py
    executor_agent.py
    summarizer_agent.py
  tools/
    mock_search.py
    calculator.py
    formatter.py
  models/
    ollama_client.py
  tests/
    test_classifier.py
    test_router.py
    test_budget.py
    test_tools.py
    test_graph_integration.py
frontend/
  index.html
  style.css
  app.js
requirements.txt
```

## Setup

1. Install Ollama: https://ollama.com/download
2. Start Ollama daemon (`http://localhost:11434`)
3. Pull small model:
```bash
ollama pull qwen2:1.5b
```
4. Create a Groq API key: https://console.groq.com/keys
5. Create virtual env and install dependencies:

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From repo root:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- Dashboard: `http://localhost:8000`
- Health: `http://localhost:8000/api/health`

Run via curl:
```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"Explain Dijkstra algorithm with optimized Python code\"}"
```

Stream:
```bash
curl -N http://localhost:8000/api/stream/<run_id>
```

## Environment Variables

Model/provider:
- `SMALL_MODEL` (default: `qwen2:1.5b`)
- `LARGE_MODEL` (default: `llama-3.1-8b-instant`)
- `LARGE_PROVIDER` (default: `groq`)
- `GROQ_API_KEY` (required if `LARGE_PROVIDER=groq`)
- `GROQ_BASE_URL` (default: `https://api.groq.com/openai/v1`)
- `GROQ_TIMEOUT_SECONDS` (default: `90`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_TIMEOUT_SECONDS` (default: `90`)

Budget behavior:
- `BUDGET_SIMPLE` (default: `200`)
- `BUDGET_MODERATE` (default: `800`)
- `BUDGET_COMPLEX` (default: `3000`)
- `SUMMARIZE_THRESHOLD` (default: `80`)
- `ESCALATION_MIN_TOKENS` (default: `280`)

Model budget caps for dashboard:
- `SMALL_MODEL_TOKEN_CAP` (default: `4000`)
- `LARGE_MODEL_TOKEN_CAP` (default: `2500`)

Cost tracking (USD per 1M tokens):
- `SMALL_MODEL_INPUT_USD_PER_1M` (default: `0.10`)
- `SMALL_MODEL_OUTPUT_USD_PER_1M` (default: `0.10`)
- `LARGE_MODEL_INPUT_USD_PER_1M` (default: `0.05`)
- `LARGE_MODEL_OUTPUT_USD_PER_1M` (default: `0.08`)

## Tests

```bash
pytest -q
```

Coverage includes:
- classifier behavior
- budget accounting
- router decisions
- tools
- graph integration paths:
  - escalation
  - provider fallbacks
  - multi-model subtask routing
  - complete subtask output coverage for multi-part prompts
