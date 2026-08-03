# text-to-cad

Text-to-CAD web app: an LLM (Gemini or Groq) writes CadQuery code from chat
prompts, the backend processes it through a 5-step LangGraph workflow, and the
browser shows the 3D result with downloadable STEP/STL files.

## Setup

```
(git bash)
python -m venv .venv
source .venv/Scripts/activate
(also make a .env file with - GEMINI_API_KEY=your_key_here  
                              LLM_PROVIDER=gemini)
pip install -r requirements.txt
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_key_here      # or: export LLM_PROVIDER=groq; export GROQ_API_KEY=...
python main.py
```

The frontend (`index.html`) needs no build step — open it directly, or serve
it from the backend.

## Configuration (all optional except the API key)

| Env var                     | Default            | Purpose                                   |
|-----------------------------|--------------------|-------------------------------------------|
| `LLM_PROVIDER`              | `gemini`           | `gemini` or `groq` — which LLM client to use (can also be set per-request via `state["provider"]`) |
| `GEMINI_API_KEY`            | — (required if provider is gemini) | Google Gemini API key         |
| `GEMINI_MODEL`              | `gemini-2.5-flash` | Model id used by gemini.py                |
| `GEMINI_TEMPERATURE`        | `0.0`              | Sampling temperature                      |
| `GEMINI_RETRY_ATTEMPTS`     | `3`                | Retries on transient API errors           |
| `GROQ_API_KEY`              | — (required if provider is groq) | Groq API key (https://console.groq.com) |
| `GROQ_MODEL`                | `llama-3.3-70b-versatile` | Model id used by groq.py     |
| `GROQ_TEMPERATURE`          | `0.0`              | Sampling temperature                      |
| `GROQ_RETRY_ATTEMPTS`       | `3`                | Retries on transient API errors           |
| `GROQ_RETRY_DELAY_SECONDS`  | `3`                | Backoff between retries                   |
| `GROQ_MAX_RETRY_DELAY_SECONDS` | `60`           | Cap on backoff (Groq's Retry-After honored)|
| `GEMINI_RETRY_DELAY_SECONDS`| `3`                | Backoff between retries                   |
| `MAX_FIX_ATTEMPTS`          | `0`                | Repair-loop cap (0 = one trial per request, no self-correction — set 1–3 to let the model fix its own code at the cost of more API calls) |
| `SANDBOX_TIMEOUT_SECONDS`   | `60`               | Watchdog timeout for generated code
| `HISTORY_WINDOW`            | `10`               | Conversation turns sent to the planner    |
| `MAX_SUBPLANS`              | `5`                | Cap on sub-plans per request              |
| `CORS_ORIGINS`              | `*`                | Comma-separated allowed origins           |
| `PORT`                      | `8000`             | Uvicorn port                              |

## Architecture

- `main.py` — FastAPI API: `/chat` and `/upload_step`, plus in-memory sessions.
- `cad_workflow.py` — the entire step process as a LangGraph state machine:
  1. `understand_request` — classifies intent (new / edit / fastener / frame /
     building / step_edit) with a small intent prompt.
  2. `plan_cad` — converts the request into connected, structured sub-plans
     and picks the right `cad_primitives` helpers.
  3. `generate_code` — one focused code-synthesis call per sub-plan; later
     fragments see earlier fragments so the code stays connected.
  4. `merge_code` + `execute_code` + `repair_code` — merges fragments into one
     script, runs it in the sandbox, and loops errors back to a repair prompt
     (capped by `MAX_FIX_ATTEMPTS`).
  5. `finalize` — returns artifact URLs and the generated code.
- `gemini.py` / `groq.py` — LLM clients; the workflow's `_llm()` dispatches to
  one of them based on `LLM_PROVIDER` (or `state["provider"]`), and each client
  reads its own provider/model/temperature from the workflow state and calls
  its API.
- `cad_primitives.py` — pre-tested building blocks the planner is told to use.
- `index.html` — chat UI + Three.js 3D viewer. The backend URL is auto-detected
  (same origin, `?backend=` param, or `window.APP_CONFIG.BACKEND_URL`).

## Sandboxing

Generated code runs in-process with restricted builtins, an import allow-list,
and a watchdog timeout (`SANDBOX_TIMEOUT_SECONDS`). This is defense-in-depth
for a student project, not a production security boundary — a determined
attacker can escape an in-process restricted `exec()`. For public exposure,
run the worker inside a real container or a hosted code-sandbox service.

## Frontend/backend URL

If `index.html` is served from the same host as the API, it uses that host
automatically. Otherwise open it with `?backend=https://your-backend` or set
`window.APP_CONFIG = { BACKEND_URL: "https://your-backend" }` before the main
script in your deployment.
