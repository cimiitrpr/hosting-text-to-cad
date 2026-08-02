"""
CIM Text-to-CAD Backend (v4)
============================
Changes vs. v3:

1. The chat pipeline is now a 5-step LangGraph workflow in cad_workflow.py
   (understand_request -> plan_cad -> generate_code -> merge+execute+repair
   -> finalize), each step with its own small prompt instead of one giant
   system prompt. This file only handles HTTP + session storage.
2. Grok/xAI support removed — Gemini is the only LLM provider, driven by the
   workflow state via gemini.py.
3. exec_worker.py removed: sandbox execution is in-process (restricted
   builtins + import allow-list + watchdog timeout) inside cad_workflow.py.
4. Nothing is hardcoded: model names, retries, timeouts, ports, CORS origins
   and the frontend's backend URL are all configurable via environment
   variables (defaults listed in README.md).

Env vars:
     GEMINI_API_KEY                  - required if LLM_PROVIDER=gemini
    GROQ_API_KEY                    - required if LLM_PROVIDER=groq
    LLM_PROVIDER                    - "gemini" (default) or "groq"
    GEMINI_MODEL                    - default: gemini-2.5-flash
    GEMINI_TEMPERATURE              - default: 0.0
    GEMINI_RETRY_ATTEMPTS           - default: 3
    GEMINI_RETRY_DELAY_SECONDS      - default: 3
    GROQ_MODEL                      - default: llama-3.3-70b-versatile
    GROQ_TEMPERATURE                - default: 0.0
    GROQ_RETRY_ATTEMPTS             - default: 3
    GROQ_RETRY_DELAY_SECONDS        - default: 3
    MAX_FIX_ATTEMPTS                - default: 0 (single trial per request; set higher to let the model fix its own code, at the cost of more API calls)
    SANDBOX_TIMEOUT_SECONDS         - default: 60
    HISTORY_WINDOW              - default: 10
    MAX_SUBPLANS                - default: 5
    CORS_ORIGINS                - comma-separated list, default: *
    PORT                        - default: 8000
"""

import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cad_workflow import OUTPUT_DIR, WORKFLOW, run_in_sandbox

import gemini

APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="CIM Club Text-to-CAD Core v4")

# CORS: allow the origins in CORS_ORIGINS (comma-separated); "*" by default.
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# ---------------------------------------------------------------------------
# In-memory session store.
#   session_id -> {
#       "history": [ {"role": "user"/"assistant", "content": str}, ... ],
#       "last_code": str | None,       # last script that executed successfully
#       "base_step_path": str | None,  # set if user uploaded a starting STEP file
#   }
# This resets when the server restarts. Fine for a demo; swap for a
# SQLite table (session_id, json blob) if you want it to survive restarts.
# ---------------------------------------------------------------------------
SESSIONS: dict[str, dict] = {}


def get_session(session_id: Optional[str]) -> tuple[str, dict]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    new_id = str(uuid.uuid4())
    SESSIONS[new_id] = {"history": [], "last_code": None, "base_step_path": None}
    return new_id, SESSIONS[new_id]


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    session_id, session = get_session(request.session_id)
    session["history"].append({"role": "user", "content": request.message})

    # Seed the workflow state with session context; the LangGraph owns every
    # step (understand -> plan -> generate -> merge -> execute -> repair -> finalize).
    try:
        result = WORKFLOW.invoke(
            {
                "session_id": session_id,
                "user_message": request.message,
                "history": session["history"],
                "last_good_code": session["last_code"],
                "base_step_path": session["base_step_path"],
            }
        )
    except (gemini.QuotaExceededError, groq.QuotaExceededError) as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM pipeline failed: {e}")

    response = result.get("response", {})

    if response.get("status") != "success":
        # Do NOT overwrite session["last_code"] — keep the last good state
        # so the next prompt still has something valid to build on.
        session["history"].append(
            {"role": "assistant", "content": f"[FAILED] {response.get('error', 'Unknown error')}"}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Model tried {response.get('attempts', 1)} time(s) and couldn't produce working code. Last error: {response.get('error')}",
        )

    session["last_code"] = response["code"]
    session["history"].append({"role": "assistant", "content": "[OK] model updated"})

    return {
        "status": "success",
        "session_id": session_id,
        "code": response["code"],
        "stl_url": response["stl_url"],
        "step_url": response["step_url"],
    }


@app.post("/upload_step")
async def upload_step(session_id: Optional[str] = Form(None), file: UploadFile = File(...)):
    """
    Lets a user upload a STEP file (e.g. a chassis they built in SolidWorks)
    to use as the starting point for further prompt-driven edits.
    """
    session_id, session = get_session(session_id)

    saved_name = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    saved_path = os.path.join(OUTPUT_DIR, saved_name)
    with open(saved_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    session["base_step_path"] = saved_path
    session["last_code"] = None  # force next /chat call to rebuild from this base

    # Generate an STL preview of the uploaded file so the viewer has something to show
    preview_code = f'import cadquery as cq\nresult = cq.importers.importStep(r"{saved_path}")\n'
    ok, error, stl_path, step_path = run_in_sandbox(preview_code)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded STEP file: {error}")

    return {
        "status": "success",
        "session_id": session_id,
        "stl_url": f"/static/{os.path.basename(stl_path)}",
        "step_url": f"/static/{os.path.basename(step_path)}",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # Render/Railway inject PORT; falls back to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
