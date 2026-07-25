"""
CIM Text-to-CAD Backend (v2)
============================
Changes vs. your original main.py:

1. STEP export added (SolidWorks/AutoCAD-openable), STL kept only as the
   in-browser preview format.
2. Code execution moved out-of-process into exec_worker.py (subprocess +
   timeout + restricted builtins) instead of exec() in the API process.
3. Session-based chat: each session remembers its last *working* code, so
   "now add a hole" edits the previous part instead of starting over. A bad
   edit never overwrites the last good state.
4. /upload_step: lets a user upload an existing STEP file (e.g. a chassis
   they made in SolidWorks) and keep editing it via prompts.
5. LLM provider is pluggable (Gemini and/or Grok/xAI) behind one function,
   so you can switch or fall back between them without touching the rest
   of the code.

Env vars you need to set before running:
    GEMINI_API_KEY   - if using Gemini
    XAI_API_KEY      - if using Grok (xAI's API is OpenAI-compatible)
    LLM_PROVIDER     - "gemini" or "grok" (default: gemini)
"""

import os
import json
import uuid
import shutil
import subprocess
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="CIM Club Text-to-CAD Core v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://cim-text-to-cad.vercel.app"] once you have your real Vercel URL
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


SYSTEM_RULES = """You are an automated Text-to-CAD compilation engine for a mechanical
engineering CAD generator built on CadQuery.

RULES:
1. Return ONLY pure, executable Python code. No markdown fences, no commentary.
2. Always import cadquery as: import cadquery as cq
3. You may also `from cad_primitives import *` to use these pre-tested helpers
   for anything beyond a single basic shape:
     - make_beam(length, width, height, origin=(0,0,0))
     - make_rail(length, width, height, hole_spacing=None, hole_diameter=6)
     - add_crossmember(base, length, width, height, x_position)
     - bolt_pattern_holes(workplane, diameter, positions)
     - make_wheel_mount(diameter, width, position=(x,y,z))
     - safe_union(base, addition)  -- prefer this over base.union(addition) directly;
       it raises a clear error if the parts don't actually touch/overlap
   Prefer these over freehand low-level geometry for multi-part assemblies
   like chassis, brackets, or anything with repeated structural members.
4. The final solid MUST be assigned to a global variable named 'result'.
5. Do not call show_object() or any exporter inside the script.
6. You are editing an ongoing part across a conversation. If previous code is
   given to you below, treat it as the current state of the model and modify
   it according to the newest instruction rather than starting from scratch,
   unless the user clearly asks for something unrelated.
7. If the user uploaded a base STEP file, the first line of the script will
   already be provided for you as a fixed prefix that imports it into
   `result` — build on top of that variable, don't redefine `result` from
   scratch in that case.
"""


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


def call_llm(system_prompt: str, conversation_text: str) -> str:
    """
    Single entry point for talking to whichever LLM backs this deployment.
    Swap providers via the LLM_PROVIDER env var without touching callers.

    Retries a couple of times on transient server-side errors (e.g. Gemini's
    503 "high demand" response) with a short backoff, since these resolve on
    their own within seconds and shouldn't be treated the same as a genuine
    bug in the generated code.
    """
    import time

    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    LLM_RETRY_ATTEMPTS = 3
    LLM_RETRY_DELAY_SECONDS = 3

    last_exception = None
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            if provider == "gemini":
                from google import genai
                from google.genai import types

                client = genai.Client()  # reads GEMINI_API_KEY from env
                model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                    ),
                )
                return response.text.strip()

            elif provider == "grok":
                # xAI exposes an OpenAI-compatible endpoint, so we reuse the openai client.
                from openai import OpenAI

                client = OpenAI(
                    api_key=os.environ["XAI_API_KEY"],
                    base_url="https://api.x.ai/v1",
                )
                model_name = os.environ.get("GROK_MODEL", "grok-4")  # check xAI docs for current model id
                completion = client.chat.completions.create(
                    model=model_name,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": conversation_text},
                    ],
                )
                return completion.choices[0].message.content.strip()

            else:
                raise HTTPException(status_code=500, detail=f"Unknown LLM_PROVIDER '{provider}'")

        except HTTPException:
            raise  # don't retry on config errors like an unknown provider
        except Exception as e:
            last_exception = e
            error_text = str(e).lower()
            is_transient = any(
                marker in error_text
                for marker in ("503", "unavailable", "overloaded", "high demand", "rate limit", "429")
            )
            if is_transient and attempt < LLM_RETRY_ATTEMPTS - 1:
                time.sleep(LLM_RETRY_DELAY_SECONDS)
                continue
            raise HTTPException(
                status_code=502,
                detail=f"LLM provider ({provider}) request failed: {e}",
            )

    raise HTTPException(status_code=502, detail=f"LLM provider ({provider}) failed after retries: {last_exception}")


def strip_code_fences(text: str) -> str:
    """Defensive cleanup in case the model ignores rule #1 and wraps in ```python fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def run_sandboxed(code: str) -> tuple[bool, str, str, str]:
    """
    Writes `code` to disk and runs it inside exec_worker.py as a separate
    OS process with a hard timeout. Returns (ok, error_message, stl_path, step_path).
    """
    run_id = str(uuid.uuid4())[:8]
    code_path = os.path.join(OUTPUT_DIR, f"script_{run_id}.py")
    stl_path = os.path.join(OUTPUT_DIR, f"part_{run_id}.stl")
    step_path = os.path.join(OUTPUT_DIR, f"part_{run_id}.step")

    with open(code_path, "w") as f:
        f.write(code)

    # Make cad_primitives.py importable from the same working directory
    proc_env = os.environ.copy()
    proc_env["PYTHONPATH"] = APP_DIR + os.pathsep + proc_env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            [sys.executable, os.path.join(APP_DIR, "exec_worker.py"), code_path, stl_path, step_path],
            cwd=APP_DIR,
            env=proc_env,
            capture_output=True,
            text=True,
            timeout=40,
        )
    except subprocess.TimeoutExpired:
        return False, "Generated script timed out after 40 seconds (likely an infinite loop or an extremely heavy boolean operation).", "", ""

    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return False, f"Worker crashed unexpectedly. stderr: {result.stderr[-800:]}", "", ""

    if not payload.get("ok"):
        return False, payload.get("error", "Unknown error"), "", ""

    return True, "", stl_path, step_path


@app.post("/chat")
async def chat(request: ChatRequest):
    session_id, session = get_session(request.session_id)
    session["history"].append({"role": "user", "content": request.message})

    # Build the prompt: system rules are separate; conversation_text carries
    # the running history plus the last known-good code so the model edits
    # in place rather than starting over.
    convo_lines = []
    for turn in session["history"][-10:]:  # cap context length
        convo_lines.append(f"{turn['role'].upper()}: {turn['content']}")

    prefix_note = ""
    if session["base_step_path"]:
        rel = os.path.relpath(session["base_step_path"], APP_DIR)
        prefix_note = (
            f"\nNOTE: The user uploaded a starting STEP file. Begin your script with:\n"
            f'result = cq.importers.importStep(r"{session["base_step_path"]}")\n'
            f"and then apply the requested change on top of that object.\n"
        )

    if session["last_code"]:
        convo_lines.append(f"CURRENT WORKING SCRIPT:\n{session['last_code']}")

    conversation_text = "\n".join(convo_lines) + prefix_note

    raw_code = call_llm(SYSTEM_RULES, conversation_text)
    code = strip_code_fences(raw_code)

    ok, error, stl_path, step_path = run_sandboxed(code)

    # Self-correction loop: complex prompts (assemblies, chassis, etc.) are
    # far more likely to have a bug on the first try than a single primitive.
    # Instead of failing immediately, hand the actual Python error back to
    # the model and ask it to fix its own code. This is the single biggest
    # reliability win for anything beyond basic shapes.
    MAX_FIX_ATTEMPTS = 2
    attempt = 0
    while not ok and attempt < MAX_FIX_ATTEMPTS:
        attempt += 1
        fix_prompt = (
            f"{conversation_text}\n\n"
            f"The script you just wrote failed to execute with this error:\n{error}\n\n"
            f"Here is the exact script that failed:\n{code}\n\n"
            f"Fix the bug and return the complete corrected script. "
            f"Follow all the same rules as before."
        )
        raw_code = call_llm(SYSTEM_RULES, fix_prompt)
        code = strip_code_fences(raw_code)
        ok, error, stl_path, step_path = run_sandboxed(code)

    if not ok:
        # Do NOT overwrite session["last_code"] — keep the last good state
        # so the next prompt still has something valid to build on.
        session["history"].append(
            {"role": "assistant", "content": f"[FAILED after {attempt + 1} attempts] {error}"}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Model tried {attempt + 1} time(s) and couldn't produce working code. Last error: {error}",
        )

    session["last_code"] = code
    session["history"].append({"role": "assistant", "content": "[OK] model updated"})

    return {
        "status": "success",
        "session_id": session_id,
        "code": code,
        "stl_url": f"/static/{os.path.basename(stl_path)}",
        "step_url": f"/static/{os.path.basename(step_path)}",
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
    ok, error, stl_path, step_path = run_sandboxed(preview_code)
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