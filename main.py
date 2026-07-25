"""
CIM Text-to-CAD Backend (v3)
============================
Changes vs. v2:

1. SYSTEM_RULES now exposes ALL six cad_primitives helpers (was missing
   make_wheel_mount, safe_union, make_bolt) and explicitly tells the model
   to use safe_union() instead of raw .union(), and make_bolt() instead of
   hand-writing thread/helix geometry.
2. SYSTEM_RULES documents the .cylinder()/.box() centering gotcha so the
   model stops generating disconnected parts that "silently union" into
   broken compounds.
3. call_llm() (Gemini branch) now catches 429 rate-limit errors and returns
   a clear 429 to the frontend instead of an opaque 500.
4. /upload_step now streams the upload with a size cap (MAX_STEP_UPLOAD_BYTES)
   instead of loading unbounded files straight into OCCT's STEP importer,
   which is what was producing the "cannot allocate memory for thread-local
   data: ABORT" crash on large files.

NOTE: this file alone does not fix the "from cad_primitives import *"
ImportError — that fix lives in exec_worker.py:

    ALLOWED_MODULES = {
        "cadquery", "math", "cadquery.selectors", "cadquery.occ_impl", "OCP",
        "cad_primitives",
    }

Make sure exec_worker.py is updated alongside this file.

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

# Max size for user-uploaded STEP files. OCCT's STEP importer can spike
# memory well beyond the file's on-disk size, and large uploads were
# crashing the worker with "cannot allocate memory for thread-local data:
# ABORT" (a container/RAM ceiling being hit, not something catchable in
# Python). Tune this to headroom on your actual hosting plan.
MAX_STEP_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB

app = FastAPI(title="CIM Club Text-to-CAD Core v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your real Vercel domain before going public
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
3. You may also `from cad_primitives import *` to use these pre-tested helpers.
   ALWAYS prefer these over freehand geometry — they are tested and avoid
   common failure modes:
     - make_beam(length, width, height, origin=(0,0,0))
     - make_rail(length, width, height, hole_spacing=None, hole_diameter=6)
     - add_crossmember(base, length, width, height, x_position)
     - bolt_pattern_holes(workplane, diameter, positions)
     - make_wheel_mount(diameter, width, position)  # position=(x,y,z)
     - make_bolt(shank_diameter, length, head_diameter=None, head_height=None, hex_head=True)
       -> use this for ANY screw/bolt/threaded-fastener request instead of
          writing your own Wire.makeHelix() call. Do not hand-write helix
          or thread geometry under any circumstances — it is fragile and
          version-sensitive.
     - safe_union(base, addition)
       -> ALWAYS use safe_union(a, b) instead of a.union(b) when fusing two
          parts. Raw .union() on two solids that don't actually touch or
          overlap silently returns a broken multi-solid compound instead of
          raising an error, which produces disconnected-looking parts.
4. If you must use cq.Workplane(...).cylinder(height, radius) or .box(...)
   directly (only for cases not covered by a primitive above), remember
   these are CENTERED on the workplane by default — they extrude height/2
   in each direction from the plane, not upward from z=0. Pass
   centered=(True, True, False) if you need the base anchored at the
   workplane instead of straddling it. Getting this wrong is a common
   cause of two parts that look "stacked" in a prompt but don't actually
   touch in the generated geometry.
5. The final solid MUST be assigned to a global variable named 'result'.
6. Do not call show_object() or any exporter inside the script.
7. You are editing an ongoing part across a conversation. If previous code is
   given to you below, treat it as the current state of the model and modify
   it according to the newest instruction rather than starting from scratch,
   unless the user clearly asks for something unrelated.
8. If the user uploaded a base STEP file, the first line of the script will
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
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        from google import genai
        from google.genai import types
        from google.genai.errors import ClientError

        client = genai.Client()  # reads GEMINI_API_KEY from env
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=conversation_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
            )
            return response.text.strip()
        except ClientError as e:
            if getattr(e, "code", None) == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Gemini free-tier rate limit hit — wait a minute (RPM cap) "
                        "or until midnight PT (daily cap) and try again."
                    ),
                )
            raise

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
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return False, "Generated script timed out after 20 seconds (likely an infinite loop or huge boolean operation).", "", ""

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

    if not ok:
        # Do NOT overwrite session["last_code"] — keep the last good state
        # so the next prompt still has something valid to build on.
        session["history"].append({"role": "assistant", "content": f"[FAILED] {error}"})
        raise HTTPException(status_code=400, detail=error)

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

    Streams the upload in chunks and rejects anything over
    MAX_STEP_UPLOAD_BYTES before it ever reaches the OCCT importer, since
    large STEP files were crashing the worker with an OOM-style abort.
    """
    session_id, session = get_session(session_id)

    saved_name = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    saved_path = os.path.join(OUTPUT_DIR, saved_name)

    size = 0
    with open(saved_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_STEP_UPLOAD_BYTES:
                out.close()
                os.remove(saved_path)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"STEP file too large ({size // (1024 * 1024)}MB). "
                        f"Importing files this size can exceed available worker RAM. "
                        f"Max allowed is {MAX_STEP_UPLOAD_BYTES // (1024 * 1024)}MB on the current plan."
                    ),
                )
            out.write(chunk)

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

    uvicorn.run(app, host="0.0.0.0", port=8000)