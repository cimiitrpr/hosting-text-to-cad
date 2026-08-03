"""
cad_workflow.py
---------------
The whole text-to-CAD step process as a LangGraph state machine, with small
single-responsibility prompts instead of one giant system prompt.

Workflow:

    START
      -> understand_request   (step 1: classify intent; new/edit/fastener/frame/building/step_edit)
      -> plan_cad             (step 2: produce connected sub-plans + chosen primitives)
      -> generate_code        (step 3: one focused code-synthesis call per sub-plan)
      -> merge_code           (step 4a: join fragments into one script with import header)
      -> execute_code         (step 4b: run it in the sandbox)
      -> finalize             (step 5: return artifact URLs + code)
             ^
             | (failure with fixes left)
             +-- repair_code -> merge_code -> execute_code

    execute_code routes: ok -> finalize; fail + fixes left -> repair_code;
    fail + fixes exhausted -> finalize (failure response).

exec_worker.py was removed: sandbox execution now happens in-process with the
same restricted builtins + import allow-list, guarded by a watchdog thread.

Config (all optional, sensible defaults):
    LLM_PROVIDER             - 'gemini' (default) or 'groq'; can also be set
                               per-request via state["provider"]
    MAX_FIX_ATTEMPTS         - repair-loop cap, default: 2 (original attempt
                               plus up to 2 self-repair attempts)
    SANDBOX_TIMEOUT_SECONDS  - watchdog timeout for generated code, default: 120
    HISTORY_WINDOW           - conversation turns sent to the planner, default: 10
    MAX_SUBPLANS             - cap on sub-plans per request, default: 5
"""

import builtins as _builtins_module
import json
import os
import re
import threading
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import gemini

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FIX_ATTEMPTS = int(os.environ.get("MAX_FIX_ATTEMPTS", "2"))
SANDBOX_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "120"))
HISTORY_WINDOW = int(os.environ.get("HISTORY_WINDOW", "10"))
MAX_SUBPLANS = int(os.environ.get("MAX_SUBPLANS", "5"))


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class CadState(TypedDict, total=False):
    # --- session context (seeded by main.py) ---
    session_id: str
    user_message: str
    history: list
    last_good_code: str
    base_step_path: str

    # --- step 1: request understanding ---
    task_type: str
    intent: str

    # --- step 2: CAD planning ---
    plan: list

    # --- step 3: code generation ---
    code_fragments: list

    # --- step 4: execution and repair ---
    merged_code: str
    ok: bool
    execution_error: str
    stl_path: str
    step_path: str
    fix_attempts: int
    executed: dict  # code -> last result, so identical scripts never re-run

    # --- step 5: final response ---
    response: dict


# ---------------------------------------------------------------------------
# Split prompts (replaces the old single giant system prompt)
# ---------------------------------------------------------------------------
INTENT_PROMPT = """You classify the intent of a Text-to-CAD request for a mechanical CAD generator built on CadQuery.

Reply with ONLY a JSON object. No markdown fences, no commentary, no extra keys:
{"task_type": "new" | "edit" | "fastener" | "frame" | "building" | "step_edit", "brief": "one sentence restatement of the geometry the user wants"}

Rules:
- "new": building something from scratch, no previous part exists.
- "edit": modifying the current working part (a previous working script exists).
- "fastener": any screw/bolt/threaded fastener, metric sizes (M2..M8) included.
- "frame": structural chassis/frame/rail/bracket assemblies.
- "building": house/room/wall/roof structures.
- "step_edit": a base STEP file is loaded and the request edits that imported part.
- If the previous working script or a base STEP file exists, prefer "edit"/"step_edit" over "new".
"""

PLAN_PROMPT = """You are a mechanical CAD planner for CadQuery. Convert the request into a short, structured plan of CONNECTED sub-plans.

Each sub-plan is one step of geometry that builds on earlier sub-plans by name — later sub-plans reference variables created by earlier ones, so the final code is one coherent part, never isolated snippets.

Primitive catalog (all from cad_primitives, already importable as: from cad_primitives import *):
- make_beam(length, width, height=None, origin=(0,0,0), thickness=None) — rectangular structural beam; `thickness` is an accepted alias for `height` (e.g. a 120mm x 120mm x 8mm plate -> make_beam(120, 120, thickness=8))
- make_rail(length, width, height, hole_spacing=None, hole_diameter=6) — long rail with optional mounting holes
- add_crossmember(base, length, width, height, x_position) — fuses a beam onto base
- bolt_pattern_holes(workplane_or_solid, diameter, positions=None, pattern=None, count=1, spacing=0, axis="x") — drill holes through a part's top face; give explicit `positions=[(x, y), ...]` OR a linear pattern via count/spacing/axis; pass a solid and the top face is used automatically; returns the drilled part
- make_wheel_mount(diameter, width, position=(x,y,z)) — cylindrical mount along Y
- make_cylinder(diameter, height, position=(0,0,0)) — vertical solid cylinder, centered on position (bottom at position z - height/2); use for ANY cylinder/disc/shaft/spacer/pin section (e.g. 90mm dia x 20mm tall -> make_cylinder(90, 20))
 - safe_union(*parts) — fuses ANY number of parts into one (safe_union(a, b) or safe_union(a, b, c, ...)); raises a clear error if consecutive parts don't touch
- make_pin_grid(base=None, pins_x=None, pins_y=None, pin_size=None, pin_height=None, pitch=None, ...) — fuses a centered grid of square cooling pins onto the TOP face of an existing base; use for ANY heatsink / pin-array / radiator request. ALSO accepts natural-language aliases: rows, cols, grid_size=(r, c), spacing (for pitch), pin_diameter/pin_width (for pin_size), base_object (for base). If pitch is omitted it is auto-derived from the plate size (plate length / pins per side). Example: 120mm plate, 12x12 pins, 4mm pins, 50mm tall -> make_pin_grid(base_plate, pins_x=12, pins_y=12, pin_size=4, pin_height=50)
- make_bolt(shank_diameter, length, head_diameter=None, head_height=None, hex_head=True) — use for ANY screw/bolt request; metric size M3 -> shank_diameter=3
- make_l_bracket(leg1_length, leg2_length, width, thickness, hole_diameter=None, hole_inset=None) — use for ANY angle/L bracket
- make_wall(length, height, thickness, origin=(0,0,0), axis="x"|"y") — wall from a STARTING CORNER
- cut_opening(wall, width, height, position, origin=(0,0,0), axis="x"|"y") — cut doors/windows BEFORE unioning that wall in
- make_box_room(length, width, height, thickness) — four cleanly-meeting walls
- cut_dovetail(block, base_width, top_width, height, length=None, position=None, axis="x") — cut an inverted-trapezoid dovetail slot opening on the block's BOTTOM face (base_width at the opening, top_width at the undercut, height deep); auto-positioned from the block's bbox; use for ANY dovetail / trapezoidal groove request
- make_pitched_roof(base_length, base_width, ridge_height, overhang=0, origin=(0,0,0)) — gable roof; origin must sit exactly on the wall top height
- make_flat_roof(length, width, thickness, origin=(0,0,0), overhang=0) — flat roof slab

Rules:
1. Produce 2 to 5 sub-plans only.
2. Each sub-plan is a JSON object: {"id": "s1", "goal": "what to build in one sentence", "primitives": ["make_rail"], "depends_on": ["s0"]}
3. Sub-plans MUST be connected: every sub-plan after the first names the variables/parts produced by earlier sub-plans it uses (depends_on).
4. The LAST sub-plan must assemble everything from earlier sub-plans into a single part assigned to `result`.
5. If the user is editing an existing part, the first sub-plan reuses the current part (`result` already holds it) and later sub-plans modify it in place.
6. Choose primitives from the catalog only; never invent helpers that don't exist.
7. Primitives accept natural-language alias keywords (thickness/depth/length/pattern/rows/cols/grid_size/spacing/base_object etc.) and ignore unknown ones — but ALWAYS prefer the documented parameter names and the user's exact dimensions.
8. Never plan fillet/chamfer/edge operations — the primitives are already finished shapes; no model code may call .fillet()/.chamfer()/.edges()/.shell().

Reply with ONLY a JSON object, no markdown fences, no commentary:
{"subplans": [...]}"""

CODE_PROMPT = """You are a CadQuery code synthesist. You write ONE sub-plan of a larger CAD script as executable Python.

Input you receive:
- FULL PLAN: all sub-plans of the part.
- CURRENT SUB-PLAN: the one you are coding now.
- EARLIER CODE: variables already defined by previous sub-plans (reuse them by name).

Rules:
1. Return ONLY executable CadQuery Python for the CURRENT sub-plan. No markdown fences, no commentary, no import statements (imports are added by the merge step; `import cadquery as cq` and `from cad_primitives import *` are already in scope).
2. Define variables whose names match the plan's sub-plan ids/goals (e.g. rail_1, cross_beam, bracket_a).
3. For the FINAL sub-plan: assign the finished part to a variable named `result`, built from the earlier variables (or from the existing `result` when editing).
4. Use exactly the primitives chosen in the plan; do not hand-write low-level geometry that the plan assigns to a catalog primitive. For arrays/grids of repeated geometry (pins, studs, hole arrays), use the dedicated helper in ONE call — never generate dozens of individual parts and manual unions.
5. Use the primitives' documented parameter names with the user's exact numbers. Never call .fillet(), .chamfer(), .edges(), or .shell() — those operations fail on the primitives' solid geometry.
6. Do not call show_object() or any exporter.
"""

REPAIR_PROMPT = """You fix a failing CadQuery script. You receive the full plan, the failed script, and the sandbox execution error.

Rules:
1. Return the COMPLETE corrected script (all sub-plan sections concatenated, in order). No markdown fences, no commentary, no import statements (imports are added by the merge step; `import cadquery as cq` and `from cad_primitives import *` are already in scope).
2. The script must end by assigning the finished part to `result`.
3. Fix the reported error while preserving the plan's intent.
4. Note: safe_union(*parts) accepts ANY number of parts (safe_union(a, b, c, ...)) — passing many parts at once is fine and is NOT the bug.
5. If the error mentions a TIMEOUT or a heavy boolean operation, rebuild the geometry with the high-level cad_primitives helpers so arrays are created in a single extrude and ONE boolean op (e.g. make_pin_grid) — never generate hundreds of individual parts or per-part unions.
6. Call each primitive with its real parameter names. make_pin_grid accepts aliases (rows, cols, grid_size=(r, c), spacing, pin_diameter, pin_width, base_object) and auto-derives pitch from the plate size; make_beam accepts thickness as an alias for height. Use the user's exact dimensions.
7. Never call .fillet(), .chamfer(), .edges(), or .shell() — if the error mentions chamfer/fillet/edges, simply DELETE that call and rebuild the shape with the primitives.
8. Do not call show_object() or any exporter.
"""

# Per-message planner notes: targeted reminders that are stickier than rules
# buried in a long prompt from turn one.
_PLANNER_NOTES = {
    ("screw", "bolt", "thread", "fastener", "m2", "m3", "m4", "m5", "m6", "m8"): (
        "This request involves a screw/bolt/fastener: plan a single make_bolt(...) part "
        "mapped from metric sizes (M3 -> shank_diameter=3). No helix/thread sweeps by hand."
    ),
    ("chassis", "frame", "rail"): (
        "This is a structural frame: plan make_rail(...) side members plus "
        "add_crossmember(...) and bolt_pattern_holes(...) instead of freehand boxes."
    ),
    ("angle bracket", "l-bracket", "l bracket", "bracket"): (
        "This involves an angle/L bracket: plan make_l_bracket(...) with hole_diameter/hole_inset "
        "rather than hand-written polylines."
    ),
    ("heatsink", "cooling pin", "pin grid", "array of pins", "radiator", "fins"): (
    "This is a heatsink/pin-array: plan a flat base plate (make_beam with thickness for the "
    "plate height) plus ONE make_pin_grid(...) call for all the pins. USE THE USER'S EXACT "
    "DIMENSIONS — copy the plate length/width/thickness and pin size/height verbatim from the "
    "request; do not invent or round numbers. The pin grid must be built on the base plate "
    "(make_pin_grid(base, pins_x=..., pins_y=..., pin_size=..., pin_height=...)); pitch is "
    "auto-derived from the plate size, so a 120mm plate with 12 pins per side gets 10mm spacing. "
    "Do NOT plan individual pins or manual per-pin unions."
),
    ("dovetail", "trapezoid"): (
        "This involves a dovetail/trapezoidal slot: plan the block (make_beam) plus ONE "
        "cut_dovetail(block, base_width, top_width, height, ...) call with the user's exact "
        "widths and depth. Never hand-write trapezoid profiles or invent other helpers."
    ),
    ("cylinder", "cylindrical", "spacer", "disc", "shaft"): (
        "This involves cylindrical section(s): plan ONE make_cylinder(diameter, height, position=(x, y, z)) "
        "call per section with the user's exact diameters/heights — never model cylinders as boxes. "
        "A cylinder centered at the origin spans z=[-height/2, +height/2], so stack sections with "
        "position=(0, 0, previous_top + next_height/2)."
    ),
    ("house", "building", "room", "roof", "cabin"): (
        "This is a building: plan make_box_room(...) or per-wall make_wall(...), cut_opening(...) "
        "for doors/windows BEFORE unioning each wall, then make_pitched_roof(...)/make_flat_roof(...) "
        "with origin exactly at the wall top height."
    ),
}


def reinforce_planning(user_message: str) -> str:
    lowered = user_message.lower()
    extra = ""
    for keywords, note in _PLANNER_NOTES.items():
        if any(kw in lowered for kw in keywords):
            extra += f"\n- {note}"
    return extra


def strip_code_fences(text: str) -> str:
    """Defensive cleanup in case a model wraps its answer in ```python fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_json(text: str):
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _llm(state: CadState, system_prompt: str, conversation_text: str) -> str:
    """One LLM call whose provider/model/temperature come from the workflow
    state, dispatched to gemini.py or groq.py. The provider is read from
    state["provider"], falling back to the LLM_PROVIDER env var."""
    sub = dict(state)
    sub["system_prompt"] = system_prompt
    sub["conversation_text"] = conversation_text
    provider = (state.get("provider") or os.environ.get("LLM_PROVIDER") or "gemini").lower()
    if provider == "gemini":
        return gemini.generate_from_state(sub)
    if provider == "groq":
        return groq.generate_from_state(sub)
    raise ValueError(
        f"Unknown LLM provider '{provider}'. Set LLM_PROVIDER to 'gemini' or 'groq'."
    )


# ---------------------------------------------------------------------------
# Sandbox execution (in-process; exec_worker.py removed)
# ---------------------------------------------------------------------------
ALLOWED_MODULES = {
    "cadquery", "math", "cadquery.selectors", "cadquery.occ_impl", "OCP",
    "cad_primitives",
}

_real_import = __import__


def _restricted_import(name, *args, **kwargs):
    top_level = name.split(".")[0]
    if top_level not in ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not permitted inside generated CAD scripts."
        )
    return _real_import(name, *args, **kwargs)


def build_safe_globals() -> dict:
    safe_builtins = {
        k: v
        for k, v in vars(_builtins_module).items()
        if k not in ("open", "exec", "eval", "compile", "input", "__import__", "exit", "quit")
    }
    safe_builtins["__import__"] = _restricted_import
    return {"__builtins__": safe_builtins}


def run_in_sandbox(code: str) -> tuple[bool, str, str, str]:
    """
    Runs generated code in-process with restricted builtins and an import
    allow-list, guarded by a watchdog thread. Returns (ok, error, stl_path, step_path).

    Honest limitation: exec_worker.py (out-of-process) was removed per project
    decision, so a genuinely stuck script leaks a daemon thread instead of being
    hard-killed; the timeout still surfaces the failure to the repair loop.
    """
    run_id = str(uuid.uuid4())[:8]
    stl_path = os.path.join(OUTPUT_DIR, f"part_{run_id}.stl")
    step_path = os.path.join(OUTPUT_DIR, f"part_{run_id}.step")

    outcome: dict = {}

    def _run():
        try:
            local_scope = {}
            exec(code, build_safe_globals(), local_scope)
            if "result" not in local_scope:
                outcome["error"] = "Generated script never assigned a 'result' variable."
                return
            import cadquery as cq

            cad_object = local_scope["result"]
            cq.exporters.export(cad_object, stl_path, cq.exporters.ExportTypes.STL)
            cq.exporters.export(cad_object, step_path, cq.exporters.ExportTypes.STEP)
            outcome["ok"] = True
        except Exception as e:
            outcome["error"] = f"Generated code raised an exception: {e}"

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(SANDBOX_TIMEOUT_SECONDS)

    if worker.is_alive():
        return (
            False,
            f"Generated script timed out after {SANDBOX_TIMEOUT_SECONDS} seconds — "
        f"likely an infinite loop or an extremely heavy boolean operation. "
        f"Rebuild the geometry with the high-level cad_primitives helpers "
        f"(e.g. make_pin_grid for pin arrays, make_box_room for rooms) so "
        f"arrays are created in a single extrude and ONE boolean op — never "
        f"generate hundreds of individual parts or per-part unions.",
            "",
            "",
        )

    if outcome.get("ok"):
        return True, "", stl_path, step_path
    return False, outcome.get("error", "Unknown error"), "", ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def understand_request(state: CadState) -> dict:
    """Step 1 — classify what kind of CAD task this is."""
    context = [f"USER REQUEST: {state['user_message']}"]
    if state.get("last_good_code"):
        context.append("PREVIOUS WORKING SCRIPT EXISTS — the user is likely editing it.")
    if state.get("base_step_path"):
        context.append("A BASE STEP FILE IS LOADED — edits apply on top of it.")
    raw = _llm(state, INTENT_PROMPT, "\n".join(context))

    parsed = _extract_json(raw)
    task_type = "edit" if (state.get("last_good_code") or state.get("base_step_path")) else "new"
    intent = state["user_message"]
    if isinstance(parsed, dict):
        task_type = parsed.get("task_type") or task_type
        intent = parsed.get("brief") or intent
    return {"task_type": task_type, "intent": intent}


def plan_cad(state: CadState) -> dict:
    """Step 2 — convert the request into connected, structured sub-plans."""
    history_lines = [
        f"{turn['role'].upper()}: {turn['content']}"
        for turn in state["history"][-HISTORY_WINDOW:]
    ]
    context = "\n".join(
        [
            "CONVERSATION:",
            "\n".join(history_lines),
            f"\nCLASSIFIED TASK TYPE: {state['task_type']}",
            f"\nUSER REQUEST: {state['user_message']}",
            f"\nPLANNER NOTES:{reinforce_planning(state['user_message'])}",
        ]
    )
    if state.get("last_good_code"):
        context += f"\n\nCURRENT WORKING SCRIPT (edits must modify this):\n{state['last_good_code']}"
    if state.get("base_step_path"):
        context += "\n\nA BASE STEP FILE is loaded; `result` already holds the imported part."

    raw = _llm(state, PLAN_PROMPT, context)
    parsed = _extract_json(raw)
    subplans = parsed.get("subplans") if isinstance(parsed, dict) else None
    if not isinstance(subplans, list) or not subplans:
        # Fallback: one assembly sub-plan so the pipeline always continues.
        subplans = [{"id": "s1", "goal": state["user_message"], "primitives": [], "depends_on": []}]
    return {"plan": subplans[:MAX_SUBPLANS]}


def generate_code(state: CadState) -> dict:
    """Step 3 — one focused code-synthesis call per sub-plan; later fragments
    see earlier fragments so the pieces stay connected."""
    fragments = []
    plan = state["plan"]
    for idx, subplan in enumerate(plan):
        is_last = idx == len(plan) - 1
        context = "\n".join(
            [
                f"FULL PLAN:\n{json.dumps(plan, indent=1)}",
                f"\nCURRENT SUB-PLAN (you code this one):\n{json.dumps(subplan)}",
                (
                    f"\nORIGINAL USER REQUEST (use these EXACT numbers, units in mm — never "
                    f"change, round, or invent dimensions):\n{state['user_message']}"
                ),
                (
                    f"\nCLASSIFIED INTENT (dimension summary — copy measurements verbatim):\n"
                    f"{state.get('intent', '')}"
                ),
            ]
        )
        if state.get("base_step_path"):
            context += "\n\nNOTE: a base STEP file is already imported into `result`; build on top of it."
        if fragments:
            context += "\n\nEARLIER CODE (variables already defined; reuse them by name):\n" + "\n".join(fragments)
        if state.get("last_good_code"):
            context += f"\n\nCURRENT WORKING SCRIPT (user is editing it):\n{state['last_good_code']}"
        context += f"\n\nIS FINAL SUB-PLAN: {'yes — you MUST assign the finished part to `result`' if is_last else 'no'}"
        fragments.append(strip_code_fences(_llm(state, CODE_PROMPT, context)))
    return {"code_fragments": fragments}


def merge_code(state: CadState) -> dict:
    """Step 4a — connect all fragments into a single working script."""
    header = ["import cadquery as cq", "from cad_primitives import *"]
    if state.get("base_step_path"):
        header.append(f'result = cq.importers.importStep(r"{state["base_step_path"]}")')
    body = []
    for i, fragment in enumerate(state["code_fragments"], start=1):
        body.append(f"# ===== sub-plan {i} =====")
        body.append(fragment)
    return {"merged_code": "\n".join(header + body)}


def execute_code(state: CadState) -> dict:
    """Step 4b — run the merged script in the sandbox.

    Dedup guard: if this exact script already ran and failed (e.g. the repair
    node returned identical code), return the stored result instead of paying
    for a second execution and burning another repair attempt."""
    code = state["merged_code"]
    executed = state.get("executed", {})
    if code in executed:
        return {"executed": executed, **executed[code]}

    ok, error, stl_path, step_path = run_in_sandbox(code)
    result = {
        "ok": ok,
        "execution_error": error,
        "stl_path": stl_path,
        "step_path": step_path,
    }
    return {"executed": {**executed, code: result}, **result}


def repair_code(state: CadState) -> dict:
    """Repair node — send the sandbox error back to the model for a fix."""
    context = "\n".join(
        [
            f"PLAN:\n{json.dumps(state['plan'], indent=1)}",
            (
                f"\nORIGINAL USER REQUEST (use these EXACT numbers, units in mm — never "
                f"change, round, or invent dimensions):\n{state['user_message']}"
            ),
            f"\nFAILED SCRIPT:\n{state['merged_code']}",
            f"\nSANDBOX ERROR:\n{state['execution_error']}",
        ]
    )
    return {
        "fix_attempts": state.get("fix_attempts", 0) + 1,
        "code_fragments": [strip_code_fences(_llm(state, REPAIR_PROMPT, context))],
    }


def should_repair(state: CadState) -> str:
    """Conditional edge after execute_code: repair while attempts remain."""
    if state.get("ok"):
        return "finalize"
    if state.get("fix_attempts", 0) < MAX_FIX_ATTEMPTS:
        return "repair"
    return "finalize"


def finalize(state: CadState) -> dict:
    """Step 5 — build the response payload with artifact URLs."""
    if state.get("ok"):
        return {
            "response": {
                "status": "success",
                "code": state["merged_code"],
                "stl_url": f"/static/{os.path.basename(state['stl_path'])}",
                "step_url": f"/static/{os.path.basename(state['step_path'])}",
            }
        }
    return {
        "response": {
            "status": "failed",
            "error": state.get("execution_error", "Unknown error"),
            "attempts": state.get("fix_attempts", 0) + 1,
        }
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_workflow():
    builder = StateGraph(CadState)
    builder.add_node("understand_request", understand_request)
    builder.add_node("plan_cad", plan_cad)
    builder.add_node("generate_code", generate_code)
    builder.add_node("merge_code", merge_code)
    builder.add_node("execute_code", execute_code)
    builder.add_node("repair_code", repair_code)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "understand_request")
    builder.add_edge("understand_request", "plan_cad")
    builder.add_edge("plan_cad", "generate_code")
    builder.add_edge("generate_code", "merge_code")
    builder.add_edge("merge_code", "execute_code")
    builder.add_conditional_edges(
        "execute_code",
        should_repair,
        {"repair": "repair_code", "finalize": "finalize"},
    )
    builder.add_edge("repair_code", "merge_code")
    builder.add_edge("finalize", END)

    return builder.compile()


WORKFLOW = build_workflow()