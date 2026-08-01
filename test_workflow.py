import os
os.environ.setdefault("MAX_FIX_ATTEMPTS", "2")

import json
import gemini
import cad_workflow as wf

INTENT = '{"task_type": "frame", "brief": "a ladder frame chassis"}'
PLAN = json.dumps({
    "subplans": [
        {"id": "s1", "goal": "two parallel rails", "primitives": ["make_rail"], "depends_on": []},
        {"id": "s2", "goal": "assemble rails into result", "primitives": ["safe_union"], "depends_on": ["s1"]},
    ]
})
S1_CODE = "rail_1 = cq.Workplane('XY').box(100, 10, 5)"
S2_CODE = "result = rail_1"
REPAIR_CODE = "rail_1 = cq.Workplane('XY').box(100, 10, 5)\nresult = rail_1\n"

def base_state():
    return {
        "session_id": "t1",
        "user_message": "make a chassis frame with two rails",
        "history": [{"role": "user", "content": "make a chassis frame with two rails"}],
        "last_good_code": None,
        "base_step_path": None,
    }

# ---- 1. SUCCESS PATH ----
gemini.generate_from_state = lambda s: {
    wf.INTENT_PROMPT: INTENT,
    wf.PLAN_PROMPT: PLAN,
    wf.CODE_PROMPT: S2_CODE if "IS FINAL SUB-PLAN: yes" in s["conversation_text"] else S1_CODE,
    wf.REPAIR_PROMPT: REPAIR_CODE,
}[s["system_prompt"]]

r1 = wf.WORKFLOW.invoke(base_state())
resp1 = r1["response"]
assert resp1["status"] == "success", resp1
assert r1["task_type"] == "frame", r1
assert len(r1["plan"]) == 2, r1
assert len(r1["code_fragments"]) == 2, r1
assert r1["merged_code"].startswith("import cadquery as cq\nfrom cad_primitives import *"), r1["merged_code"]
assert "rail_1 = cq.Workplane" in r1["merged_code"], r1["merged_code"]
assert "result = rail_1" in r1["merged_code"], r1["merged_code"]
assert resp1["stl_url"].startswith("/static/") and resp1["step_url"].startswith("/static/"), resp1
print("PASS success path: task_type=%s, fragments=%d, response=%s" % (r1["task_type"], len(r1["code_fragments"]), resp1["status"]))

# ---- 2. FALLBACK PLAN (model returns garbage JSON) ----
gemini.generate_from_state = lambda s: {
    wf.INTENT_PROMPT: "not json at all",
    wf.PLAN_PROMPT: "sorry i cannot plan",
    wf.CODE_PROMPT: "result = cq.Workplane('XY').box(1, 1, 1)",
    wf.REPAIR_PROMPT: "result = cq.Workplane('XY').box(1, 1, 1)",
}[s["system_prompt"]]

r2 = wf.WORKFLOW.invoke(base_state())
assert r2["plan"] == [{"id": "s1", "goal": "make a chassis frame with two rails", "primitives": [], "depends_on": []}], r2["plan"]
assert r2["response"]["status"] == "success", r2["response"]
print("PASS fallback path: garbage JSON -> single sub-plan fallback")

# ---- 3. REPAIR LOOP (first execute fails, repair fixes it) ----
calls = {"exec": 0}
def sandbox_flaky(code):
    calls["exec"] += 1
    if calls["exec"] == 1:
        return False, "Generated code raised an exception: boom", "", ""
    return True, "", r"D:\x\part.stl", r"D:\x\part.step"
wf.run_in_sandbox = sandbox_flaky

gemini.generate_from_state = lambda s: {
    wf.INTENT_PROMPT: INTENT,
    wf.PLAN_PROMPT: PLAN,
    wf.CODE_PROMPT: S2_CODE if "IS FINAL SUB-PLAN: yes" in s["conversation_text"] else S1_CODE,
    wf.REPAIR_PROMPT: REPAIR_CODE,
}[s["system_prompt"]]

r3 = wf.WORKFLOW.invoke(base_state())
assert r3["fix_attempts"] == 1, r3
assert r3["response"]["status"] == "success", r3["response"]
assert calls["exec"] == 2, calls
print("PASS repair loop: failed once, repaired, succeeded (attempts=%d)" % r3["fix_attempts"])

# ---- 4. EXHAUSTED (always fails) ----
wf.run_in_sandbox = lambda code: (False, "Generated code raised an exception: always", "", "")
gemini.generate_from_state = lambda s: {
    wf.INTENT_PROMPT: INTENT,
    wf.PLAN_PROMPT: PLAN,
    wf.CODE_PROMPT: "result = boom",
    wf.REPAIR_PROMPT: "result = boom2",
}[s["system_prompt"]]

r4 = wf.WORKFLOW.invoke(base_state())
resp4 = r4["response"]
assert resp4["status"] == "failed", resp4
assert resp4["attempts"] == 3, resp4  # 1 original + 2 fixes (MAX_FIX_ATTEMPTS=2)
assert "always" in resp4["error"], resp4
print("PASS exhausted: failed after %d total attempts" % resp4["attempts"])

# ---- 5. REAL END-TO-END (stub LLM, REAL sandbox + CadQuery export) ----
import importlib
importlib.reload(wf)
importlib.reload(gemini)

gemini.generate_from_state = lambda s: {
    wf.INTENT_PROMPT: INTENT,
    wf.PLAN_PROMPT: PLAN,
    wf.CODE_PROMPT: S2_CODE if "IS FINAL SUB-PLAN: yes" in s["conversation_text"] else S1_CODE,
    wf.REPAIR_PROMPT: REPAIR_CODE,
}[s["system_prompt"]]

r5 = wf.WORKFLOW.invoke(base_state())
assert r5["response"]["status"] == "success", r5["response"]
assert os.path.exists(r5["stl_path"]) and os.path.exists(r5["step_path"]), r5
assert os.path.getsize(r5["stl_path"]) > 100, r5["stl_path"]
assert r5["response"]["stl_url"].startswith("/static/part_"), r5["response"]
print("REAL E2E: stl=%s (%d bytes) step=%s" % (r5["stl_path"], os.path.getsize(r5["stl_path"]), r5["step_path"]))

# ---- 6. REAL REPAIR LOOP (broken code hits the real sandbox, repair fixes it) ----
gen_calls = {"n": 0}
def gen_repair_flow(state):
    gen_calls["n"] += 1
    if state["system_prompt"] == wf.REPAIR_PROMPT:
        return REPAIR_CODE
    if state["system_prompt"] == wf.CODE_PROMPT:
        return "broken_undefined_var"  # first code gen produces a broken fragment
    if state["system_prompt"] == wf.INTENT_PROMPT:
        return INTENT
    return PLAN
gemini.generate_from_state = gen_repair_flow

r6 = wf.WORKFLOW.invoke(base_state())
assert r6["fix_attempts"] == 1, r6
assert r6["response"]["status"] == "success", r6["response"]
assert os.path.getsize(r6["stl_path"]) > 100, r6
print("REAL REPAIR LOOP: broken code -> real sandbox error -> repaired -> exported (%s)" % r6["stl_path"])

print("\nALL WORKFLOW TESTS PASSED")