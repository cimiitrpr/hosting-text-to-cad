import os
os.environ.setdefault("MAX_FIX_ATTEMPTS", "2")

import gemini
import chat_graph

GOOD = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)\n"
BAD = "import cadquery as cq\nresult = undefined_symbol_that_fails\n"

def base_state(message="make a cube"):
    return {
        "message": message,
        "history": [{"role": "user", "content": message}],
        "last_code": None,
        "base_step_path": None,
    }

# 1. success path
gemini.generate_from_state = lambda s: GOOD
r1 = chat_graph.CHAT_GRAPH.invoke(base_state())
assert r1["ok"] is True, r1
assert r1["code"].startswith("import cadquery"), r1
assert os.path.exists(r1["stl_path"]) and os.path.exists(r1["step_path"]), r1
assert r1["fix_attempts"] == 0
print("PASS success path  -> stl:", os.path.basename(r1["stl_path"]))

# 2. fix-loop recovery: first attempt broken, second good
calls = {"n": 0}
def flaky(state):
    calls["n"] += 1
    return GOOD if calls["n"] > 1 else BAD
gemini.generate_from_state = flaky
r2 = chat_graph.CHAT_GRAPH.invoke(base_state("chassis"))
assert r2["ok"] is True, r2
assert r2["fix_attempts"] == 1, r2
assert calls["n"] == 2, calls
print("PASS fix-loop path  -> fixed after", r2["fix_attempts"], "attempt(s)")

# 3. exhausted path: always broken, should end with ok=False after MAX_FIX_ATTEMPTS fixes
gemini.generate_from_state = lambda s: BAD
r3 = chat_graph.CHAT_GRAPH.invoke(base_state("house with a roof"))
assert r3["ok"] is False, r3
assert r3["fix_attempts"] == int(os.environ["MAX_FIX_ATTEMPTS"]), r3
assert "exception" in r3["error"].lower(), r3
print("PASS exhausted path -> failed after", r3["fix_attempts"] + 1, "total attempts")

# 4. build_prompt reinforcement + STEP prefix wiring
gemini.generate_from_state = lambda s: GOOD
r4 = chat_graph.CHAT_GRAPH.invoke({
    **base_state("add an M3 screw"),
    "base_step_path": r"C:\fake\base.step",
    "last_code": GOOD,
})
assert r4["ok"] is True, r4
print("PASS keyword/STEP-prefix wiring (conversation built without error)")

print("\nALL GRAPH TESTS PASSED")
