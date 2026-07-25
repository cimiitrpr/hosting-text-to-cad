"""
exec_worker.py
--------------
Runs untrusted, LLM-generated CadQuery code in an isolated OS process.

Why this exists:
    The naive approach (`exec(code)` inside your FastAPI process) means a bad
    or malicious prompt can crash your API server, hang it forever, or read/
    write files anywhere the server can. Running the code in a *separate
    process* fixes the "crash the whole server" and "run forever" problems
    (we kill the process on timeout). Restricting builtins/imports fixes most
    of the "read/write anything" problem.

Honest limitation: this is defense-in-depth for a student project, not a
production security boundary. A determined attacker can likely still escape
a builtins-restricted exec() in the same OS process. If you ever expose this
publicly beyond a demo, run this file inside a real container/sandbox
(e.g. Docker with --network none and a read-only filesystem, or a hosted
code-sandbox service like e2b.dev, which has a free tier built exactly for
"run LLM-generated code safely"). Swapping this file's internals for an
e2b.dev call is a drop-in upgrade later — you don't have to redesign anything.
"""

import sys
import json
import builtins as _builtins_module

# `resource` is Unix-only (Linux/macOS). It does not exist on Windows, so
# import it defensively — this is what lets local dev on Windows run this
# file at all, while still applying the real CPU limit on Render (Linux).
try:
    import resource
except ImportError:
    resource = None

# ---- 1. Resource limits (Linux/macOS only, no-op on Windows) ----
try:
    if resource is not None:
        # Max 35 seconds of CPU time for the generated script itself (complex
        # multi-part assemblies with several boolean unions can be genuinely slow)
        resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
    # NOTE: we intentionally do NOT set RLIMIT_AS (virtual address space) here.
    # CAD kernels like OCCT/OCP reserve large virtual memory regions up front
    # even for modest actual (resident) memory use, and STEP import in
    # particular is heavier than generating a primitive shape. A tight
    # RLIMIT_AS causes glibc to hard-abort with a cryptic
    # "cannot allocate memory for thread-local data" error well before real
    # physical memory is exhausted. Real memory protection is better left to
    # the hosting platform's container-level limit (e.g. Render kills the
    # process cleanly with an OOM signal instead of this glibc abort).
except Exception:
    pass  # belt-and-braces — setrlimit can still fail for other reasons

# ---- 2. Restricted import allow-list ----
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


def build_safe_globals():
    safe_builtins = {
        k: v
        for k, v in vars(_builtins_module).items()
        if k not in ("open", "exec", "eval", "compile", "input", "__import__", "exit", "quit")
    }
    safe_builtins["__import__"] = _restricted_import
    return {"__builtins__": safe_builtins}


def main():
    if len(sys.argv) != 4:
        print(json.dumps({"ok": False, "error": "usage: exec_worker.py <code_path> <stl_out> <step_out>"}))
        sys.exit(1)

    code_path, stl_out, step_out = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(code_path, "r") as f:
        code = f.read()

    safe_globals = build_safe_globals()
    local_scope = {}

    try:
        exec(code, safe_globals, local_scope)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Generated code raised an exception: {e}"}))
        sys.exit(1)

    if "result" not in local_scope:
        print(json.dumps({"ok": False, "error": "Generated script never assigned a 'result' variable."}))
        sys.exit(1)

    cad_object = local_scope["result"]

    try:
        import cadquery as cq

        cq.exporters.export(cad_object, stl_out, cq.exporters.ExportTypes.STL)
        cq.exporters.export(cad_object, step_out, cq.exporters.ExportTypes.STEP)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Export failed: {e}"}))
        sys.exit(1)

    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()