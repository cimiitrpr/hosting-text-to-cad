"""
cad_primitives.py
-----------------
A small helper library the LLM is instructed (via the system prompt) to call
for anything beyond a single primitive shape.

Why this exists:
    Asking an LLM to freehand raw CadQuery for something like "a car chassis"
    from nothing produces unreliable, often-broken code, because it has to
    get dozens of coordinates right in one shot. Instead, we give it a small
    vocabulary of pre-tested building blocks (beams, rails, cross-members,
    mounting holes) and instruct it to *compose* those. This is the same
    "constrained generation" idea used in real production LLM-codegen tools:
    shrink the model's problem from "write correct low-level code" to
    "pick the right high-level calls and numbers."

This file gets copied next to the generated script at execution time so
`from cad_primitives import *` works inside the sandboxed subprocess.
"""

import cadquery as cq


def make_beam(length, width, height, origin=(0, 0, 0)):
    """A simple rectangular structural beam, centered at `origin`."""
    return (
        cq.Workplane("XY")
        .center(origin[0], origin[1])
        .workplane(offset=origin[2])
        .box(length, width, height)
    )


def make_rail(length, width, height, hole_spacing=None, hole_diameter=6):
    """
    A structural rail (long beam) with optional evenly spaced mounting holes
    drilled through the top face. Useful as the two long side-members of a
    ladder-frame chassis.
    """
    rail = cq.Workplane("XY").box(length, width, height)
    if hole_spacing:
        n_holes = max(2, int(length // hole_spacing))
        rail = (
            rail.faces(">Z")
            .workplane()
            .rarray(hole_spacing, 1, n_holes, 1)
            .hole(hole_diameter)
        )
    return rail


def add_crossmember(base, length, width, height, x_position):
    """Fuses a cross-member beam onto `base` at a given x position."""
    cross = (
        cq.Workplane("XY")
        .center(x_position, 0)
        .box(width, length, height)
    )
    return base.union(cross)


def bolt_pattern_holes(workplane, diameter, positions):
    """Drill holes at explicit (x, y) positions on the current workplane."""
    for x, y in positions:
        workplane = workplane.moveTo(x, y).hole(diameter)
    return workplane


def make_wheel_mount(diameter, width, position):
    """
    A cylindrical wheel/axle mount, oriented along the Y axis, placed at
    `position` = (x, y, z). Useful for chassis/vehicle-style builds.
    """
    x, y, z = position
    return (
        cq.Workplane("YZ")
        .center(y, z)
        .workplane(offset=x)
        .circle(diameter / 2)
        .extrude(width)
    )


def safe_union(base, addition):
    """
    Union two solids, raising a clear error instead of a cryptic OCC failure
    if the parts don't actually intersect or touch (a common mistake when
    positioning cross-members or mounts).
    """
    try:
        return base.union(addition)
    except Exception as e:
        raise ValueError(
            f"Union failed — the two parts likely don't touch or overlap. Original error: {e}"
        )


def make_bolt(shank_diameter, length, head_diameter=None, head_height=None, hex_head=True):
    """
    A screw/bolt: a cylindrical shank plus a head, with the THREAD SHOWN AS A
    COSMETIC HELICAL GROOVE rather than fully carved thread geometry.

    Why: true 3D thread geometry (a swept V-profile along a helix, intersected
    with the shank) is expensive to compute and one of the most failure-prone
    things to generate reliably — even professional CAD tools usually treat
    threads as a cosmetic/annotated feature rather than exact geometry, since
    exact thread geometry is rarely needed except for manufacturing checks.
    This primitive gives you a fast, reliable bolt shape with a shallow
    decorative helical groove that reads clearly as "threaded" in a preview
    and print, without the computational cost/fragility of a true thread.

    shank_diameter: nominal diameter of the threaded rod (e.g. 3 for M3)
    length: length of the shank, not counting the head
    head_diameter: defaults to 1.8x shank_diameter if not given
    head_height: defaults to 0.6x shank_diameter if not given
    hex_head: True for a hex head, False for a plain cylindrical head
    """
    if head_diameter is None:
        head_diameter = shank_diameter * 1.8
    if head_height is None:
        head_height = shank_diameter * 0.6

    shank = cq.Workplane("XY").circle(shank_diameter / 2).extrude(length)

    # Cosmetic thread: a shallow helical groove cut into the shank surface.
    # This is deliberately a light visual cue, not a mechanically accurate
    # thread profile.
    pitch = max(shank_diameter * 0.2, 0.4)
    groove_depth = shank_diameter * 0.06
    helix = cq.Wire.makeHelix(pitch=pitch, height=length, radius=shank_diameter / 2)
    groove_profile = (
        cq.Workplane("XZ")
        .center(shank_diameter / 2, 0)
        .rect(groove_depth * 2, pitch * 0.3)
    )
    try:
        groove = groove_profile.sweep(helix, isFrenet=True)
        shank = shank.cut(groove)
    except Exception:
        # If the cosmetic groove sweep fails for any geometric edge case,
        # fall back silently to a plain (unthreaded-looking) shank rather
        # than failing the whole part — a plain shank is still a usable bolt.
        pass

    if hex_head:
        head = (
            cq.Workplane("XY")
            .workplane(offset=length)
            .polygon(6, head_diameter)
            .extrude(head_height)
        )
    else:
        head = (
            cq.Workplane("XY")
            .workplane(offset=length)
            .circle(head_diameter / 2)
            .extrude(head_height)
        )

    return shank.union(head)