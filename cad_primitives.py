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