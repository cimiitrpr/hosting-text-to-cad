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
    A screw/bolt: a cylindrical shank plus a head, with the THREAD SHOWN AS
    CHEAP COSMETIC RING GROOVES rather than a continuous helix or fully
    carved thread geometry.

    Why ring grooves instead of a helix: an earlier version of this function
    swept a profile along a full-length helix and cut it from the shank.
    That is correct CadQuery usage, but computationally heavy — OCCT has to
    build and boolean-intersect a long, high-curvature swept solid — and it
    reliably timed out on constrained hosting (e.g. Render's free tier).
    A stack of simple concentric ring grooves reads visually as "threaded"
    at a glance and is dramatically cheaper: each cut is a small, simple
    boolean operation instead of one large complex one. This is a deliberate
    trade of geometric accuracy for reliability — not something you'd do for
    a manufacturing drawing, but a reasonable and honest choice for a fast,
    dependable preview/demo tool.

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

    # Cosmetic thread: a bounded number of shallow ring grooves, cheap to
    # compute regardless of screw length. Capped at MAX_RINGS so a very long
    # screw can't balloon into dozens of boolean operations.
    pitch = max(shank_diameter * 0.35, 0.5)
    groove_depth = shank_diameter * 0.06
    groove_width = pitch * 0.35
    MAX_RINGS = 12
    n_rings = min(int(length // pitch) - 1, MAX_RINGS)

    if n_rings > 0:
        try:
            for i in range(1, n_rings + 1):
                z = i * (length / (n_rings + 1))
                ring_tool = (
                    cq.Workplane("XY")
                    .workplane(offset=z - groove_width / 2)
                    .circle(shank_diameter / 2 + 0.1)
                    .circle(shank_diameter / 2 - groove_depth)
                    .extrude(groove_width)
                )
                shank = shank.cut(ring_tool)
        except Exception:
            # If ring grooves fail for any geometric edge case, fall back
            # silently to a plain shank rather than failing the whole part.
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


def make_l_bracket(leg1_length, leg2_length, width, thickness, hole_diameter=None, hole_inset=None):
    """
    An L-shaped angle bracket: two perpendicular legs of a solid cross-section,
    extruded along `width`. Optionally drills a mounting hole in each leg.

    leg1_length, leg2_length: outer length of each leg (measured from the
        outer corner, including the shared thickness)
    width: extrusion depth (how "deep"/long the bracket is along the bend line)
    thickness: material thickness of both legs
    hole_diameter: if given, drills one hole through each leg's face
    hole_inset: distance of each hole's center from the outer corner along
        its leg (defaults to half the thickness in from the leg's midpoint area
        if not given — kept simple/central by default)
    """
    pts = [
        (0, leg1_length),
        (thickness, leg1_length),
        (thickness, thickness),
        (leg2_length, thickness),
        (leg2_length, 0),
        (0, 0),
    ]
    profile = cq.Workplane("XY").polyline(pts).close()
    result = profile.extrude(width)

    if hole_diameter:
        if hole_inset is None:
            hole_inset = thickness * 1.5
        # Extrusion is along Z, so the flat top face (">Z") shares the same
        # X/Y coordinate space as the 2D profile above — hole positions can
        # be given directly in those same profile coordinates.
        hole_positions = [
            (thickness / 2, leg1_length - hole_inset),  # hole through the vertical leg
            (leg2_length - hole_inset, thickness / 2),  # hole through the horizontal leg
        ]
        result = (
            result.faces(">Z")
            .workplane()
            .pushPoints(hole_positions)
            .hole(hole_diameter)
        )

    return result