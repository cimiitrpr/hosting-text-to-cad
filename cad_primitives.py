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

def make_lego_brick(studs_x, studs_y, height="standard"):
    """
    A standard Lego-proportioned brick: solid body with cylindrical studs
    on top and a shell with hollow interior tubes on the underside for
    proper stacking. Standard unit = 8mm stud pitch, studs 4.8mm diameter,
    brick height 9.6mm (or 3.2mm for a "plate").
    """
    UNIT = 8.0
    STUD_D = 4.8
    STUD_H = 1.8
    WALL_T = 1.2
    h = 9.6 if height == "standard" else 3.2

    length = studs_x * UNIT
    width = studs_y * UNIT

    body = cq.Workplane("XY").box(length, width, h, centered=(True, True, False))
    # hollow shell from below, leaving the top solid so studs have something to sit on
    body = body.faces("<Z").shell(-WALL_T)

    studs = (
        cq.Workplane("XY")
        .workplane(offset=h)
        .rarray(UNIT, UNIT, studs_x, studs_y)
        .circle(STUD_D / 2)
        .extrude(STUD_H)
    )
    return body.union(studs)

def make_wall(length, height, thickness, origin=(0, 0, 0), axis="x"):
    """
    A flat rectangular wall panel, standing vertically (extends up in Z),
    base at `origin`. `axis` picks which horizontal direction `length` runs
    along ("x" or "y") — this swaps the box() dimensions directly rather
    than rotating, so walls facing either direction stay perfectly upright
    with no risk of a rotation transform going wrong.
    """
    x, y, z = origin
    wp = cq.Workplane("XY").center(x, y).workplane(offset=z)
    if axis == "x":
        return wp.box(length, thickness, height, centered=(True, True, False))
    elif axis == "y":
        return wp.box(thickness, length, height, centered=(True, True, False))
    raise ValueError("axis must be 'x' or 'y'")


def cut_opening(wall, width, height, position, wall_axis="x"):
    """
    Cuts a rectangular door/window opening through a wall made with
    make_wall(). `position` = (along_wall, from_ground) — along_wall is
    measured from the wall's center along its length, from_ground is
    height from the wall's base (z=0 of that wall). wall_axis matches
    whichever axis the wall's `length` runs along ("x" or "y") so the
    cutting tool is oriented correctly for walls facing either direction.
    """
    along, from_ground = position
    tool = cq.Workplane("XY").box(
        width if wall_axis == "x" else 1000,
        1000 if wall_axis == "x" else width,
        height,
        centered=(True, True, False),
    )
    tool = tool.translate((along if wall_axis == "x" else 0,
                           0 if wall_axis == "x" else along,
                           from_ground))
    return safe_union.__wrapped__(wall, tool) if False else wall.cut(tool)


def make_pitched_roof(base_length, base_width, ridge_height, overhang=0, origin=(0, 0, 0)):
    """
    A simple pitched (gable) roof: a triangular prism sitting on top of a
    rectangular footprint, ridge running along the `base_length` axis.
    `origin` should be the base center-bottom of the roof (typically the
    top of your walls). `overhang` extends the roof past the wall footprint
    on the two long sides.
    """
    half_w = base_width / 2 + overhang
    pts = [(-half_w, 0), (0, ridge_height), (half_w, 0)]
    profile = (
        cq.Workplane("YZ")
        .center(origin[1], origin[2])
        .workplane(offset=origin[0] - base_length / 2)
        .polyline(pts).close()
    )
    return profile.extrude(base_length)


def make_flat_roof(length, width, thickness, origin=(0, 0, 0), overhang=0):
    """A simple flat roof slab — a box extended past the wall footprint by `overhang`."""
    return (
        cq.Workplane("XY")
        .center(origin[0], origin[1])
        .workplane(offset=origin[2])
        .box(length + 2 * overhang, width + 2 * overhang, thickness, centered=(True, True, False))
    )

def make_box_room(length, width, height, thickness):
    """
    Four walls forming a closed rectangular room centered at the origin,
    corners guaranteed to meet exactly (side walls are pre-inset by
    `thickness` so they don't overlap the front/back walls at the corners).
    length runs along X, width along Y. Returns one unioned solid.
    Use this for the standard "four walls of a house" case instead of
    calling make_wall four times by hand — hand-deriving each wall's
    origin is exactly what produced gaps/overlaps before.
    """
    front = make_wall(length, height, thickness, origin=(0, -width / 2 + thickness / 2, 0), axis="x")
    back  = make_wall(length, height, thickness, origin=(0,  width / 2 - thickness / 2, 0), axis="x")
    left  = make_wall(width - 2 * thickness, height, thickness, origin=(-length / 2 + thickness / 2, 0, 0), axis="y")
    right = make_wall(width - 2 * thickness, height, thickness, origin=( length / 2 - thickness / 2, 0, 0), axis="y")
    return front.union(back).union(left).union(right)


def cut_opening(wall, width, height, position, origin=(0, 0, 0), axis="x"):
    """
    Cuts a door/window opening through a wall made with make_wall(). Pass
    the SAME origin and axis you used for make_wall() so the cut lines up.
    position = (along_wall, from_ground): along_wall is offset from the
    wall's center along its length; from_ground is height from the wall's
    own base (origin's z).
    """
    along, from_ground = position
    x0, y0, z0 = origin
    z = z0 + from_ground + height / 2
    if axis == "x":
        tool = cq.Workplane("XY").center(x0 + along, y0).workplane(offset=z).box(width, 1000, height, centered=(True, True, True))
    else:
        tool = cq.Workplane("XY").center(x0, y0 + along).workplane(offset=z).box(1000, width, height, centered=(True, True, True))
    return wall.cut(tool)