"""Build the fused platform with a carved rectangular socket.

The platform dimensions are explicit assumptions based on Platform.jpg.
All dimensions are millimetres.
"""

from pathlib import Path

import cadquery as cq


# Main body assumption
MAIN_LENGTH = 400.0
MAIN_WIDTH = 300.0
MAIN_HEIGHT = 60.0

# Lower offset assumption; it overlaps the main body so the result is one solid.
LOWER_LENGTH = 240.0
LOWER_WIDTH = 160.0
LOWER_HEIGHT = 30.0
LOWER_X = (MAIN_LENGTH - LOWER_LENGTH) / 2
LOWER_Y = -80.0
LOWER_Z = -LOWER_HEIGHT

# Socket and matching insert assumption
SOCKET_LENGTH = 140.0
SOCKET_WIDTH = 100.0
SOCKET_DEPTH = 20.0
SOCKET_X = (MAIN_LENGTH - SOCKET_LENGTH) / 2
SOCKET_Y = (MAIN_WIDTH - SOCKET_WIDTH) / 2
INSERT_CLEARANCE = 2.0
INSERT_LENGTH = SOCKET_LENGTH - 2 * INSERT_CLEARANCE
INSERT_WIDTH = SOCKET_WIDTH - 2 * INSERT_CLEARANCE
INSERT_HEIGHT = SOCKET_DEPTH - 2.0


def build_platform() -> cq.Workplane:
    """Fuse the main body and lower offset, then carve the top socket."""
    main_body = cq.Workplane("XY").box(
        MAIN_LENGTH, MAIN_WIDTH, MAIN_HEIGHT,
        centered=(False, False, False),
    )

    lower_offset = (
        cq.Workplane("XY")
        .box(LOWER_LENGTH, LOWER_WIDTH, LOWER_HEIGHT,
             centered=(False, False, False))
        .translate((LOWER_X, LOWER_Y, LOWER_Z))
    )

    fused_platform = main_body.union(lower_offset)

    socket = (
        cq.Workplane("XY")
        .box(SOCKET_LENGTH, SOCKET_WIDTH, SOCKET_DEPTH,
             centered=(False, False, False))
        .translate((SOCKET_X, SOCKET_Y, MAIN_HEIGHT - SOCKET_DEPTH))
    )

    return fused_platform.cut(socket)


def build_insert_box() -> cq.Workplane:
    """Create the separate insert that fits inside the platform socket."""
    return (
        cq.Workplane("XY")
        .box(INSERT_LENGTH, INSERT_WIDTH, INSERT_HEIGHT,
             centered=(False, False, False))
        .translate((
            SOCKET_X + INSERT_CLEARANCE,
            SOCKET_Y + INSERT_CLEARANCE,
            MAIN_HEIGHT - SOCKET_DEPTH + 1.0,
        ))
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    platform = build_platform()
    insert = build_insert_box()

    platform.export(str(output_dir / "02_platform.step"))
    platform.export(str(output_dir / "02_platform.stl"))
    insert.export(str(output_dir / "02_insert_box.step"))

    assembly = cq.Assembly(name="platform_with_insert")
    assembly.add(platform, name="fused_platform", color=cq.Color("slategray"))
    assembly.add(insert, name="insert_box", color=cq.Color("steelblue"))
    assembly.save(str(output_dir / "02_platform_assembly.step"))

    print("Created platform, insert, and assembly STEP/STL files")


if __name__ == "__main__":
    main()

