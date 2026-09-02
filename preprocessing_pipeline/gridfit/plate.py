"""The rigid plate: what cells exist, where the fiducials are, and what makes a
cell name mean the same physical well on every capture.

The kernel insert is a fixed 4x7 lattice sitting in a round petri dish. The four
corner cells fall outside the rim and are never cut, leaving 24 cells; two of
those 24 hold an ArUco fiducial instead of a kernel, leaving 22 kernel wells.

    C:      0     1     2     3     4     5     6
    R0:    rim  kernel kernel kernel kernel kernel rim
    R1:  kernel kernel kernel kernel kernel kernel kernel
    R2:  kernel kernel kernel kernel kernel kernel MARKER
    R3:    rim  kernel kernel MARKER kernel kernel rim

THE PLATE IS DOUBLE-SIDED. Each fiducial cell carries a different ArUco code on
each face, so a capture says out loud which face it is:

    cell (2,6):  id1 seen from the DORSAL face,  id2 from the VENTRAL face
    cell (3,3):  id0 seen from the DORSAL face,  id3 from the VENTRAL face

All four are DICT_4X4_50 and all four read in normal (un-mirrored) polarity from
their own side -- verified by decoding the 4x4 bit grid off real day-1 dish-0
captures in both modes: dorsal gave an exact bit-for-bit match to ids 1 and 0,
ventral to ids 2 and 3. They are NOT mirrored copies of one another, so a
detector that only knows ids 0/1 finds nothing it trusts on a ventral capture.

Because the plate is physically turned over between the two captures, the
ventral view of the lattice is a MIRROR of the dorsal view. Naming a well by
(row, col) anchored to the fiducial cells makes that mirror invisible:
(2,6) is the same physical well from either side, so R2C6 on the ventral capture
and R2C6 on the dorsal capture are the same kernel. The lattice basis simply
comes out left-handed on one side and right-handed on the other, which is
recorded (`chirality`) and is itself a check -- every dorsal capture of the
collection must agree with every other one.

Coordinates everywhere in gridfit are the WORKING FRAME: x = column, y = row of
the (n_lines, width) uint8 render that `render.py` produces, which is the same
orientation as the preview PNGs the pipeline already writes. `render.to_cube_xy`
converts back when a mask has to index the cube.
"""

N_ROWS, N_COLS = 4, 7

# Clipped by the dish rim: no well is moulded there at all.
RIM_CORNERS = frozenset({(0, 0), (0, 6), (3, 0), (3, 6)})

# ArUco id -> the plate cell it sits in. Both faces are listed; which pair is
# expected follows from the side, but detection never assumes it -- reading ids
# 2/3 on a folder labelled dorsal is a finding, not something to correct away.
MARKER_CELL = {0: (3, 3), 1: (2, 6), 2: (2, 6), 3: (3, 3)}
SIDE_IDS = {"dorsal": frozenset({0, 1}), "ventral": frozenset({2, 3})}
ID_SIDE = {i: s for s, ids in SIDE_IDS.items() for i in ids}
ARUCO_DICT_NAME = "DICT_4X4_50"

MARKER_CELLS = frozenset(MARKER_CELL.values())          # {(3,3), (2,6)}

# Nominal geometry, measured over the real collection (day 1/2/9, both modes,
# both sides): pitch_x 73.1-74.4 px, pitch_y 133.3-134.8 px, ratio 1.80-1.83.
# Used only to bound the search and to sanity-gate the result -- the fit
# measures the real pitch per capture.
PITCH_X_NOM, PITCH_Y_NOM = 74.0, 134.0
PITCH_X_RANGE = (60.0, 90.0)
PITCH_Y_RANGE = (110.0, 160.0)
PITCH_RATIO_RANGE = (1.60, 2.05)


def all_cells():
    """The 24 moulded cells, row-major. Rim corners are not cells at all."""
    return [(r, c) for r in range(N_ROWS) for c in range(N_COLS)
            if (r, c) not in RIM_CORNERS]


def kernel_cells():
    """The 22 wells that hold a kernel, row-major -- index i is KERNEL_CELLS[i].

    Row-major over the fixed plate model, so index i names the same physical
    well on every capture whatever the plate's rotation, which face is up, or
    where it landed on the stage. That is the whole point of anchoring to the
    fiducials.
    """
    return [rc for rc in all_cells() if rc not in MARKER_CELLS]


KERNEL_CELLS = tuple(kernel_cells())
CELL_INDEX = {rc: i for i, rc in enumerate(KERNEL_CELLS)}
N_KERNEL_CELLS = len(KERNEL_CELLS)                      # 22


def cell_name(r, c):
    """The canonical name for a plate cell. Stable across days, sides, modes."""
    return f"R{r}C{c}"


def cell_kind(r, c):
    if (r, c) in RIM_CORNERS:
        return "rim"
    if (r, c) in MARKER_CELLS:
        return "marker"
    return "kernel"


# --------------------------------------------------------------------- poses --
# A detected lattice arrives with arbitrary axis directions: the basis estimator
# picks whichever way the nearest-neighbour vectors happened to point. Four
# relabelings map that lattice onto the plate model without changing which
# physical node is which -- flip the column axis, the row axis, both, or
# neither. (90-degree rotations are excluded: they would turn a 4x7 into a 7x4.)
#
# Two of the four reverse handedness, which is exactly what a ventral capture
# needs, so nothing special has to be done for the flipped face -- the same
# search covers it and reports which one it used.
POSES = (
    ("identity", False, False),
    ("flip_col", False, True),
    ("flip_row", True, False),
    ("rot180", True, True),
)


def apply_pose(pose, j, i):
    """Lattice index (j along the long-pitch axis, i along the short) -> (row, col)."""
    _, flip_r, flip_c = pose
    r = (N_ROWS - 1 - j) if flip_r else j
    c = (N_COLS - 1 - i) if flip_c else i
    return r, c
