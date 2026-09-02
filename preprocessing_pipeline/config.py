"""Scan information for the capture that is currently sitting in raw_image/.

Edit the four variables at the top, then run:

    python3 process.py

Everything below them is rig configuration that should change rarely, and
every value of it is echoed in the run log and recorded in the capture's
meta.json, so a cube can always be traced back to the settings that made it.
"""

# ===========================================================================
# THE FOUR PER-CAPTURE VARIABLES -- set these before every run.
# ===========================================================================

DAY = 2                       # scan day: 1, 2, 3, 4, ...
CAPTURE = "TRANSMITTANCE"     # "TRANSMITTANCE" or "REFLECTANCE"
DISH_NUMBER = 0               # dish index, 0-24
CAPTURE_SIDE = "DORSAL"       # "DORSAL" or "VENTRAL"

# ===========================================================================
# Everything below: rig configuration.
# ===========================================================================

# --------------------------------------------------------------- layout --
# Where the raw archive goes. Point this at another mount to keep ~700 MB of
# raw binaries per capture off the disk holding the corrected cubes -- at
# ~100 captures a day the two together are the binding constraint, and
# process.py warns before a run it cannot fit.
RAW_BINARIES_ROOT = "raw_binaries"

DISH_RANGE = (0, 24)          # inclusive; a typo outside this stops the run

# --------------------------------------------------------------- runtime --
# Threads used to read + unpack the ~3,300 line files of a capture. Measured on
# this box: 1 -> 9.1s, 4 -> 2.1s, 8 -> 2.2s, 16 -> 2.9s, all byte-identical.
# Past 4 the unpack is memory-bandwidth bound and more threads only add churn.
STITCH_WORKERS = 4

# Refuse-to-be-surprised margin, in GB, kept free after a capture's own worst
# case. A warning, not a hard stop: the run still completes.
MIN_FREE_GB = 20

# Optional diagnostics, both off. preview/ holds exactly one PNG per capture --
# <mode>_day<D>_dish<N>_<side>.png -- and nothing else. When either of these is
# turned on it is written beside that capture's cube instead, never into
# preview/.
#
#   SAVE_CHECKERBOARD_QC   which checkerboards were detected, accepted and
#                          rejected. The one thing that distinguishes a geometry
#                          factor measured off the real board from one measured
#                          off the well grid; the accept/reject decisions are in
#                          the log regardless. Reflectance writes
#                          capture_checkerboard_qc.png beside each cube, since
#                          its boards are in the scene and every capture carries
#                          its own measurement. Transmittance measures a shared
#                          board capture once per calibration set, so its overlay
#                          goes into calibration/.cache/ alongside the cached
#                          factor rather than being copied per dish.
#   SAVE_SATURATION_MAP    capture_saturation.png (transmittance only): per-pixel
#                          fraction of bands that clipped. Nothing is lost with
#                          it off -- the full per-voxel mask is in
#                          capture_masks.npz and the summary is in meta.json.
SAVE_CHECKERBOARD_QC = False
SAVE_SATURATION_MAP = False

# ================================ TRANSMITTANCE ============================
# T = (raw - D_long) / (W - D_short) * (t_short / t_long)
#
# Calibration inputs, resolved under transmittance_scripts/calibration/:
#     dark_sample/<exp>/     dark at the SAMPLE's exposure   -> sets t_long
#     dark_white/<exp>/      dark at the WHITE's exposure
#     white/<exp>/           the dedicated white line scan   -> sets t_short
#     checkerboard/          board capture, same exposure and stage speed
# The <exp> folder name is the exposure: "60k" -> 60000 us, "2100" -> 2100 us.
# A day-specific override wins over the shared set: drop a re-shot dark into
#     transmittance_scripts/calibration/day3/dark_sample/60k/
# and it applies to DAY = 3 only.

# Bands where the lamp is below this fraction of its peak are dropped to NaN.
# The single most important knob for whether a spectrum looks sane. The lamp
# only usefully illuminates ~930-1620 nm; at 1700 nm it delivers 3% of peak and
# at 900 nm about 7%. Transmittance divides by the lamp, so out there the
# correction divides a near-constant floor by a number heading for zero and
# every spectrum turns up sharply at both ends -- arithmetic, not signal.
# 0.50 keeps bands 8-200 (929-1617 nm) on this rig, which took the reported
# spread over a kernel region from 11.9x (at 0.10) to 6.2x of real structure.
TRANS_MIN_BAND_FRAC = 0.5

# Exposure overrides in milliseconds. None = read from the calibration folder
# names, which is the safer default: a mismatch between the folder name and the
# real exposure then shows up as an open-beam check that fails, rather than
# being silently overridden here.
TRANS_EXPOSURE_LONG_MS = None      # sample exposure  (None -> dark_sample/<exp>)
TRANS_EXPOSURE_SHORT_MS = None     # white exposure   (None -> white/<exp>)

TRANS_CHECKERBOARD_INNER = (19, 19)   # inner corners of the 20x20-square target
TRANS_N_BOARDS = 1                    # one dedicated board in the frame
TRANS_CELL_SIZE = (1.0, 1.0)          # true cell extent; only the ratio matters
# 1.0 for this rig: the raw board measurement already brings the dish visibly
# round in the corrected preview, so there is nothing left for a plane factor to
# absorb. That is an eyeball result, not a measured constant -- nothing here
# measures the sample plane. Reflectance's 1.091 belongs to its own dish/insert
# stack and must not be carried over.
TRANS_PLANE_FACTOR = 1.0
# Fixed scan-axis factor. None = measure it from the checkerboard capture (the
# result is cached, so it costs one detection per calibration set, not one per
# dish). Set a number to skip detection entirely.
TRANS_SCAN_SCALE = None

# ================================= REFLECTANCE =============================
# reflectance = (raw - D) / white, with the white taken from the two teflon tape
# strips in the frame -- there is no dedicated white capture in this mode.
#
# Calibration inputs, resolved under reflectance_scripts/calibration/:
#     dark/<exp>/            dark at the capture exposure
# Same day-specific override rule as transmittance.

REFL_TAPE_PCT = 85.0          # brightness percentile a tape blob must exceed
REFL_WHITE_PCT = 75.0         # percentile of a row's tape pixels used as its white
# A width row's tape sample is trusted only if it holds at least this fraction
# of the median per-row count. Rows below it lie in a blob's taper, where the
# few surviving pixels are part background and read dim; trusting them sent
# reflectance to 1.33 on a scene the rest of the frame put at 1.00. Those rows
# are interpolated from the interior instead. 0 trusts every covered row.
REFL_WHITE_MIN_COVERAGE = 0.75

REFL_CHECKERBOARD_INNER = (3, 3)   # inner corners of the 4x4-square boards
REFL_N_BOARDS = 2                  # two boards in the frame, measured and pooled
REFL_CELL_SIZE = (1.0, 1.0)
# Measured, not assumed: pooling the kernel well-lattice pitch over the dry-run
# captures in BOTH plate orientations cancels the plate's own pitch ratio and
# leaves a residual scan/spatial scale of 0.9165, i.e. this factor. It
# cross-checks against the dish rim (which independently implies 1.096) to 0.5%,
# and is fixed rather than per-capture -- unchanged across a 50% conveyor-speed
# difference and a 90deg plate rotation. Applying it brings the dish rim from
# 0.888-0.901 to 0.990-0.998 of round on all five dry-run captures.
# Re-derive it if the camera height, the dish/insert stack, or the targets
# change. 1.0 disables it and gives the raw board measurement.
REFL_PLANE_FACTOR = 1.091
REFL_SCAN_SCALE = None        # None = measure from the in-scene boards per capture
