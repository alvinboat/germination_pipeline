"""Paths, scope and the constants every stage shares.

Everything that decides what the dataset *is* lives here, so `build_meta.json`
can record it and a later run can prove it used the same settings.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

# ------------------------------------------------------------------- inputs --
COCO_JSON = ROOT / "annotations" / "instances_default.json"
PREVIEW_DIR = ROOT / "annotations" / "preview_coco"
PREPROCESSING = REPO / "preprocessing_pipeline"   # the gridfit package lives here
GRID_VIEW = PREPROCESSING / "grid_view"           # its fitted lattice, one dir per capture
# The annotator's spreadsheet is the source of truth. A .csv with
# kernel_uid,first_germinated_day columns is also accepted -- see
# barley.germination.read_germination_labels. Overridable by env var so a
# fixture can drive the trainers without them growing a --labels flag:
#   BARLEY_GERMINATION_LABELS=some.csv .venv/bin/python train_germination.py
GERMINATION_LABELS = Path(os.environ.get("BARLEY_GERMINATION_LABELS") or
                          ROOT / "germination_label_annotator.xlsx")
if not GERMINATION_LABELS.is_absolute():
    GERMINATION_LABELS = ROOT / GERMINATION_LABELS

# The measured clock: per-dish t0, every capture's real hour, every scoring
# visit. Acquired once from the acquisition machine's file mtimes and the
# scoring photos' EXIF; the tooling that did it has been removed. This is a
# hand-acquired INPUT like the label sheet above, not a build product -- it
# lives here and is versioned, not in DATASET/ which is regenerable.
TIMING_JSON = ROOT / "timing.json"

# ------------------------------------------------------------------ outputs --
DATASET = ROOT / "dataset"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
# Exploratory figures that describe the data rather than a model run: one
# subfolder per question (germination/, pca/, ...), each self-contained with its
# own summary.md so a folder can be read, or deleted, on its own.
EXPLORATION = ROOT / "exploration"

PATCHES_NPY = DATASET / "patches.npy"
MASKS_NPY = DATASET / "masks.npy"
SPECTRA_NPY = DATASET / "spectra.npy"
INDEX_CSV = DATASET / "index.csv"
BANDS_JSON = DATASET / "bands.json"
SPLITS_JSON = DATASET / "splits.json"

# -------------------------------------------------------------------- scope --
# The capture-rig folder names are slot names, not timestamps: day1 is 0 h,
# day9 is 8 h, day2 is 24 h. day2 is out of scope for now, and it is also the
# only day with unannotated captures. Nothing downstream may sort on the folder
# name -- the index carries `hours`.
HOURS = {1: 0.0, 9: 8.0}
MODES = ("reflectance", "transmittance")
SIDES = ("dorsal", "ventral")

N_DISHES = 25
DISHES_PER_VARIETY = 5
N_VARIETIES = N_DISHES // DISHES_PER_VARIETY

# Five consecutive dishes per variety, in dish order. These are the real names;
# `variety_of()` returns the 1-based number and index.csv stores that integer, so
# renaming here is enough -- no dataset rebuild.
#
# Two pairs share a cultivar name. They are four distinct varieties, not
# replicates, but prospect1/prospect2 and laureate1/laureate2 being related is
# the reason to expect them to sit close together spectrally: the laureate pair
# accounts for most of the confusion in every run so far.
VARIETY_NAMES = ("prospect1", "prospect2", "laureate1", "laureate2", "unknown")
assert len(VARIETY_NAMES) == N_VARIETIES

# ----------------------------------------------------------- dud varieties --
# prospect2 (dishes 5-9) did not germinate at all: 110 of 110 scored kernels are
# -1. It is dropped from the modelling path -- `barley.index.Index.load` removes
# its rows, so no trainer, split, task or figure has to remember to.
#
# Dropped rather than kept as a class, because a variety with one outcome cannot
# teach anything about germination and actively corrupts what it is asked. It is
# 20% of the data sitting entirely on one side of every germination target, so a
# model that learns nothing but "is this prospect2?" scores well on germination
# without predicting germination; and since variety is confounded with dish and
# folds hold out whole dishes, that shortcut survives cross-validation.
#
# Nothing is deleted. The rows stay in patches.npy and index.csv and still show
# up in the mask review; set this to () to bring them back, with no rebuild.
EXCLUDED_VARIETIES = (2,)

# Wells with no kernel in them, in every view of every capture. They are not
# annotation misses: the same well reads empty from both faces of the plate, on
# both days, in both modes. N is 548 kernels, not 550.
KNOWN_EMPTY_WELLS = ("dish13_R0C4", "dish21_R3C1")

# Kernels the annotator missed in exactly one view, so they have 7 rather than 8.
# Flagged for the mask fact-check; listed here so verify_dataset can tell a known
# gap from a new one and still fail on anything that is not on this list.
#   dish13_R3C2  reflectance   day9 ventral
#   dish3_R1C2   transmittance day9 dorsal
#   dish8_R2C0   transmittance day9 dorsal
KNOWN_INCOMPLETE_KERNELS = ("dish3_R1C2", "dish8_R2C0", "dish13_R3C2")

# -------------------------------------------------------------- germination --
# A THIRD clock, and the one most likely to be misread. There are two others:
#   HOURS               the HSI capture clock. day1 = 0 h, shot BEFORE the seeds
#                       were wetted; day9 = 8 h; day2 = 24 h.
#   labels/germination_photos/day1..day5   calendar days of the scoring assay.
# All three count from the same origin: the moment water was added.
#
# The annotator's convention, which is the one that governs:
#   label 1 -> germinated by day 1 =  24 h
#   label 2 ->                day 2 =  48 h    ... and so on
#   label 5 ->                day 5 = 120 h
#   label -1 -> never germinated: right-censored at 120 h, NOT a missing value
#   blank    -> not scored yet: a missing label, dropped, and never read as -1
#
# Join on `hours` or `kernel_uid`; never on anything called `day`.
GERMINATION_DAYS = (1, 2, 3, 4, 5)
GERMINATION_NEVER = -1


def germination_hours():
    """Assay scoring day -> hours since moisture."""
    return tuple(d * 24.0 for d in GERMINATION_DAYS)


# Everything not germinated by the last scoring day is right-censored here. It
# is not a germination time and must never be used as one.
GERMINATION_CENSOR_H = germination_hours()[-1]      # 120.0

# ------------------------------------------------------------ preprocessing --
BAND_LO, BAND_HI = 8, 200       # [lo, hi): the range valid in every capture
WL_LO, WL_HI, N_BANDS_RAW = 900.0, 1700.0, 224
N_BANDS = BAND_HI - BAND_LO     # 192

PATCH_H, PATCH_W = 128, 64      # rows follow the plate's row axis

# Scale applied to the cell quad before warping it onto the patch. 1.0 is the
# cell exactly as gridfit drew it; >1 pads outwards.
#
# An earlier extractor inset by 0.15 instead, because it sampled well brightness
# and had to keep the bright wall out. Here the annotator's mask already decides
# what is kernel, so an inset buys nothing and costs real signal: the kernels sit
# about 0.27 cell-widths off-centre in their wells (the retaining clip holds them
# to one side), so a 15% inset slices straight through the grain. The pad
# guarantees the whole mask lands inside the patch -- verify_dataset checks that
# no mask reaches the patch border.
CELL_SCALE = 1.10

# Erosion of the mask boundary, in annotation pixels, applied in cube layout
# before the warp (barley.assign, which treats any non-zero value as on).
# Intended to keep a clipped sliver of the plastic retaining clip out of the
# kernel's spectrum.
#
# 2 px was rejected: it cost a median 23.5% of mask area (worst 33.8%), too much
# of the grain to pay for a contaminant that the masks may not actually contain.
# The per-pixel PCA has since measured that contaminant -- in reflectance it is
# ~half of all voxel variance -- so the trade is worth revisiting for that mode.
#
# UNRESOLVED: this value is 1, but PROJECT_REPORT.md and both READMEs describe
# erosion as "off". Which one the published dataset was built with is recorded
# in `dataset/build_meta.json` ("mask_erode_px"), and that file is the authority
# -- `verify_dataset.py` prints it on every run. Reconcile before quoting either
# the erosion setting or a number that depends on it.
MASK_ERODE_PX = 1

# Pseudo-absorbance floor. Transmittance reaches -0.0005 on real captures and a
# reflectance voxel can be exactly 0 after dark subtraction; both make log10
# undefined. 1e-4 caps absorbance at 4.0, well above the observed range.
ABSORBANCE_EPS = 1e-4

SEED = 20260817
N_FOLDS = 5


def wavelengths():
    import numpy as np
    return np.linspace(WL_LO, WL_HI, N_BANDS_RAW)[BAND_LO:BAND_HI]


# Anchor for the raw capture tree. gridfit's cells.json records the absolute
# path of the cube it was fitted on, from whichever machine ran the fit, so a
# checkout that has moved -- different user, different disk -- would otherwise
# be pointed at a directory that does not exist. Everything is re-anchored under
# REPO instead. Override with BARLEY_CAPTURE_ROOT if real_data lives elsewhere.
CAPTURE_ROOT = Path(os.environ.get("BARLEY_CAPTURE_ROOT") or REPO)
CAPTURE_DIRNAMES = ("real_data",)


def local_capture_dir(recorded):
    """A path recorded by gridfit -> the same capture in THIS checkout.

    Keeps everything from the `real_data/` component onwards and re-roots it at
    CAPTURE_ROOT. Falls through unchanged if the path has no recognisable
    anchor, so an unusual layout fails loudly at the open() rather than quietly
    reading the wrong cube.
    """
    p = Path(recorded)
    for i, part in enumerate(p.parts):
        if part in CAPTURE_DIRNAMES:
            return CAPTURE_ROOT.joinpath(*p.parts[i:])
    return p


def variety_of(dish):
    """Dish number -> 1-based variety label."""
    return int(dish) // DISHES_PER_VARIETY + 1


def kept_varieties():
    """1-based variety numbers the modelling path uses, in order."""
    return tuple(v for v in range(1, N_VARIETIES + 1) if v not in EXCLUDED_VARIETIES)


def kept_dishes():
    """Dish numbers the modelling path uses, in order."""
    return tuple(d for d in range(N_DISHES) if variety_of(d) not in EXCLUDED_VARIETIES)


def excluded_names():
    return tuple(VARIETY_NAMES[v - 1] for v in EXCLUDED_VARIETIES)
