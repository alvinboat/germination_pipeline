# collection_pipeline

Per-capture processing for the five-day collection. One dish, one run.

## The loop

1. Save the scan into `raw_image/` (eBUS Player drops its `.bin` line files flat
   in there; no subfolder).
2. Set the four variables at the top of `config.py`.
3. `python3 process.py`

That writes the corrected cube, the raw archive and the preview, then empties
`raw_image/` so the next dish can go straight in. Re-imaging a dish overwrites
that dish's files in place.

`raw_image/` is cleared **last**, after every output is on disk. A run that fails
anywhere leaves the raw capture untouched and can simply be run again.

## config.py

```python
DAY = 1                       # 1, 2, 3, 4, ...
CAPTURE = "TRANSMITTANCE"     # or "REFLECTANCE"
DISH_NUMBER = 0               # 0-24
CAPTURE_SIDE = "DORSAL"       # or "VENTRAL"
```

All four are validated before anything is read. A typo in `CAPTURE_SIDE` would
otherwise write a dorsal scan into the ventral folder, and nothing downstream
could ever tell.

Everything below them in `config.py` is rig configuration, and every value of it
is echoed in the log and recorded in the capture's `meta.json`.

## What a run writes

For `DAY=1, CAPTURE=TRANSMITTANCE, DISH_NUMBER=1, CAPTURE_SIDE=DORSAL`:

```
transmittance_images/day1/dish1/dorsal/capture.npy         corrected cube, float32
                                       capture_masks.npz   saturation / white-validity
                                       capture_meta.json   full provenance
raw_binaries/transmittance_images/day1/dish1/dorsal/capture.bin    raw archive
                                                    capture.json   line index
preview/transmittance_day1_dish1_dorsal.png                preview
```

Reflectance is the same minus `capture_masks.npz` (nothing there is NaN, and
there is no band-drop or white-validity mask to carry).

`preview/` holds exactly one PNG per capture. The two optional diagnostics
(`SAVE_CHECKERBOARD_QC`, `SAVE_SATURATION_MAP`, both off) are written beside the
cube instead.

### capture.bin

One file per capture, not ~3,300. It is the camera's own Mono12Packed lines
concatenated in scan order and otherwise untouched — verified byte-for-byte
identical to the concatenation of the per-line files it replaces. `capture.json`
records the line count, the frame geometry and the original filenames (which
carry eBUS's per-line timestamps, the only thing concatenation would discard).

Read it back with:

```python
import common
cube = common.load_capture_bin("raw_binaries/.../capture.bin")   # (640, n_lines, 224)
```

## Calibration

```
transmittance_scripts/calibration/
    dark_sample/60k/      dark at the SAMPLE's exposure   -> sets t_long
    dark_white/2100/      dark at the WHITE's exposure
    white/2100/           the dedicated white line scan   -> sets t_short
    checkerboard/         board capture, same exposure and stage speed
reflectance_scripts/calibration/
    dark/dark_3k/         dark at the capture exposure
```

The `<exp>` folder name **is** the exposure: `60k` → 60000 µs, `2100` → 2100 µs.
Transmittance's exposure ratio comes from those names, so renaming a folder
changes the correction.

**Per-day overrides.** A day-specific folder wins over the shared set. Re-shot
the dark on day 3? Drop it in `calibration/day3/dark_sample/60k/` and it applies
to `DAY = 3` only, leaving the days already processed alone. The log says
`[shared]` or `[day-specific]` for every input on every run, with its file count
and newest mtime — a silently stale dark is the failure mode that is invisible in
the output and ruins a day of data.

**Caching.** The dark/white frames and the transmittance geometry factor are
cached under `calibration/.cache/`, keyed on the *contents* of the capture
directory that produced them (name, size, mtime of every file). Any change to a
calibration folder changes the key, so a stale frame cannot be reused silently.
Delete `.cache/` any time; it is only ever a speed-up.

The currently seeded calibration is copied from the dry-run rig. Replace it with
the real day-1 references before collecting.

## Reprocessing

None of these touch `raw_image/` or the archive:

```bash
python3 process.py --image raw_binaries/.../capture.bin --day 1 --dish 1 --side dorsal --capture transmittance
python3 process.py --image some_dir_of_bin/          # a capture directory
python3 process.py --keep-raw-image                  # normal run, but keep raw_image/ full
```

## Two things to watch

**The reflectance tape may be clipping.** On the dry-run capture this pipeline was
validated against, 27.9% of the teflon tape's voxels are railed at 4095 across
bands 19–130 (peak 76%). The tape *is* the reflectance white reference, so a
clipped tape pixel makes `(W − D)` read low and reflectance read **high** at those
bands, for every kernel in the frame. It is not recoverable after capture. Every
run reports tape clipping and scene clipping separately and warns above 1%; if it
fires on day 1, shorten the reflectance exposure before collecting the rest.

**Disk.** A capture is ~0.7 GB raw plus 0.5–0.6 GB corrected. At 25 dishes × 2
sides × 2 modes that is ~120–160 GB per day, against 438 GB free at the time of
writing — roughly three days. `process.py` estimates and warns before each run,
but the warning does not create space. Either point `RAW_BINARIES_ROOT` at another
mount or plan to move completed days off this disk.

## Kernel indexing — `gridfit/` + `grid_index.py`

Once the cubes exist, this gives every kernel a name that means the same physical
well on every capture, so a kernel can be followed across days.

```bash
python3 grid_index.py                       # everything under real_data
python3 grid_index.py --day 1 --dish 0      # one dish
```

A kernel is `dish<N>_R<row>C<col>`, 22 per dish. The plate is a rigid 4×7 lattice
with the four rim corners absent and two ArUco fiducials among the remaining 24
cells, so anchoring the lattice to the fiducials makes `R2C6` the same well
whatever the plate's rotation or where it landed on the stage.

**The plate is double-sided.** Cell (3,3) carries id0 on the dorsal face and id3
on the ventral; cell (2,6) carries id1 and id2. They are different codes, not
mirrored copies. Two things follow: a capture says out loud which face it is
(and `grid_index.py` checks that against the folder), and because the plate is
physically turned over, the ventral view of the lattice is a MIRROR of the
dorsal one — handled by letting the pose search reverse handedness, so `R2C6`
still names the same well from either side.

Output mirrors the data tree:

```
grid_view/<mode>_images/day<D>/dish<N>/<side>/grid.png    the overlay
                                              cells.json   28 cells + the fit
grid_view/contact_sheets/<mode>_day<D>_<side>.png         25 dishes at a glance
grid_view/summary.csv                                     one row per capture
grid_view/tracking.json                                   per dish, per kernel
grid_view/.cache/                                         renders; safe to delete
```

A fit is `pass`, `needs_review` or `fail`, and **`fail` means do not use the
cells**. A lattice off by one cell still draws a clean grid and silently
attributes every spectrum to the wrong kernel, so the fit is allowed to refuse.

`tracking.json` then checks the captures of a dish against each other: the
fiducial ids must name the same face every time, every dorsal capture in the
collection must come out the same handedness, the pitch must not move, and each
well's contents must match the same well in the dish's other captures better
than any relabelling of the plate would. That last one uses no fiducial, so it
is an independent check on the labelling rather than a restatement of it.

Cell geometry is in the render's frame; `gridfit.cells.cell_mask` converts a
cell to a `(width, n_lines)` mask that indexes the cube directly.

## Annotation previews — `preview_coco/generate.py`

Kernel masks now come from outside this repo: a SAM tool, annotated by hand,
exported as COCO. This script produces the images that gets annotated, one per
capture, and guarantees the one property that makes the export usable.

```bash
python3 preview_coco/generate.py                    # all 300 captures, ~6 s warm
python3 preview_coco/generate.py --mode reflectance  # one mode
python3 preview_coco/generate.py --day 1 --dish 0   # one dish, both modes
python3 preview_coco/generate.py --list             # what would be written
```

Output mirrors `real_data/`, with a globally unique basename inside each folder so
a flat drag-and-drop into an annotation tool still says which capture a COCO
`file_name` refers to:

```
preview_coco/<mode>_images/day<D>/dish<N>/<side>/<mode>_day<D>_dish<N>_<side>.png
preview_coco/manifest.csv       one row per PNG: mode/day/dish/side, size, cube path
```

**The frame contract.** A PNG is the capture's own array indexing, untouched:

```
png[i, j]  <->  cube[i, j, :]        png.shape == cube.shape[:2] == (640, n_lines)
```

so a COCO mask decoded off the PNG is a boolean array that indexes the cube with
no rotation, flip, transpose or rescale:

```python
m = pycocotools.mask.decode(ann["segmentation"]).astype(bool)   # (640, n_lines)
spectra = cube[m]                                              # (n_masked, 224)
```

This is *not* gridfit's frame. `gridfit/render.py` rotates every plane 90° clockwise
so the dish reads upright and cv2's `(x, y)` is `(column, row)`; every overlay, cell
corner and lattice frame under `grid_view/` lives there. The preview rotates that
back, because a mask drawn on it has to index the cube and nothing else.
`--frame working` writes the upright version for eyeballing — do not annotate those,
their masks are transposed with respect to `capture.npy`.

The image is the render gridfit detects on, minus the overlay: band-mean,
percentile-stretched, uint8 grey, 640 rows by 921–1361 columns depending on the
capture's line count. No grid, no cell index, nothing drawn on it.

**The contract is measured, not asserted.** `verify()` runs on the first capture of
each mode of every run, and a failure refuses to write:

1. Rebuilds the preview from the cube *in cube layout* — its own stretch, no call to
   any frame helper — and requires it bit-identical. A rotation, transpose, flip,
   sub-pixel shift, rescale or wrong-capture mixup all fail this.
2. Takes `grid_view`'s 22 kernel cell masks, which reach cube layout through
   gridfit's working-frame geometry rather than through the cube, and requires them
   to land on wells rather than plate. Reflectance separates 22/22 by 90–171 grey
   levels; transmittance passes 16/22 because the open aperture rails at 255 and the
   opaque moulding floors at 0, so six cells carry no 8-bit contrast either way.
   Hence the loose 60% bar — check 1 is the fine-grained proof.

Check 1 was worth writing twice over: it first failed on a single pixel of one
capture, which turned out to be `float64` vs `float32` accumulation of the band
mean landing on a grey-level boundary, not a frame error.

## Layout

```
config.py                 the four variables + rig configuration
process.py                the runner: one capture, start to finish
common.py                 stitching, raw archiving, calibration cache, geometry
loadstich/                the Mono12Packed codec (do not reimplement the shifts)
transmittance_scripts/    correct.py + its calibration set
reflectance_scripts/      correct.py + its calibration set
gridfit/                  plate model, well/fiducial detection, lattice fit
grid_index.py             fit every capture, write grid_view/
preview_coco/             generate.py + the annotation PNGs, in cube layout
extract_kernels.py        cut every kernel into one array a DataLoader can index
raw_image/                where you save the scan
preview/                  one PNG per capture
transmittance_images/     corrected cubes, day/dish/side
reflectance_images/       corrected cubes, day/dish/side
raw_binaries/             capture.bin archive, mirroring the same day/dish/side tree
grid_view/                overlays, per-cell records and the tracking report
```

## Relationship to dryrun_scripts

The corrections are the dry-run scripts', verified equivalent rather than
rewritten:

* reflectance is **bit-identical** to `dryrun_scripts/reflectance_image/process_image.py`
  on the same capture (kernel-grid stage 8 dropped, as requested).
* transmittance is **bit-identical** to `dryrun_scripts/transmission_imagev2/process.py`
  — cube and all five masks — when the scan-axis factor is held fixed.

Three things changed on purpose:

1. **Stitching reads each line once** and writes `capture.bin` in the same pass,
   on a thread pool: 9.1 s → 2.1 s on a real 3,276-line capture, byte-identical
   cube. The dry-run scripts read the raw bytes to stitch and then walked the
   directory again to move ~3,300 files.
2. **Calibration frames and the transmittance geometry factor are cached**, so a
   dark is collapsed once per calibration set instead of once per dish.
3. **One geometry implementation for both modes.** The dry-run transmission script
   measured the board with mean corner-to-corner step lengths — the column norms
   of the lattice map, which are not rotation-invariant. The reflectance script
   fits the lattice and takes row norms, which are. On the transmission board
   (rotated −2.3°) the two disagree by 0.6%: the old factor was 0.6729, the
   row-norm fit gives 0.6769, and the new closed-loop re-measurement confirms the
   latter to +0.03% residual anisotropy. The old code had no such check.
