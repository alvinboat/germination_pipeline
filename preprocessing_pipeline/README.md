# preprocessing_pipeline

Raw camera lines in, corrected cubes and a fitted plate lattice out. One dish,
one run.

```
config.py               the four per-capture variables + rig configuration
process.py              the runner: one capture, start to finish
common.py               stitching, raw archiving, calibration cache, geometry
loadstich/              the Mono12Packed codec (do not reimplement the shifts)
reflectance_scripts/    correct.py + its calibration set
transmittance_scripts/  correct.py + its calibration set
gridfit/                plate model, well/fiducial detection, lattice fit
grid_index.py           fit every capture -> grid_view/
```

## The loop

1. Save the scan into `raw_image/` (eBUS Player drops its `.bin` line files flat
   in there; no subfolder).
2. Set the four variables at the top of `config.py` — `DAY`, `CAPTURE`,
   `DISH_NUMBER`, `CAPTURE_SIDE`. All four are validated before anything is
   read: a typo in `CAPTURE_SIDE` would file a dorsal scan as ventral and
   nothing downstream could ever tell.
3. `python3 process.py`

That writes the corrected cube, the raw archive and the preview, then empties
`raw_image/` so the next dish can go straight in. `raw_image/` is cleared
**last**, after every output is on disk, so a run that fails anywhere leaves the
capture intact and can simply be repeated.

```
<mode>_images/day<D>/dish<N>/<side>/capture.npy        corrected cube, float32
                                    capture_masks.npz  saturation / white-validity
                                    capture_meta.json  full provenance
raw_binaries/<mode>_images/day<D>/dish<N>/<side>/capture.bin   raw archive
                                                     capture.json  line index
preview/<mode>_day<D>_dish<N>_<side>.png              one PNG per capture
```

Reflectance is the same minus `capture_masks.npz`. Everything below the four
variables in `config.py` is rig configuration, and every value of it is echoed
in the log and recorded in `capture_meta.json`, so a cube traces back to the
settings that made it.

`capture.bin` is one file per capture rather than ~3,300: the camera's own
Mono12Packed lines concatenated in scan order, verified byte-for-byte identical
to the per-line files it replaces. `capture.json` keeps the line count, the frame
geometry and the original filenames, which carry eBUS's per-line timestamps.
Read it back with `common.load_capture_bin(path)`.

### Reprocessing

None of these touch `raw_image/` or the archive:

```bash
python3 process.py --image raw_binaries/.../capture.bin --day 1 --dish 1 --side dorsal --capture transmittance
python3 process.py --image some_dir_of_bin/   # a capture directory
python3 process.py --keep-raw-image           # normal run, but keep raw_image/ full
```

## Correction

Two modes, two corrections, sharing a calibration-resolution and caching layer.
Both are per (spatial pixel, band): every sensor pixel gets its own dark offset
and its own gain. The physics and the design decisions are in
`PROJECT_REPORT.md` §5; the operational parts are here.

```
transmittance_scripts/calibration/
    dark_sample/60k/    dark at the SAMPLE's exposure   -> sets t_long
    dark_white/2100/    dark at the WHITE's exposure
    white/2100/         the dedicated white line scan   -> sets t_short
    checkerboard/       board capture, same exposure and stage speed
reflectance_scripts/calibration/
    dark/dark_3k/       dark at the capture exposure
```

The `<exp>` folder name **is** the exposure: `60k` → 60000 µs, `2100` → 2100 µs.
Transmittance's exposure ratio comes from those names, so renaming a folder
changes the correction. If either exposure cannot be determined the run stops
rather than assuming 1.0.

**Per-day overrides.** A day-specific folder wins over the shared set: drop a
re-shot dark into `calibration/day3/dark_sample/60k/` and it applies to `DAY = 3`
only. The log says `[shared]` or `[day-specific]` for every input on every run,
with its file count and newest mtime — a silently stale dark is invisible in the
output and ruins a day of data.

**Caching.** Dark/white frames and the transmittance geometry factor are cached
under `calibration/.cache/`, keyed on the *contents* of the directory that
produced them (name, size, mtime of every file). Any change invalidates the key,
so a stale frame cannot be reused silently. Delete `.cache/` any time.

### Two things to watch

**The reflectance tape clips.** The teflon tape *is* the reflectance white
reference, and on the validated capture 27.9% of its voxels are railed at 4095
across bands 19–130 (peak 76%). A clipped tape pixel makes `(W − D)` read low and
therefore makes **reflectance read high** at those bands, for every kernel in the
frame. Not recoverable after capture — it needs a shorter exposure at collection
time. Every run reports it and warns above 1%. It fired, and the data was
collected anyway.

**Disk.** A capture is ~0.7 GB raw plus 0.5–0.6 GB corrected: ~120–160 GB per day
at 25 dishes × 2 sides × 2 modes. `process.py` estimates and warns before each
run, but the warning does not create space. Point `RAW_BINARIES_ROOT` at another
mount or move completed days off the disk.

## Kernel indexing — `gridfit/` + `grid_index.py`

Once the cubes exist, this gives every kernel a name that means the same physical
well on every capture, so a kernel can be followed across days and faces.

```bash
python3 grid_index.py                    # everything under ../real_data
python3 grid_index.py --day 1 --dish 0   # one dish
```

A kernel is `dish<N>_R<row>C<col>`, 22 per dish. The plate is a rigid 4×7 lattice
with the four rim corners absent and two ArUco fiducials among the remaining 24
cells, so anchoring the lattice to the fiducials makes `R2C6` the same well
whatever the plate's rotation or where it landed on the stage.

**The plate is double-sided.** Cell (3,3) carries id0 on the dorsal face and id3
on the ventral; cell (2,6) carries id1 and id2. Different codes, not mirrored
copies — so a capture states which face it is, and `grid_index.py` checks that
against the folder it was filed in. Because the plate is physically turned over,
the ventral view of the lattice is a **mirror** of the dorsal one, handled by
letting the pose search reverse handedness.

```
grid_view/<mode>_images/day<D>/dish<N>/<side>/grid.png   the overlay
                                              cells.json  28 cells + the fit
grid_view/contact_sheets/<mode>_day<D>_<side>.png        25 dishes at a glance
grid_view/summary.csv                                    one row per capture
grid_view/tracking.json                                  per dish, per kernel
grid_view/.cache/                                        renders; safe to delete
```

**A fit may refuse.** Each capture's fit is `pass`, `needs_review` or `fail`, and
`fail` means the cells must not be used — a lattice off by one cell still draws a
clean grid while attributing every spectrum to the wrong kernel. All 300 captures
in this collection fit `pass`.

`tracking.json` then checks each dish's captures against one another: fiducial
ids must name the same face every time, every dorsal capture must come out the
same handedness, the pitch must not move, and each well's contents must match the
same well in the dish's other captures better than any relabelling would. That
last check uses no fiducial, so it is independent evidence about the labelling
rather than a restatement of it.

Cell geometry is in the render's frame; `gridfit.cells.cell_mask` converts a cell
to a `(width, n_lines)` mask that indexes the cube directly.

## Two notes on history

**The corrections were adopted, not written from scratch.** Both were validated
as **bit-identical** to the earlier dry-run scripts they replaced, on the same
captures, before adoption. Three things changed on purpose: stitching reads each
line once and writes `capture.bin` in the same pass (9.1 s → 2.1 s on a 3,276-line
capture, byte-identical cube); calibration frames and the transmittance geometry
factor are cached; and one geometry implementation now serves both modes, fitting
the lattice and taking row norms, which are rotation-invariant, where the old
transmission script used corner-to-corner step lengths, which are not. On the
transmission board (rotated −2.3°) the two disagree by 0.6%. The superseded
pipelines were removed from the tree — `git show bb4b05e` has them.

**`preview_coco/generate.py` is gone.** It rendered the one PNG per capture that
was hand-annotated, and enforced the frame contract that makes the export usable:
`png[i, j] ↔ cube[i, j, :]`, both `(640, n_lines)`, so a decoded mask indexes the
cube with no rotation, flip, transpose or rescale. This is deliberately *not*
gridfit's frame, which rotates every plane 90° so the dish reads upright. The
script was lost in the consolidation; the 5,456 masks it enabled are safe, and
the contract it enforced is still checked on every dataset build by
`verify_dataset.py` ("cube slice agrees with `gridfit.render.to_working`"), so
nothing downstream is unverified. Only re-rendering the annotation images would
need it rewritten.
