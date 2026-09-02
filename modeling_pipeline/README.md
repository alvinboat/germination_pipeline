# modeling_pipeline

Everything downstream of the corrected cubes: hand-drawn kernel masks and the
gridfit lattice in, trained models out.

All results, controls and caveats are in `PROJECT_REPORT.md` at the repo root —
they are not repeated here, so there is exactly one place to correct a number.
This file is how the code works.

```
config.py             paths, scope, every constant that shapes the dataset
timing.json           the measured clock -- a hand-acquired INPUT, versioned
germination_label_annotator.xlsx   the scored labels, likewise an input
annotations/          the COCO mask export (input, 5,456 hand-drawn masks)

                      -- pipeline --
build_dataset.py      cubes + COCO + gridfit -> dataset/
verify_dataset.py     the 24-check gate; must be green before anything trains

                      -- modelling: two tracks --
train_germination.py  PLS-DA, binary + threshold tuning. The germination track.
train_pls.py          PLS-DA, multiclass. The variety track.
train_cnn.py          ResNet-18 with a spectral stem. Either track, --task.
report_run.py         build reports/<run>/ from a run's json + npz

                      -- looking at the data, not modelling it --
explore/review_masks.py         one sheet per capture: the masks as stored
explore/explore_germination.py  label histograms, pooled and per variety
explore/pca_spectra.py          unsupervised structure in the kernel means
explore/pca_pixels.py           unsupervised structure inside a kernel

barley/
  coco.py             RLE decode, iscrowd fix, largest-component, erosion knob
  assign.py           mask -> gridfit well, as a required bijection
  extract.py          cube slice -> working frame -> absorbance -> warp
  index.py            index.csv + Selection; applies the prospect2 exclusion
  tasks.py            VARIETY, GERMINATION, DISH (the control)
  germination.py      the label sheet, and the germination-rate folds
  germination_sheet.py  the annotator .xlsx, read with stdlib zipfile
  timing.py           the measured clock; per-dish t0, capture hours, intervals
  kernels.py          side-averaged, one row per kernel
  splits.py           dish-grouped stratified folds
  transforms.py       snv, savgol, the PLS and CNN recipes
  datasets.py         SpectraDataset, PatchDataset
  pca.py              the projection machinery both PCA scripts share
  metrics.py          balanced accuracy, per-kernel aggregation, AUC, controls
  models/pls.py       PLS-DA + inner-CV component selection
  models/cnn.py       ResNet-18, 192-channel stem or PCA-3 + pretrained
  runlog.py           what a run leaves behind: the json + npz pair
reporting/
  style.py            validated palette, matplotlib style, title/reference helpers
  figures.py          one function per figure
  session.py          assembles a report folder and writes summary.md

dataset/ runs/ reports/ mask_review/   generated, gitignored
exploration/                           generated figures that describe the DATA
                                       rather than a run; small, and kept
```

## Running

```bash
python3 build_dataset.py --dry-run     # counts and bytes, writes nothing
python3 build_dataset.py               # ~14 GB, ~5 min
python3 verify_dataset.py              # the gate; must be green
python3 -m barley.splits               # dataset/splits.json          (variety)
python3 -m barley.germination          # dataset/splits_germination.json

.venv/bin/python train_germination.py                      # THE model
.venv/bin/python train_pls.py --task variety --controls
.venv/bin/python train_cnn.py --task germination --hours 8

python3 explore/review_masks.py        # mask_review/
python3 explore/explore_germination.py # exploration/germination/
python3 explore/pca_spectra.py         # exploration/pca/mean_spectra/
python3 explore/pca_pixels.py          # exploration/pca/per_pixel/
```

**Two interpreters, by design.** `build_dataset.py`, `verify_dataset.py` and
everything in `explore/` need only numpy/scipy/cv2/matplotlib and run on the
system `python3`. The trainers need scikit-learn and torch, which live in
`.venv/`. `.venv/` bakes in absolute paths — recreate it, never copy it:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy opencv-python matplotlib scikit-learn torch
```

Developed against numpy 2.5.2, scipy 1.18.0, opencv 5.0.0, matplotlib 3.11.1,
scikit-learn 1.9.0, torch 2.6.0+cu124. There is no `requirements.txt`.

**Proving a move worked:**

```bash
.venv/bin/python train_germination.py --features refl_8h --deriv 0 --no-controls
```

should print AUC ≈ 0.765, within-dish ≈ 0.813.

## What a row is

One row per **(kernel, mode, side, hours)** view — one kernel as seen in one
capture. `day1` is 0 h and `day9` is 8 h; `day2` (24 h) is out of scope.

```
25 dishes x 22 wells x 8 views = 4400
  - 16   two wells are empty in every view (dish13_R0C4, dish21_R3C1)
  -  3   three masks the annotator missed, one view each
  = 4381
```

`kernel_uid` — `dish7_R2C6` — is the spine. It is gridfit's own well name, it
means the same physical well on every capture from either face, and it is the key
the label sheet joins on.

## What is stored

```
patches.npy   (4381, 128, 64, 192) float16   the CNN input, memmapped
masks.npy     (4381, 128, 64)      bool      the annotator's mask, same warp
spectra.npy   (4381, 192)          float32   mask-mean per view; the PLS input
index.csv     one row per view, with per-view quality columns
bands.json    band indices, wavelengths, patch geometry
build_meta.json  COCO sha256, scope, every constant that shaped the build
```

Values are per-pixel **pseudo-absorbance**, `A = -log10(max(x, 1e-4))`. Each cell
is warped onto a fixed 128×64 rectangle by a homography from its four corners,
which does four jobs at once: removes the plate's rotation, removes the
dorsal/ventral mirror, puts both modes on a common grid, and makes an 0 h patch
pixel-comparable with an 8 h one. The mask goes through the *same* homography, so
patch and mask are registered by construction. `config.CELL_SCALE = 1.10` pads
the cell outwards first, because the kernels sit ~0.27 cell-widths off-centre in
their wells (a retaining clip holds them to one side).

**SNV is not baked in.** `barley/transforms.py` applies it on every load path, so
changing preprocessing is a flag rather than an hour-long rebuild. Two paths from
the same stored absorbance: PLS takes the mask mean then SNVs that spectrum; the
CNN does per-pixel SNV across bands, then zeroes outside the mask.

`spectra.npy` is 3 MB, so spectral work never touches the 14 GB of patches.

## The gate

`verify_dataset.py` exits non-zero and nothing should train against a dataset it
refuses. Every check exists because its failure is invisible downstream — a
mirrored patch, a mask attributed to the neighbouring well, or a spectrum that no
longer matches the pixels it summarises all still produce a clean loss curve and
a plausible confusion matrix.

It re-derives the mask→cell assignment from source and requires agreement with
what was stored; checks the RLE decoder against the exporter's own `area`/`bbox`;
checks the cube slice against `gridfit.render.to_working`; requires masks to
survive the warp non-empty, connected and off the patch border; recomputes every
sampled spectrum from `patches.npy` + `masks.npy` and requires an **exact** match;
and confirms the index's own structure. `--contact-sheet` renders 20 random
patches with their mask outline — a mirrored or misindexed patch is obvious to
the eye and invisible in a loss curve.

## Settled decisions — do not silently re-litigate

The *reasoning* for each of these is in `PROJECT_REPORT.md`; here are the
mechanics.

**prospect2 is excluded** (`config.EXCLUDED_VARIETIES = (2,)`, dishes 5–9) in
exactly one place — `barley.index.Index.load`, which every trainer, split
builder, dataset and figure goes through — and it prints what it dropped.
`include_excluded=True` is the escape hatch, needed by
`germination.build_cell_map` (the annotator scored the whole plate) and
`explore/review_masks.py --include-excluded`. Nothing is deleted: the rows stay
in `patches.npy` and `index.csv`, and setting the tuple to `()` brings the
variety back with no rebuild — but re-run `python3 -m barley.splits` and
`python3 -m barley.germination`, since the fold files list kernel ids. Current
scope: **438 kernels, 3,502 views, 20 dishes, 4 varieties.**

**Folds hold out whole dishes, always**, stratified on per-dish germination rate
**and** variety by a Latin square, so every fold holds exactly one dish of each
cultivar. Two fold files: `dataset/splits.json` (task label) and
`dataset/splits_germination.json` (germination rate — the right one for the
binary target).

**The nominal clock is wrong, and the measured one is per dish.** `config.HOURS`
and `config.GERMINATION_DAYS` are the *protocol*; `timing.json` is the
measurement. `Index.load` attaches `capture_time`, `capture_hours` and
`capture_time_source`. `hours` in `index.csv` stays the NOMINAL slot because
every existing selection, split and task is written against it — use
`capture_hours` when you want the truth. Transmittance times were never
collected, so those 1,751 views inherit the reflectance timestamp for the same
(day, dish, side), flagged `capture_time_source == "proxy_reflectance"`; **check
the flag before quoting one.** The nominal value is never substituted.

`timing.json` is a **hand-acquired input, not a build product** — the tooling
that collected it has been removed. It lives outside the gitignored `dataset/`
for that reason. Do not delete it.

**Labels are intervals, not times.** "Visit 2" means ungerminated at visit 1 and
germinated at visit 2. `timing.intervals(uids, dishes, table)` returns
`(lower, upper)`; `upper` is `inf` for right-censored, both `nan` for unscored.

**PCA needs column mean-centring, which SNV does not do.** SNV centres each
spectrum against itself; PCA needs each band centred across rows.
`barley/pca.py` does both. Bands are never autoscaled — all 192 columns are the
same quantity in the same units.

**`barley/tasks.py` is the modularity contract:** moving from variety to
germination changes that file only. The index, transforms, datasets, splits and
models take a `Task` and never ask what it means. That holds for swapping one
*label* for another; it does not extend to a different learning problem, because
`argmax`, the confusion matrix and `done = y_pred >= 0` are all class-shaped.
It holds exactly three tasks: `variety`, `germination`, and `dish` — the last is
not a target but the control that says whether either of the others is being
read off the plate. Do not add a fourth without being asked.

## Reports

Every training run writes `runs/<name>.json` (config, per-fold records, pooled
metrics, confusion) and `runs/<name>.npz` (per-view predictions: dataset_row,
fold, y_true, y_pred, score), then builds `reports/<name>/`. Keeping the per-view
predictions is what makes reporting a separate step — a figure can be added,
fixed or restyled later and every report regenerated in seconds, without
retraining.

```bash
python3 report_run.py --latest          # rebuild the most recent run
python3 report_run.py --all
python3 report_run.py --dataset         # dataset overview, no run needed
python3 report_run.py --latest --mode dark
```

`summary.md` restates every figure's numbers as tables. It is the accessible twin
of the figures, the required relief for the three categorical hues that sit below
3:1 contrast on a light surface, and the part that still reads in a terminal or a
diff a year from now. The five categorical hues in `reporting/style.py` are used
**only** in line charts, where position cannot separate the series; bar and dot
charts whose axis already names the category get a single hue.

## Footguns

- **`cells.json` records absolute paths from the machine that fitted the grid.**
  Read a capture directory through `config.local_capture_dir()`, never
  `Path(rec["capture"]["source"])`, or the code only works on one machine.
- **`train_pls.py --task germination` is not `train_germination.py`** and will
  look far worse on the same data. Not a bug: the generic trainer takes the
  argmax of a one-hot PLS fit, an implicit 0.5 threshold, which on an 89/11 split
  says yes to nearly everything. `train_germination.py` tunes the threshold
  inside each training fold. Use it for the binary target.
- **Accuracy is meaningless at 89/11** — always-yes scores 0.890. Nothing quotes
  bare accuracy. **Average precision** returns 1.000 silently for a single-class
  column and its chance level *is* the prevalence, not 0; `barley/metrics.py`
  returns `None` for the degenerate case and every report prints AP beside the
  prevalence it must be read against.
- `germination.build_cell_map()` **must** see the full index. Called with a
  filtered one, every excluded kernel's perfectly good label looks like a score
  on a well that holds no kernel and the parser rejects the whole file.
- `read_germination_labels(kernel_uids=...)` checks labels against whatever set
  you hand it. Pass a *selection* and every kernel outside it is reported as an
  orphan.
- **Class ids are positions in `config.kept_varieties()` / `kept_dishes()`**,
  never the raw variety or dish number. A hole in the numbering gives PLS an
  always-zero one-hot column, an empty confusion-matrix row and the wrong chance
  level.
- `pls.choose_components` takes the **max** of its score dict — a raw Brier score
  would select the worst model in the grid.
- Transmittance patches are mostly NaN *over the cell box* (the open aperture
  rails the detector) but only 1–2% *inside the mask*. `index.csv`'s
  `nan_frac`/`sat_frac` are whole-patch and read as alarming for no reason;
  filter with `Selection` rather than rebuilding.
- **`config.MASK_ERODE_PX` is 1, but the prose everywhere says erosion is off.**
  Unresolved. `dataset/build_meta.json` records what the published dataset used
  and is the authority; `verify_dataset.py` prints it on every run.
