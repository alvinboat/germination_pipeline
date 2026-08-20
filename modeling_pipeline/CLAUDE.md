# CLAUDE.md — modeling_pipeline

Guidance for Claude Code working in this directory. `README.md` is the full
reference; this file is the operating manual — what is settled, what will bite
you, and what to run.

## Working on a different machine — read this first

**`modeling_pipeline/` is not in git.** As of 2026-08-20 it is 82 untracked
files, 8.8 MB, and the repo's last commit predates the whole directory. Commit
it before moving, or nothing goes with you.

### What travels in git (~8.8 MB)

All the code, plus four things that are **inputs and cannot be regenerated**:

| | why it is irreplaceable |
|---|---|
| `germination_label_annotator.xlsx` | hand-scored labels, 548 kernels |
| `timing.json` | the measured clock; acquired off the acquisition machine, tooling since removed |
| `annotations/instances_default.json` | the CVAT export — 5,456 hand-drawn masks |
| `exploration/` | finished figures, 4.8 MB |

### What does not travel

| | size | needed by | regenerable |
|---|---|---|---|
| `dataset/` | 13 GB | **everything that models** | yes — `build_dataset.py`, ~5 min, *if* you have the cubes |
| `../real_data/` | 163 GB | `build_dataset.py`, `verify_dataset.py`, `explore/review_masks.py` | **no** — raw capture |
| `../preprocessing_pipeline/collection_pipeline/` | 348 MB + code | the `gridfit` package (imported by sys.path) and `grid_view/*/cells.json` | **no** |
| `annotations/preview_coco/` | 75 MB | nothing at runtime | yes, from the cubes |
| `labels/germination_photos/` | 464 MB | nothing now — EXIF already read into `timing.json` | **no** — primary evidence for the labels |
| `mask_review/ runs/ reports/` | ~100 MB | nothing | yes |

### The shortcut that matters

**Copy `dataset/` (13 GB) and you can do all the modelling without the 163 GB of
cubes.** Both trainers, both PCA scripts and the label histograms read only
`dataset/`, `timing.json` and the xlsx. Exactly three scripts touch raw cubes:
`build_dataset.py`, `verify_dataset.py`, `explore/review_masks.py`.

- **modelling only** — git + `dataset/` ≈ 13 GB. Both tracks work end to end.
  `verify_dataset.py` will not run; note that in anything you report, because
  the gate is then unverified on that machine.
- **full** — add `../real_data/` and `../preprocessing_pipeline/collection_pipeline/`
  ≈ 177 GB.

### Environment

Python 3.12. **Two interpreters, by design:**

- **system `python3`** — numpy, scipy, cv2, matplotlib. Runs `build_dataset.py`,
  `verify_dataset.py`, everything in `explore/`.
- **`.venv/bin/python`** — adds scikit-learn and torch. Required by the
  trainers, and it has everything, so use it when unsure.

`.venv/` is gitignored and bakes in absolute paths. **Recreate it, never copy it:**

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy opencv-python matplotlib scikit-learn torch
```

Developed against numpy 2.5.2, scipy 1.18.0, opencv 5.0.0, matplotlib 3.11.1,
scikit-learn 1.9.0, torch 2.6.0+cu124. There is no `requirements.txt`; add one
if the environment ever bites.

### Paths

`config.ROOT` is `modeling_pipeline/` and `config.REPO` its parent; everything
derives from those. The tree relocates as long as `modeling_pipeline/`,
`preprocessing_pipeline/` and `real_data/` stay siblings.

**One trap, already defused.** All 300 `grid_view/*/cells.json` files record the
absolute path of the cube they were fitted on, from whichever machine and
whichever checkout name ran the fit
— from whichever machine ran the fit. A different user or disk would send
`build_dataset.py` at a directory that does not exist. Both readers now go
through `config.local_capture_dir()`, which keeps everything from the
`real_data/` component onwards and re-roots it at `config.CAPTURE_ROOT`
(defaults to `REPO`, override with `BARLEY_CAPTURE_ROOT`). If `real_data/` lives
somewhere else entirely, set that variable rather than editing 300 JSON files.

### First hour on a new machine

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy opencv-python \
    matplotlib scikit-learn torch
.venv/bin/python -c "import config; from barley import index; \
    print(index.Index.load().describe())"      # proves dataset/ + timing.json are there
.venv/bin/python -m barley.splits              # rebuild both fold files
.venv/bin/python -m barley.germination
.venv/bin/python train_germination.py --features refl_8h --deriv 0 --no-controls
```
That last line should print **AUC ≈ 0.765, within-dish ≈ 0.813**. If it does,
the move worked.

## Two tracks, and only two

|  | PLS | CNN |
|---|---|---|
| **variety typing** (4 classes) | `train_pls.py --task variety` | `train_cnn.py --task variety` |
| **germination** (binary, 89/11) | `train_germination.py` | `train_cnn.py --task germination` |

`tasks.py` holds exactly three: `variety`, `germination`, and `dish` — the last
is not a target but the control that says whether either of the other two is
being read off the plate rather than the grain. Do not add a fourth without
being asked. The `germ_by_dN` family and the germination-time regressor were
both deleted; timing is out of scope.

`train_germination.py` is the binary trainer and is not interchangeable with
`train_pls.py --task germination` — see Footguns.

## What this is

Everything downstream of the corrected cubes: hand-drawn kernel masks + the
gridfit lattice in, trained models out. Source cubes live outside this tree at
`../real_data/<mode>_images/day<N>/dish<M>/<side>/capture.npy`; the gridfit
lattice at `../preprocessing_pipeline/collection_pipeline/grid_view/`.

See **Working on a different machine** above for the two interpreters, what is
and is not in git, and how to stand this up somewhere else.

## Settled decisions — do not silently re-litigate these

**prospect2 is excluded.** `config.EXCLUDED_VARIETIES = (2,)`, dishes 5–9,
110/110 kernels never germinated. Applied in exactly one place,
`barley.index.Index.load`, which every trainer, split builder, dataset and
figure goes through, and it prints what it dropped. `include_excluded=True` is
the escape hatch and is needed by two callers: `germination.build_cell_map` (the
annotator scored the whole plate) and `explore/review_masks.py --include-excluded`.
Changing the tuple requires re-running `python3 -m barley.splits` and
`python3 -m barley.germination` — the fold files list kernel ids.
Current scope: **438 kernels, 3,502 views, 20 dishes, 4 varieties.**

**The nominal clock is wrong, and the measured one is per dish.** `config.HOURS`
and `config.GERMINATION_DAYS` are the *protocol*, not the measurement.
`timing.json` at the repo root records what actually happened; `barley/timing.py`
reads it and `Index.load` attaches `capture_time` and `capture_hours`.

`timing.json` is a **hand-acquired input, not a build product.** Capture times
came from file mtimes on the acquisition machine (foss-research-jetson-4) and
visit times from EXIF on `labels/germination_photos`; the tooling that collected
them (`capture_times.sh`, `build_timing.py`, `reflectance_images_times.txt`) has
been removed now that the job is done. It cannot be rebuilt from anything in
this repo — rebuilding means going back to the Jetson. It is versioned and it
lives outside the gitignored `dataset/` for exactly that reason. **Do not
delete it**, and treat it like `germination_label_annotator.xlsx`.

  - **t0 is per dish**, defined as that dish's own day1 reflectance scan (dry,
    immediately before wetting — so a lower bound on the wetting moment). The
    day-1 session took **8.3 h** to work through 25 plates, so a global t=0
    would carry that spread as label noise.
  - day9 is **7.59 h** after t0 on average (range 6.91–8.95), not 8.
  - Scoring visits land at **22.6 / 48.4 / 72.0 / 95.7 / 123.7 h** on average,
    not 24/48/72/96/120. Visit 1 is early, visit 5 is **+3.7 h late**.
  - `hours` in index.csv stays the NOMINAL slot (0/8/24). Every existing
    selection, split and task is written against it. Use `capture_hours` when
    you want the truth.
  - Transmittance capture times were never collected, so those 1,751 views
    **inherit the reflectance timestamp for the same (day, dish, side)**,
    flagged `capture_time_source == "proxy_reflectance"` on the index and in
    timing.json. The error is bounded by one dish slot, ~14 min: a dish's two
    reflectance scans are ~1.2 min apart while consecutive dishes are ~13.7 min
    apart, so the ~12.5 min of slack per dish is where the transmittance scans
    sit if the modes were interleaved per plate. Immaterial against a 7.6 h
    offset and 21-28 h label brackets, but **check the flag before quoting a
    transmittance capture time**. The NOMINAL value is never substituted.

**Germination labels are intervals, not times.** "Day 2" means ungerminated at
visit 1 and germinated at visit 2 — a bracket, not an event time. Brackets are
**21.4–28.2 h wide** (median 24.5). A never-germinator is right-censored at its
own dish's last visit, which is **120.9–128.6 h**, not 120.
`timing.intervals(uids, dishes, table)` returns `(lower, upper)`; `upper` is
`inf` for censored and both are `nan` for unscored. `config.GERMINATION_CENSOR_H`
(120.0) is the nominal value and is now only correct as a description of intent.

**Three different things are called "day".** `HOURS = {1: 0.0, 9: 8.0}` maps
capture folders to hours since wetting — day1 is 0 h, shot *before* the seeds
were wetted; day9 is 8 h; day2 is 24 h and out of scope. The germination labels
are assay days 1–5 = 24–120 h. `labels/germination_photos/day1..day5` is a third
thing. **Join on `kernel_uid` or `hours`, never on anything called `day`.**

**In the label sheet, `-1` and blank are not the same.** `-1` means scored and
never germinated — a right-censored observation that belongs in training. Blank
means nobody scored it — a missing label, dropped. Merging them teaches the
model to predict dormancy from unscored plates. All 548 are now scored, so no
blanks remain, but the parser still enforces this.

**Never put reflectance and transmittance in one matrix.** Different radiometry,
different dynamic range. Joint PCA gives PC1 = "which camera mode"; a joint
model learns the same. One matrix, one model, one figure per mode.

**Never average 0 h with 8 h.** They are the dry seed and the imbibed seed —
two physical states, not two looks at one. Averaging over *sides* is fine.

**Folds hold out whole dishes, always.** Variety is perfectly confounded with
dish: every kernel of a variety sits in five consecutive dishes and nowhere
else. A kernel-level split lets a model memorise the plate and score near
perfect. Two fold files: `dataset/splits.json` (stratified on the task label)
and `dataset/splits_germination.json` (stratified on per-dish germination rate —
the right one for germination targets).

**SNV is applied at load time, not baked into the file.** `patches.npy` stores
per-pixel pseudo-absorbance `A = -log10(x)`; `barley/transforms.py` applies SNV
on every path. Changing preprocessing is a flag, not an hour-long rebuild.

**PCA needs column mean-centring, which SNV does not do.** SNV centres each
spectrum against itself; PCA needs each band centred across rows.
`barley/pca.py` does both. Bands are never autoscaled — all 192 columns are the
same quantity in the same units.

## Known data facts

- `config.KNOWN_EMPTY_WELLS` — dish13_R0C4, dish21_R3C1 hold no kernel in any
  view. Not annotation misses.
- `config.KNOWN_INCOMPLETE_KERNELS` — three kernels the annotator missed in
  exactly one view, so they have 7 not 8. One (dish8_R2C0) is in the excluded
  variety; the other two survive as one-sided means.
- Transmittance patches are ~40% NaN over the cell box (railed open beam) but
  only ~1% *inside the mask*. Always quote the in-mask figure; `index.csv`'s
  `nan_frac`/`sat_frac` are whole-patch and read as alarming for no reason.
- `config.MASK_ERODE_PX` is **off**. It was switched off because 2 px cost a
  median 23.5% of mask area for an unmeasured contaminant. The contaminant is
  now measured — see PCA findings below.

## What has been found so far

Numbers live in the `exploration/*/summary.md` files; the short version:

**Germination timing.** All 548 kernels scored. Of the 438 in scope: 89%
germinate, modal and median day 2. The hazard peaks at day 2 and falls in every
group, which structurally excludes exponential/Weibull/gamma. A single
log-logistic with a **per-variety lag** (~18 h; `unknown` ~23 h) plus a
per-variety germinable fraction beats letting each variety have its own shape
and scale. Six daily bins is five degrees of freedom, so this can reject a
family and cannot crown one — do not quote "germination time is log-logistic".
→ `exploration/germination/`

**The plate effect is large.** In the mask-mean PCA, `dish` beats `variety` on
every matrix. After centring within variety — leaving only what variety cannot
explain — the pure plate effect still reaches **0.41** of a leading component's
variance on the reflectance delta. This is the standing threat to every result.

**No unsupervised germination axis.** Germinated-vs-never never exceeds 0.081 of
any leading component. That is expected — PCA is unsupervised and has never
heard of germination — and it is **not** evidence that germination is
unpredictable. Do not let that inference stand.

**The one real signal is reflectance at 8 h.** Germination-day variance share
0.234, falling to **0.155** after removing variety. Nothing else is close. The
8 h − 0 h delta looked promising and is the *most* dish-dominated matrix of the
six: differencing cancels the kernel and keeps the session drift.

**Half the reflectance voxel variance is mask edge.** Eroding 2 px removes
48–58% of all reflectance voxel variance (transmittance: 14–15%) and roughly
doubles the between-kernel share of PC1. That is partial-volume mixing of grain
and well, visible as a ring on the PC1 score maps. This is the measured version
of the contaminant `MASK_ERODE_PX` was switched off for.
→ `exploration/pca/per_pixel/summary.md`

**The germination-TIME regressor was deleted (2026-08-20).** The project scope
is binary: did this kernel germinate, or not. Gone with it:
`train_survival_{pls,cnn}.py`, `barley/survival{,_metrics}.py`,
`barley/models/*_survival.py`, `reporting/survival_*.py`, the
`germination_time` task, and `fit_germination_time.py` (the distribution fit).
Their numbers all predated the prospect2 exclusion and the measured clock, so
none were comparable with anything current. The label reader and the
germination fold builder survived the split into `barley/germination.py`. Do
not resurrect any of it without being asked — timing is out of scope.

**The binary germination call works, and the signal is not the plate.** Folds
are stratified on per-dish germination rate **and variety** (Latin square, one
dish of each cultivar per fold); representation, component count and threshold
are all chosen inside the training fold; the whole CV is repeated 5x.

- **nested / unbiased: AUC 0.716 ± 0.025, within-dish 0.761**, AP(never) 0.432
  against 0.110 prevalence, permutation floor 0.491.
- **fixed to `refl_8h`: AUC 0.765 ± 0.013, within-dish 0.813.**

**Fix the representation; do not let the model choose it.** The nested selector
is *worse* (0.716 vs 0.765) — 16 training dishes is too few for the inner AUC to
choose between 14 candidates. Pinning it to reflectance-at-8h costs no test
information: the 0 h / 8 h gap is systematic across all four mode x derivative
combinations and the PCA found it unsupervised. Derivative is a coin flip.

**It does not work on `unknown`.** Per-variety within-dish AUC: prospect1
**1.000**, laureate1 0.734, laureate2 0.705, **unknown 0.625** (pooled 0.563 —
chance). The pooled number is an average over cultivars the model treats very
differently. Never quote it without this breakdown.
→ `reports/germ5/summary.md`

## Where things stand (2026-08-20)

Done and settled:

- Dataset built and gated: 4,381 views on disk, **3,502 / 438 kernels / 20
  dishes / 4 varieties** after the prospect2 exclusion. `verify_dataset.py`
  green, 24/24.
- Masks reviewed sheet by sheet (`mask_review/`, 160 captures). Two known-empty
  wells, two kernels one-sided, nothing else amiss.
- Labels complete, 548/548 scored; interval structure and the measured clock
  resolved into `timing.json`.
- **Germination track: working.** Nested AUC 0.716 ± 0.025, fixed-`refl_8h`
  0.765 ± 0.013, within-dish 0.813. Numbers and caveats below.
- **Variety track: baseline only.** `train_pls.py --task variety` gives 0.685
  per-kernel balanced accuracy against 0.250 chance. Nobody has tuned it, run
  the controls on it, or tried the CNN seriously.
- PCA done; the plate effect is quantified and large.

Open, in rough priority order:

1. **`unknown` is at chance** on the germination call (within-dish 0.625 against
   1.000 / 0.734 / 0.705 for the others). Understand that before tuning anything.
2. **The CNN on germination is untouched** — one smoke epoch, nothing more. It
   also needs the threshold treatment `train_germination.py` has; argmax on an
   89/11 split is useless.
3. **Mask erosion.** `MASK_ERODE_PX` is off, but 48-58% of reflectance voxel
   variance is mask-edge partial-volume. Turning it on is a dataset rebuild.
4. Transmittance capture times are still a flagged proxy.

## Running

```bash
python3 build_dataset.py --dry-run       # counts and bytes, writes nothing
python3 build_dataset.py                 # ~14 GB, ~5 min
python3 verify_dataset.py                # the gate; must be green before training
python3 -m barley.splits                 # dataset/splits.json
python3 -m barley.germination            # dataset/splits_germination.json

python3 explore/review_masks.py          # mask_review/, one sheet per capture
python3 explore/explore_germination.py   # exploration/germination/
python3 explore/pca_spectra.py           # exploration/pca/mean_spectra/
python3 explore/pca_pixels.py            # exploration/pca/per_pixel/

.venv/bin/python train_germination.py    # THE model: binary, threshold-tuned
.venv/bin/python train_cnn.py --task germination --mode reflectance --hours 8
.venv/bin/python train_pls.py --task variety --mode reflectance --controls
```

`verify_dataset.py` is the gate. If it is not green, nothing trains.

## Footguns

- **`cells.json` records absolute paths from the machine that fitted the grid.**
  Always read a capture directory through `config.local_capture_dir()`, never
  `Path(rec["capture"]["source"])` directly, or the code only works on one
  machine.
- `germination.build_cell_map()` **must** see the full index. Called with a
  filtered one, every excluded kernel's perfectly good label looks like a score
  on a well that holds no kernel and the parser rejects the whole file.
- `read_germination_labels(kernel_uids=...)` checks labels against whatever set
  you hand it. Pass a *selection* and every kernel outside it is reported as an
  orphan. `tasks._germinated_by` uses the cached `germination._table()` for this
  reason.
- Class ids are positions in `config.kept_varieties()` / `kept_dishes()`, never
  the raw variety or dish number. A hole in the numbering gives PLS an
  always-zero one-hot column, an empty confusion-matrix row and the wrong
  chance level.
- Average precision returns **1.000 silently** for a single-class column, and
  its chance level *is* the prevalence, not 0. `barley/metrics.py` returns
  `None` for the degenerate case and every report prints AP beside the
  prevalence it must be read against (0.110 here).
- `train_pls.py --task germination` will look far worse than
  `train_germination.py` on the same data (balanced accuracy 0.585 vs 0.688).
  That is not a bug: the generic trainer takes the argmax of a one-hot PLS fit,
  which is a 0.5 threshold, and on an 89/11 split that says yes to nearly
  everything. `train_germination.py` tunes the threshold inside each training
  fold. Use it for the binary target.
- `pls.choose_components` takes the **max** of its score dict — a raw Brier
  would select the worst model in the grid.
- `SpecimStitcher.load_lines()` (upstream, `../preprocessing_pipeline/`) deletes
  its source directory on success. Nothing here calls it; do not start.

## Conventions

- One script per question, `argparse --help` on all of them, output to a folder
  that carries its own `summary.md` restating every number in the figures.
- Anything hand-acquired and unreproducible lives at the repo **root** and is
  versioned (`timing.json`, the label xlsx). Anything a script can rebuild lives
  in a gitignored folder (`dataset/`, `runs/`, `reports/`, `mask_review/`). If
  you create a third kind, say which it is in the file itself.
- Figures go through `reporting/style.py`. The five categorical hues are for
  overlapping line charts only — on a bar or dot chart whose axis already names
  the category, colour re-encodes what the chart shows. Direct-label lines.
- Generated folders (`dataset/`, `runs/`, `reports/`, `mask_review/`) are
  gitignored and reproducible. `exploration/` is small and kept.
- Every quality gate defaults to **off**, so what is excluded is always an
  explicit choice recorded in the run rather than a silent default.
