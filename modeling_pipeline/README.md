# modeling_pipeline

Everything downstream of the corrected cubes: hand-drawn kernel masks and the
gridfit lattice in, trained models out.

**Two modelling tracks, and only two:**

|  | PLS | CNN |
|---|---|---|
| **variety typing** — 4 cultivars | `train_pls.py --task variety` | `train_cnn.py --task variety` |
| **germination** — binary, 89/11 | `train_germination.py` | `train_cnn.py --task germination` |

`barley/tasks.py` holds those two plus `dish`, which is not a target but the
control that says whether either is being read off the plate rather than the
grain.

```bash
# build
python3 build_dataset.py --dry-run     # counts and bytes, writes nothing
python3 build_dataset.py               # ~14 GB, ~5 min
python3 verify_dataset.py              # the gate; must be green
python3 -m barley.splits               # dataset/splits.json         (variety)
python3 -m barley.germination          # dataset/splits_germination.json

# model
.venv/bin/python train_germination.py                      # binary, all feature sets
.venv/bin/python train_cnn.py --task germination --hours 8
.venv/bin/python train_pls.py --task variety --controls
.venv/bin/python train_cnn.py --task variety

# look at the data
python3 explore/review_masks.py        # mask_review/
python3 explore/explore_germination.py # exploration/germination/
python3 explore/pca_spectra.py         # exploration/pca/mean_spectra/
python3 explore/pca_pixels.py          # exploration/pca/per_pixel/
```

`build_dataset.py` and `verify_dataset.py` need only numpy/scipy/cv2 and run on
the system python. The models need scikit-learn and torch, which live in
`.venv/` — hence the two different interpreters above.

## Excluded varieties

`config.EXCLUDED_VARIETIES = (2,)` drops **prospect2** (dishes 5-9) from the
modelling path: 110 of its 110 scored kernels never germinated.

It is dropped rather than kept as a class because a variety with one outcome
cannot teach anything about germination and actively corrupts what it is asked.
It is a fifth of the plate sitting entirely on one side of every germination
target, so a model that learns nothing but "is this prospect2?" scores well on
germination without predicting germination — and since variety is confounded
with dish and folds hold out whole dishes, that shortcut survives
cross-validation.

The rows are still in `patches.npy` and `index.csv`. The exclusion is applied in
exactly one place, `barley.index.Index.load`, which every trainer, split
builder, dataset and figure goes through, and it prints what it dropped rather
than doing it silently. `include_excluded=True` is the escape hatch, used by the
label parser's cell map (the annotator scored the whole plate) and by
`review_masks.py --include-excluded`. Setting the tuple to `()` brings the
variety back with no rebuild — but re-run `python3 -m barley.splits` and
`python3 -m barley.germination` afterwards, since the fold files list kernels.

Class ids are positions in `config.kept_varieties()`, not raw variety numbers,
so the excluded variety leaves no always-zero one-hot column, no empty
confusion-matrix row, and no wrong chance level. Scope after the exclusion:
**438 kernels, 3,502 views, 20 dishes, 4 varieties** (chance 0.250).

## What a row is

One row per **(kernel, mode, side, hours)** view. Day 1 is 0 h and day 9 is 8 h;
day 2 (24 h) is out of scope. That gives 4,381 rows over 548 kernels in 25
dishes:

```
25 dishes x 22 wells x 8 views = 4400
  - 16   two wells are empty in every view (dish13 R0C4, dish21 R3C1)
  -  3   three masks are genuinely missing
  = 4381
```

`kernel_uid` — `dish7_R2C6` — is the spine. It is gridfit's own well name, it
means the same physical well on every capture from either face of the plate, and
it is the key the germination CSV must join on.

**"day" is overloaded in this project.** The capture folders `day1/day9/day2` are
rig slot names meaning 0 h / 8 h / 24 h. The germination photos under
`labels/germination_photos/day1..day5` are calendar days of the assay. Join on
`hours` or on `kernel_uid`, never on a column called `day`.

## What is stored

`dataset/` holds per-pixel **pseudo-absorbance**, `A = -log10(max(x, 1e-4))`:

```
patches.npy   (4381, 128, 64, 192) float16   the CNN input, memmapped
masks.npy     (4381, 128, 64)      bool      the annotator's mask, same warp
spectra.npy   (4381, 192)          float32   mask-mean per view; the PLS input
index.csv     one row per view, with per-view quality columns
bands.json    band indices, wavelengths, patch geometry
build_meta.json  COCO sha256, scope, every constant that shaped the build
```

**SNV is not baked in.** Every dataloader path applies it by default, so a model
sees exactly `-log10 → SNV` as intended, but it lives in `barley/transforms.py`
where swapping it for a Savitzky-Golay derivative is a flag rather than an
hour-long rebuild. That matters while the masks are still provisional.

Two SNV paths, both from the same stored absorbance:

* **PLS** — mean over the mask's pixels, then SNV that spectrum.
* **CNN** — per-pixel SNV across bands, then zero outside the mask.

`spectra.npy` is 3 MB, so spectral work never touches the 14 GB of patches.

### The patch geometry

Each cell is warped onto a fixed 128×64 rectangle. The warp removes the plate's
rotation, removes the dorsal/ventral mirror, puts both modes on a common grid
and makes an 0 h patch pixel-comparable with an 8 h one. An axis-aligned crop
does none of that.

`config.CELL_SCALE = 1.10` pads the cell outwards before warping. This is not
cosmetic: the kernels sit about **0.27 cell-widths off-centre** in their wells,
held to one side by the retaining clip, so the 15% inset inherited from
`extract_kernels.py` sliced straight through the grain. `verify_dataset.py`
checks that no mask reaches the patch border.

## The gate

`verify_dataset.py` exits non-zero and nothing should train against a dataset it
refuses. Every check exists because its failure is invisible downstream — a
mirrored patch, a mask attributed to the neighbouring well, a spectrum that no
longer matches the pixels it summarises all still produce a clean loss curve.

It re-derives the mask→cell assignment from source and requires it to agree with
what was stored; checks the RLE decoder against the exporter's own `area`/`bbox`;
checks the cube slice against `gridfit.render.to_working`; requires masks to
survive the warp non-empty, connected and off the border; recomputes every
sampled spectrum from `patches.npy` + `masks.npy` and requires an exact match;
and confirms the index's own structure (8 views per kernel, one variety per dish,
hours from the day map, the two empty wells absent).

`--contact-sheet` renders 20 random patches with their mask outline to
`reports/`. A mirrored or misindexed patch is obvious to the eye and invisible
in a loss curve.

## Tasks, and why the splits look the way they do

`barley/tasks.py` is the modularity contract: **moving from variety to
germination changes that file only.** The index, transforms, datasets, splits
and models take a `Task` and never ask what it means.

That holds for swapping one *label* for another. It does not extend to a
different learning problem: `argmax`, the confusion matrix and
`done = y_pred >= 0` are all class-shaped. A regression or survival target would
need its own trainer, not a new `Task`. One was written and then deleted — see
[Scope](#scope).

**Variety is perfectly confounded with dish** — every kernel of variety 1 is in
dishes 0–4 and nowhere else. So any dish-level artifact (plate, illumination,
capture session) is a *perfect* variety predictor, and a random kernel-level
split would score near-perfect while learning nothing about barley. All folds
therefore hold out **whole dishes**: 5 folds, one dish per variety each.

Two controls, via `train_pls.py --controls`:

* **permutation floor** — the same pipeline with labels shuffled between dishes.
  Not `1/n_classes`; with grouped folds and correlated views the real chance
  level sits above it, and the gap to it is the only meaningful claim.
* **dish identity** — how well the same features predict *which dish* a view came
  from. If dish is as predictable as variety, the model is reading the plate.

Results are reported per view and **per kernel** (the 8 views of a kernel
averaged). Per-view answers "can we call this measurement?"; per-kernel answers
"can we call this grain?".

## Scope

**Did this kernel germinate, or not.** One bit per kernel, predicted from its
spectra, with PLS and with a CNN. That is the whole project.

A germination-**time** model was built and then deleted on 2026-08-20 —
`train_survival_{pls,cnn}.py`, `barley/survival{,_metrics}.py`,
`barley/models/*_survival.py`, `reporting/survival_*.py`, the `germination_time`
task and the distribution fit. Every number it produced predated both the
prospect2 exclusion and the measured clock, so none of it was comparable with
anything current. The label reader and the germination-rate fold builder
survived, in `barley/germination.py`.

Timing is not gone from the data — `barley/timing.py` and
`timing.intervals()` hold the measured brackets, and they are what a future
timing model would be built on. It is gone from the *models*.

## The germination label

### The label file

`germination_label_annotator.xlsx` — one row per dish, one column per kernel
index 0–21. That index is gridfit's `cell_index`, the same numbering stamped on
`labels/germination_photos/index_reference_{dorsal,ventral}.png`, so it names the
same physical well on every dish.

**Three states, not two.** Everything downstream depends on keeping them apart:

| cell | means | what happens to it |
|---|---|---|
| `1`–`5` | first seen germinated at that scoring visit | an interval, see below |
| `-1` | scored, never germinated | a real observation — **kept** |
| blank | that dish has not been scored yet | a missing label — **dropped** |

Reading blank as `-1` would train the model to predict dormancy from unscored
plates; reading `-1` as blank would throw away the kernels the whole question is
about. All 548 are now scored, so no blanks remain, but the parser still
enforces it. A `.csv` with `kernel_uid,first_germinated_day` is accepted too.

**Three clocks, one origin** (the moment water was added). `config.HOURS` is the
capture clock — day1 = 0 h, shot *before* wetting; day9 = 8 h; day2 = 24 h.
`config.GERMINATION_DAYS` is the scoring clock, nominally day k = 24k h. The
photo folders under `labels/germination_photos/` use hours. Join on `hours` or
`kernel_uid`, never on anything called `day`. All three nominal clocks are
approximations — see [The measured clock](#the-measured-clock-and-interval-labels).

```bash
python3 verify_dataset.py                   # gates the label file, reports coverage
python3 -m barley.germination               # dataset/splits_germination.json

```

Point any of them elsewhere with `BARLEY_GERMINATION_LABELS=...`.

### The confound, which was the story — now cut out

**A whole variety was degenerate.** `prospect2` (dishes 5–9) is 110/110
never-germinated. Variety is perfectly confounded with dish and folds hold out
whole dishes, so a model could score well on the ever/never call purely by
recognising a prospect2 plate. On the first real run, before the exclusion:

* pooled never-call balanced accuracy **0.868**
* mixed varieties only (prospect1 + laureate1) **0.656**

and the constant predictor's **12.7 h** MAE beat the model's **17.7 h** — so the
model was worse than doing nothing at the *timing* half and only won on the
confounded half.

Scoring is now complete (548/548) and prospect2 is excluded outright — see
[Excluded varieties](#excluded-varieties). The remaining four varieties
germinate at 87–91%, so the ever/never call is now a 1-in-9 minority class
rather than a variety-recognition task, and the run numbers above are not
comparable with anything produced after the exclusion. `train_germination.py`
still reports a by-variety breakdown and both within-dish and within-variety
AUC on every run; none of that is optional, because the four survivors still
differ (see
`exploration/germination/varieties_compared.png`: `unknown` starts far slower,
4% up at 24 h against 26–33%, and catches up by 72 h).

### Three things that fail quietly

* **Average precision** returns 1.000 silently for a single-class column, and
  its chance level *is* the prevalence, not 0. `barley/metrics.py` returns
  `None` for the degenerate case, and every report prints AP beside the
  prevalence it has to be read against (0.110 here).
* **Accuracy is meaningless at 89/11.** Answering "yes" to everything scores
  0.890. Nothing quotes bare accuracy.
* **`train_pls.py --task germination` will look much worse than
  `train_germination.py`** — 0.585 balanced accuracy against 0.688 on identical
  data. Not a bug: the generic trainer takes the argmax of a one-hot PLS fit,
  which is an implicit 0.5 threshold, and on this imbalance that says yes to
  nearly everything. `train_germination.py` tunes the threshold inside each
  training fold. Use it for the binary target; `train_pls.py` is for the
  near-balanced variety control.

## Reports

Every training run writes two artifacts and then builds its own report folder:

```
runs/<name>.json     config, per-fold records, pooled metrics, confusion
runs/<name>.npz      per-view predictions: dataset_row, fold, y_true, y_pred, score
reports/<name>/      summary.md + figures
```

Keeping the per-view predictions is what makes reporting a separate step — a
figure can be added, fixed or restyled later and every report regenerated in
seconds, without retraining anything.

```bash
python3 report_run.py --latest              # rebuild the most recent run
python3 report_run.py --all                 # every run in runs/
python3 report_run.py --dataset             # dataset overview, no run needed
python3 report_run.py --latest --mode dark  # dark-surface figures
.venv/bin/python train_pls.py ... --no-report   # skip the automatic build
```

Each figure answers a question that actually came up on this project:

| figure | answers |
|---|---|
| `01_fold_scores` | how much does the score move between held-out dish sets? |
| `02_confusion` | which classes get mistaken for which? |
| `03_per_class_recall` | is the headline hiding a class that does not work? |
| `04_per_dish` | is a bad fold a hard variety, or one awkward plate? |
| `05_spectra_by_class` | are these classes even separable in the spectra? |
| `06_components` | PLS: did component selection saturate, or is it truncated? |
| `06_training_curves` | CNN: is it learning, or memorising its training dishes? |

`summary.md` restates every figure's numbers as tables — headline, per fold, per
class, the confusion counts, the hardest dishes, and the run config. It is the
accessible twin of the figures, it is the required relief for the three
categorical hues that sit below 3:1 contrast on a light surface, and it is the
part that still reads in a terminal or a diff a year from now.

The palette in `reporting/style.py` is a documented, validated set: hues are
assigned in fixed slot order and never cycled, magnitude uses one hue light to
dark, and separation was computed rather than eyeballed (worst adjacent CVD ΔE
9.1 light / 8.4 dark; worst adjacent normal-vision 19.6 / 19.3). The five
categorical hues are used **only** in line charts, where position cannot
separate the series. Bar and dot charts whose axis already names the category
get a single hue — colouring them by category would re-encode what the chart
already shows, and comparing any two of five colours invokes an all-pairs
separation gate that only three of these slots clear.

## Layout

```
config.py             paths, scope, every constant that shapes the dataset
timing.json           the measured clock -- a hand-acquired INPUT, versioned,
                      not reproducible here (see The measured clock)
germination_label_annotator.xlsx   the scored labels, likewise an input

                      -- pipeline --
build_dataset.py      cubes + COCO + gridfit -> dataset/
verify_dataset.py     the gate; must be green before anything trains

                      -- modelling: two tracks --
train_pls.py          PLS-DA, multiclass. The variety track.
train_germination.py  PLS-DA, binary + threshold tuning. The germination track.
train_cnn.py          ResNet-18 with a spectral stem. Either track, --task.
report_run.py         build reports/<run>/ from a run's json + npz

                      -- looking at the data, not modelling it --
explore/review_masks.py       one sheet per capture: the masks as stored
explore/explore_germination.py  label histograms, pooled and per variety
explore/pca_spectra.py        unsupervised structure in the kernel means
explore/pca_pixels.py         unsupervised structure inside a kernel
reporting/
  style.py            validated palette, matplotlib style, title/reference helpers
  figures.py          one function per figure
  session.py          assembles a report folder and writes summary.md
barley/
  coco.py             RLE decode, iscrowd fix, largest-component, erosion knob
  assign.py           mask -> gridfit well, as a required bijection
  extract.py          cube slice -> working frame -> absorbance -> warp
  index.py            index.csv + Selection
  tasks.py            VARIETY, GERMINATION, DISH (the control)
  germination.py      the label sheet, and the germination-rate folds
  timing.py           the measured clock; per-dish t0, capture hours, intervals
  kernels.py          side-averaged, one row per kernel
  pca.py              the projection machinery both PCA scripts share
  germination_sheet.py  the annotator .xlsx, read with stdlib zipfile
  transforms.py       snv, savgol, the PLS and CNN recipes
  datasets.py         SpectraDataset, PatchDataset
  splits.py           dish-grouped stratified folds
  metrics.py          balanced accuracy, per-kernel aggregation, controls
  models/pls.py       PLS-DA + inner-CV component selection
  models/cnn.py       ResNet-18, 192-channel stem or PCA-3 + pretrained
  runlog.py           what a run leaves behind: the json + npz pair
annotations/          the COCO export (versioned) + preview PNGs (not)
labels/               germination.csv (pending) + scoring photos
dataset/ runs/ reports/   generated
mask_review/          generated: 160 capture sheets (~100 MB, gitignored)
exploration/          generated: figures that describe the data, not a run
  germination/        label histograms, per variety, and the scoring visits
  pca/mean_spectra/   unsupervised structure in the kernel-mean spectra
  pca/per_pixel/      unsupervised structure inside a kernel
```

## Looking at the data

Two things that are not models and not the gate:

* `review_masks.py` — one PNG per (mode, day, dish, side), at
  `mask_review/<mode>/day<N>_<H>h/dish<NN>_<side>.png`. Left half: gridfit's
  render of the cube `index.csv` names, with every dataset mask outlined and
  every extraction quad drawn. Right half: the 4x7 plate, each well holding the
  patch as stored in `patches.npy` and the mask as stored in `masks.npy`, read
  from those files rather than recomputed, with everything outside the mask
  washed out because every model path discards it. Each panel is titled with
  its germination label, so a mask attributed to the neighbouring well shows up
  as a label that does not match the grain under it — the failure that changes
  every downstream number and appears in no metric. `summary.csv` carries the
  per-capture counts.
* `explore_germination.py` — count vs germination time, pooled and one per
  variety, plus a cumulative-incidence overlay and `summary.md`/`counts.csv`.
  Counts are kernels, not views, and each bar spans the interval the
  germination actually happened in. `never` is drawn past an axis break because
  it is right-censored at 120.9–128.6 h, not a sixth interval. Also
  `scoring_visits.png`, which is why those edges are ranges.

## PCA

Two scripts over `barley/pca.py`, which reuses `transforms.spectrum_pipeline`
for SNV and the derivative and adds the one thing PCA needs that the modelling
path does not: **column mean-centring**. SNV centres each spectrum against
itself; PCA needs each band centred across rows, and doing the first does not do
the second. Bands are deliberately not autoscaled — all 192 columns are the same
quantity in the same units, so unit-variance scaling would only promote the
noisiest bands.

Reflectance and transmittance always get separate PCAs. In one matrix, PC1 is
"which camera mode" and explains most of the variance and nothing else.

* `pca_spectra.py` — six matrices of **438 kernels x 192 bands**: two modes x
  {0 h, 8 h, 8 h − 0 h}. The eight views per kernel are averaged over **sides
  only**: 0 h is the dry seed and 8 h the imbibed one, two physical states, so
  averaging them would throw away the imbibition contrast.
* `pca_pixels.py` — the same four (mode, hours) cells at voxel level, ~100k
  rows each, fitted on the mask as drawn and on the mask eroded 2 px.

The controls are the point. Variety is perfectly confounded with dish, so
`controls.csv` reports, per component, the share of variance between dishes
beside the share between varieties, plus both **after centring within variety**
(`dish|var`, `day|var`) — the only columns that can be read as being about the
grain. What came out:

* `dish` beats `variety` in every matrix. The pure plate effect `dish|var`
  reaches **0.41** on the reflectance delta.
* Nothing separates germinated from never: `germ` never exceeds 0.081.
* The one real germination signal is **reflectance at 8 h** — `day` 0.234
  falling to `day|var` **0.155** once variety is removed. Nothing else is close.
* Transmittance carries roughly half reflectance's plate effect, and less
  germination signal. The two modes are not redundant.
* At voxel level, **48–58% of all reflectance variance lives in the outermost
  two pixel rings of the mask** — partial-volume mixing of grain and well, which
  the score maps show as a ring on PC1. Transmittance loses only 14–15% to the
  same erosion. This is the measured version of the contaminant
  `config.MASK_ERODE_PX` was switched off for, and it is worth revisiting for
  reflectance.

## Germination: the binary call

`train_germination.py` — will a kernel germinate by the last scoring visit?
One row per kernel (side-averaged, 438 x 192 per block), PLS-DA, five
dish-grouped folds, whole outer CV repeated 5x with fresh splits.

**Stratification.** Folds are balanced on *both* per-dish germination rate and
**variety**, by a Latin square: within each variety the dishes are ranked by
their own rate and rank *r* goes to fold `(r + offset) % 5` with a different
offset per variety. Every fold therefore holds exactly one dish of each of the
four cultivars (22 kernels each) with never-counts of 7-12. Stratifying on rate
alone — which is what it did — produced a fold that was 75% one variety with two
varieties absent. The same Latin square is used for the inner folds.

**What is chosen inside each training fold**, never on the test fold:
representation and component count by inner **AUC** (threshold-free, so the
component count is not entangled with an operating point picked on the same
data), then the decision threshold by inner balanced accuracy. The target is
390/48, so accuracy is worthless — an always-yes model scores 0.890.

### The honest number

| | nested (procedure) | fixed `refl_8h` |
|---|---|---|
| AUC | **0.716 ± 0.025** | **0.765 ± 0.013** |
| within-dish AUC | 0.761 | **0.813** |
| within-variety AUC | 0.745 | 0.767 |
| permutation floor | 0.491 | 0.469 |
| AP (never) | 0.432 | 0.419 |
| balanced accuracy | 0.641 | 0.661 |
| sensitivity / specificity | 0.656 / 0.625 | 0.697 / 0.625 |

The nested column lets the model pick its own representation inside each fold
and is unbiased. The fixed column pins it to reflectance-at-8 h a priori.

**The nested selector is worse than fixing the representation** — 0.716 against
0.765. With 16 training dishes the inner AUC estimate is too noisy to choose
between fourteen candidates reliably; it took `refl_8h` in three folds and the
768-column `all` in two, and the `all` folds scored 0.614 and 0.844. So **fix
the representation**. Doing so is defensible without touching a test score: the
0 h / 8 h gap is systematic across all four mode x derivative combinations
(0.60-0.62 dry, 0.74-0.77 imbibed) and the PCA reached the same conclusion
unsupervised. The derivative makes no difference (0.765 vs 0.768).

### It does not work equally on all four varieties

| variety | never | AUC | within-dish AUC |
|---|---|---|---|
| prospect1 | 10 | **0.996** | **1.000** |
| laureate1 | 13 | 0.712 | 0.734 |
| laureate2 | 14 | 0.734 | 0.705 |
| **unknown** | 11 | **0.563** | **0.625** |

Read the within-dish column: `prospect1`'s duds are almost perfectly separable
and it is not a plate effect. The laureates sit around 0.72. **`unknown` is at
chance.** A quarter of the data is dragging the headline down while three
quarters does better than it looks — and `unknown` is also the variety with the
~5 h longer germination lag. Whether those are the same fact is the next
question.

Within-dish AUC above pooled, throughout, is what says the signal is grain and
not plate: it compares only kernels that shared a plate, so no plate signature
can contribute. Full table, controls and figures in `reports/germ5/summary.md`.

## The measured clock, and interval labels

`config.HOURS` and `config.GERMINATION_DAYS` describe the protocol. They are not
what happened. `timing.json` records the real timeline, reconstructed from two
independent sources — capture-file mtimes pulled off the acquisition machine,
and EXIF `DateTimeOriginal` on the scoring photos.

**It is a hand-acquired input, not a build product.** The tooling that collected
it has been removed now that the job is done, so it cannot be regenerated here;
rebuilding means going back to the acquisition machine. It sits at the repo root
and is versioned for that reason, alongside the label spreadsheet. Do not delete
it.

**t=0 is per dish.** The day-1 session took **8.3 hours** to work through 25
plates: dish0 was scanned at 10:55, dish24 at 19:14. There is no single zero.
Each dish's zero is its own dry, pre-wetting scan, and its later scans and
photos are measured from that. The per-dish lags are stable precisely because
the operator worked the plates in the same order every time.

| clock | nominal | measured (mean) | range |
|---|---|---|---|
| day1 capture | 0 h | 0.02 h | 0.00 – 0.22 |
| day9 capture | 8 h | **7.59 h** | 6.91 – 8.95 |
| scoring visit 1 | 24 h | **22.61 h** | 21.37 – 25.99 |
| scoring visit 2 | 48 h | 48.39 h | 45.43 – 53.39 |
| scoring visit 3 | 72 h | 72.02 h | 69.02 – 77.05 |
| scoring visit 4 | 96 h | 95.65 h | 92.65 – 100.77 |
| scoring visit 5 | 120 h | **123.69 h** | 120.93 – 128.58 |

**Labels are intervals.** A kernel labelled "day 2" was seen ungerminated at
visit 1 and germinated at visit 2. That is a bracket, not a germination time:

```
label k   ->  ( visit[k-1] , visit[k] ]     visit[0] = 0, the wetting moment
never     ->  ( visit[5] , inf )            right-censored, fully observed
unscored  ->  (nan, nan)                    dropped, never read as never
```

390 events with brackets **21.4–28.2 h wide** (median 24.5); 158 censored at
**120.9–128.6 h**. `timing.intervals(uids, dishes, table)` returns them.

**Using it.** `Index.load()` attaches `capture_time` and `capture_hours` from
`timing.json` — outside index.csv, because the times are a fact about captures
rather than rows and correcting them must not cost a 14 GB rebuild. `hours`
stays the nominal slot every existing selection and split is written against.

```python
from barley import timing
timing.capture_hours("reflectance", 9, 7, "dorsal")   # 7.42, not 8.0
timing.visit_hours(7)                                 # that dish's five visits
lo, hi = timing.intervals(uids, dishes, table)        # the brackets
```

**Transmittance times are a flagged proxy.** They were never collected, so those
1,751 views inherit the reflectance timestamp for the same (day, dish, side) and
carry `capture_time_source == "proxy_reflectance"`. The error is bounded by one
dish slot, ~14 min — a dish's two reflectance scans are ~1.2 min apart while
consecutive dishes are ~13.7 min apart, leaving ~12.5 min of slack per dish
where the transmittance scans sit if the modes were interleaved per plate.
Against a 7.6 h offset and 21–28 h label brackets that is immaterial, but check
the flag before quoting one. The nominal value is never substituted.

## Known data facts

* All 5,456 annotations carry `iscrowd: 1` — CVAT stamps that on RLE exports and
  most COCO tooling silently drops crowd annotations. `barley/coco.py`
  normalises it.
* 7.5% of masks carry a detached speck (worst 58 px). Largest-component only.
* Mask boundary erosion is available (`config.MASK_ERODE_PX`) but **off**: at
  2 px it cost a median 23.5% of mask area, too much to pay pre-emptively.
* Transmittance patches are ~73% NaN *over the cell box* because the open
  aperture rails the detector — but only ~1.7% inside the kernel mask, and the
  mask-mean spectra are fully finite. Filter on `nan_frac`/`sat_frac` via
  `Selection` rather than rebuilding.
* Three masks are genuinely missing, for the fact-check:
  `refl day9 dish13 ventral R3C2`, `trans day9 dish3 dorsal R1C2`,
  `trans day9 dish8 dorsal R2C0`.
* The masks are provisional. `build_meta.json` records the COCO sha256 and
  `verify_dataset.py` warns when the file on disk no longer matches; rebuilding
  is one command and ~5 minutes.
