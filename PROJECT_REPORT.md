# Predicting barley germination from hyperspectral images

**Project report — 20 August 2026**

From raw push-broom line scans to a validated germination classifier: what was
collected, how it was corrected, how it was labelled, what was modelled, and
what the results do and do not support.

---

## 1. Summary

Twenty-five petri dishes of barley kernels were imaged with a near-infrared
hyperspectral line-scan camera in two optical modes, from both faces, before
wetting and eight hours after. The kernels were then grown for five days and
scored by eye for germination. The question is whether the spectra taken **eight
hours after wetting** predict which kernels will germinate — hours before any
visible change.

**They do, on three of the four varieties tested.**

| | value | reference |
|---|---|---|
| AUC, unbiased (nested selection) | **0.716 ± 0.025** | 0.500 = chance |
| AUC, representation fixed a priori | **0.765 ± 0.013** | 0.500 |
| AUC comparing only kernels from the same dish | **0.813 ± 0.014** | 0.500 |
| AUC, cluster-bootstrap 95% CI | **[0.644, 0.843]** | — |
| Average precision, never-germinated class | 0.419 | 0.110 = class prevalence |
| Permutation floor (labels shuffled between dishes) | 0.469 | — |
| Balanced accuracy | 0.661 | 0.500 |

Three results carry the report:

1. **The signal is in the grain, not the plate.** Comparing only kernels that
   shared a petri dish — where no plate-level artefact can help — performance is
   *higher* than pooled (0.813 vs 0.765). This was the outcome in doubt, because
   unsupervised analysis showed the dish is the single largest source of
   spectral variation in the data.
2. **Imbibition is what makes it work.** Dry seed at 0 h gives AUC 0.60–0.62.
   The same kernels eight hours after wetting give 0.74–0.82. The gap holds in
   both optical modes and at both preprocessing settings, and an independent
   unsupervised analysis reached the same conclusion.
3. **One variety fails completely.** Per-variety AUC is 0.98 / 0.78 / 0.82 and
   **0.49** — the fourth (`unknown`) is at chance. The pooled number is an
   average over cultivars the model treats very differently and must never be
   quoted alone.

A fifth variety, `prospect2`, was dropped from all modelling before any of the
above: 110 of its 110 kernels never germinated, so it carried no within-variety
contrast and would have let a model score well by recognising a plate.

---

## 2. Glossary

Terms used precisely throughout. Nothing here is standard vocabulary that can be
assumed.

| Term | Definition |
|---|---|
| **kernel** | One barley seed. The unit of prediction. |
| **dish** | One petri dish holding up to 22 kernels in a fixed lattice of wells. 25 dishes, numbered 0–24. |
| **well** | A moulded position in the dish that holds one kernel. Named `R<row>C<col>` on a 4-row × 7-column lattice. |
| **kernel_uid** | Global kernel name, e.g. `dish7_R2C6`. Refers to the same physical seed in every image of the collection. |
| **variety** | Barley cultivar. Five, five consecutive dishes each. |
| **side** | Which face of the dish was imaged: `dorsal` or `ventral`. The plate is double-sided and physically turned over. |
| **mode** | Optical geometry: `reflectance` (light reflected off the seed) or `transmittance` (light passed through it). |
| **capture** | One line scan = one (mode, day, dish, side) combination. 300 exist. |
| **view** | One kernel as seen in one capture. A kernel has up to 8 views: 2 modes × 2 sides × 2 capture times. |
| **cube** | A corrected capture as a 3-D array, `(640 spatial pixels, n_lines, 224 spectral bands)`. |
| **band** | One spectral channel of the camera; 224 in total. |
| **patch** | A fixed 128 × 64 image of one kernel's well, cut from a cube, with all 192 kept bands. |
| **mask** | A hand-drawn outline of one kernel, as a boolean image. |
| **never-germinated** | A kernel that was scored at every visit and never observed germinated. The minority class, 11%. Not "dead" — it means "had not germinated by the last visit". |
| **scoring visit** | One occasion on which every dish was photographed and read for germination. Five of them. |
| **t0** | The moment a dish was wetted. Per dish, not global. |
| **pooled AUC** | Area under the ROC curve over all kernels together. Chance = 0.5. |
| **within-dish AUC** | AUC computed only over pairs of kernels from the same dish, then averaged. No plate-level artefact can contribute. |
| **permutation floor** | The same pipeline run with labels shuffled between whole dishes. What the procedure scores when the label carries no information. |
| **SNV** | Standard normal variate: each spectrum has its own mean subtracted and is divided by its own standard deviation. Standard scatter correction for near-infrared. |
| **PLS-DA** | Partial least squares discriminant analysis: PLS regression against class indicators, the standard workhorse for NIR spectra. |

---

## 3. The experiment

**The plate.** Each dish is a rigid 4 × 7 lattice of 28 cell positions. Four
corner positions are absent (the dish rim cuts them off) and two carry printed
ArUco fiducial markers rather than seeds, leaving **22 kernel wells per dish**.

The plate is double-sided and is physically flipped to image the other face.
Cell (3,3) carries fiducial id0 on the dorsal face and id3 on the ventral; cell
(2,6) carries id1 and id2. These are different codes, not mirror images, so a
capture states which face it is, and that is checked against the folder it was
filed in.

**Scope.**

| | |
|---|---|
| dishes | 25 (0–24) |
| kernel wells per dish | 22 |
| wells that never held a kernel | 2 (`dish13_R0C4`, `dish21_R3C1`) |
| **kernels in the collection** | **550 − 2 = 548** |
| varieties | 5, in five consecutive dishes each |

| variety | dishes | note |
|---|---|---|
| prospect1 | 0–4 | |
| prospect2 | 5–9 | **excluded** — 110/110 never germinated |
| laureate1 | 10–14 | |
| laureate2 | 15–19 | |
| unknown | 20–24 | cultivar identity not recorded |

`prospect1`/`prospect2` and `laureate1`/`laureate2` share cultivar names but are
four distinct varieties, not replicates.

**Variety is perfectly confounded with dish.** Every kernel of a variety sits in
five consecutive dishes and nowhere else. Nothing in this data can separate "the
model recognises the cultivar" from "the model recognises those five plates".
This single fact drives most of the methodological choices below.

**Capture schedule.** Three imaging sessions, each covering all 25 dishes in both
modes and from both faces.

| folder name | meaning | in the modelling dataset? |
|---|---|---|
| `day1` | **0 h** — dry seed, imaged *before* wetting | yes |
| `day9` | **8 h** after wetting | yes |
| `day2` | **24 h** after wetting | no — captured and corrected, never annotated for modelling |

The folder names are slot labels chosen at the rig, not elapsed days. `day9` is
the 8-hour session. Nothing downstream sorts on the folder name; every stage
joins on hours or on `kernel_uid`.

**Germination assay.** After the imaging, dishes were kept wetted and
photographed once a day for five days. A human read each photo and recorded, per
kernel, the first visit at which it had germinated, or `-1` for never.

---

## 4. Instrument and acquisition

A **Specim push-broom hyperspectral camera**. Push-broom means it images one
spatial line at a time and builds a 2-D scene by moving the sample past the
sensor on a translation stage. One frame is **640 spatial pixels × 224 spectral
bands**; a capture is a few thousand such lines stacked along the scan axis.

Pixels arrive as **Mono12Packed**: two 12-bit pixels packed into three bytes.
The packing codec (`loadstich/hsi_save_load.py`) is the single source of truth
for that bit layout and is deliberately not reimplemented anywhere else.

**Spectral range is an assumption, not a factory calibration.** The 224 bands
are treated as linearly spaced over **900–1700 nm**. No per-band wavelength
calibration file exists anywhere in the project. Every wavelength quoted in this
report inherits that assumption; band *indices* are exact, nanometre values are
not.

**Acquisition loop.** Per capture the operator sets four variables (day, mode,
dish, side) and runs one script. All four are validated before anything is read
— a mistyped side would file a dorsal scan as ventral and nothing downstream
could ever detect it. The run writes the corrected cube, a raw archive, a
preview, and a `capture_meta.json` recording every configuration value that
produced it, then clears the input directory so the next dish can go in. The raw
input is cleared **last**, so a failed run leaves the capture intact and can
simply be repeated.

The raw archive is one `capture.bin` per capture — the camera's own packed lines
concatenated in scan order, verified byte-for-byte identical to the per-line
files it replaces — plus a JSON index recording the line count, frame geometry
and the original filenames, whose embedded per-line timestamps concatenation
would otherwise discard.

**Result: 300 captures** = 2 modes × 3 sessions × 25 dishes × 2 sides.
Corrected cubes are float32, roughly 0.5 GB each, ~163 GB in total. The scan-axis length differs per capture because it depends on how long the
stage ran: across the 25 day-1 dorsal captures it spans 927–938 lines in
reflectance and 1009–1110 in transmittance.

---

## 5. Radiometric correction

Two modes, two different corrections, sharing a calibration-resolution and
caching layer. Both are per (spatial pixel, band): every sensor pixel gets its
own dark offset and its own gain, broadcast along the scan axis only.

### 5.1 Transmittance

    T = (raw − D_long) / (W − D_short) × (t_short / t_long)

- `D_long` — dark frame at the **sample's** exposure (60 ms).
- `W`, `D_short` — the dedicated white line scan and its matched dark, both at
  the **white's** exposure (2.1 ms).
- `t_short / t_long` — the exposure ratio. **Required, not optional.** The
  sample and white are shot at different exposures, so without it numerator and
  denominator are in different units and T comes out ~29× too large. An earlier
  version omitted it and reported 15% of pixels above a transmittance of 1.0.
  It is a single scalar: it sets the units and cannot distort the spatial flat
  field, the spectral shape, or one kernel relative to another.

Exposures are read from the calibration folder names (`60k` → 60000 µs, `2100` →
2100 µs). If either cannot be determined the run stops rather than assuming 1.0.

**The end-to-end check: an open beam must read T = 1.** A light path with
nothing in it is the only test of dark, white and exposure factor *together*. It
is gated to bands that clip almost nowhere, because a clipped beam reads 4095
regardless of how bright it really was — measured behaviour is T = 1.07 at 0%
clipped, 1.00 at 0.9%, 0.88 at 7%, 0.31 at 28%. Ungated, the check would fail
for a reason unrelated to the correction.

**Band dropping.** The lamp only usefully illuminates ~930–1620 nm; at 1700 nm
it delivers 3% of peak and at 900 nm about 7%. Transmittance divides by the
lamp, so outside that range the correction divides a near-constant floor by a
number heading for zero and every spectrum turns sharply upward at both ends —
arithmetic, not signal. Bands below 50% of peak lamp output are set to NaN,
which keeps bands 8–200.

**Bad-pixel fill.** A (spatial row, band) cell with no usable white is NaN for
every line of the scan and punches a hole through every spectrum from that row.
Isolated defects are interpolated from the eight neighbours at (row ± 1,
band ± 1). A wide contiguous dead stripe is **not** filled — a cell in the
middle of 19 dead rows has no valid neighbour and averaging across the stripe
would invent data. What was measured and what was filled are both recorded.

### 5.2 Reflectance

    reflectance = (raw − D) / (W − D)

There is no dedicated white capture in this mode. The white reference is **two
teflon tape strips placed in the frame with the dish**, so both halves of the
ratio come from the same capture and the white is measured per spatial row
rather than assumed flat.

1. **Dark frame** — the *mean* frame over the dark capture's lines. (Mean, not
   the median transmittance uses: this is what the pipeline was validated
   against, and changing the collapse would move every value for no measured
   gain.)
2. **Dark subtraction** — `clip(raw − D, 0)`. Unlike transmittance this *does*
   clip negatives: the reflectance scene is bright everywhere, so a negative is
   noise about zero rather than weak signal.
3. **Tape detection** — the strips are captured at the start and end of the
   scan, so they are bright blobs near the two ends of the scan axis. Each blob
   is irregular and does not span the full 640 px, so a two-stage locate/refine
   recovers each blob's true per-row extent. Everything outside the blobs' span
   is pre/post-scan padding and is cropped, which also brings the cube down from
   ~1.9 GB to ~0.5 GB.
4. **Per-row white** — the 75th percentile of that row's dark-subtracted tape
   pixels. Rows holding less than 75% of the median tape count are **not**
   trusted: a blob tapers over several rows and those boundary pixels are part
   background, so they read dim, and taking them as known white made reflectance
   overshoot to 1.33 on a scene the rest of the frame put at 1.00. Untrusted
   rows are interpolated from the interior instead.

**A known, unrecoverable defect.** The teflon tape is the white reference, and
it clips. On the capture the pipeline was validated against, 27.9% of tape
voxels are railed at the sensor maximum across bands 19–130 (peak 76%); the
dish-0 day-1 capture in this collection reports 28.6% over bands 21–130 (peak
79% at band 76). A clipped tape pixel makes `(W − D)` read low and therefore
makes **reflectance read high** at those bands, for every kernel in the frame.
This cannot be fixed after capture — it needs a shorter exposure at collection
time. Every run reports it; it fired, and the data was collected anyway.

### 5.3 Scan-axis geometry

A push-broom scan is stretched along the scan axis by however much faster or
slower the stage moved than the frame rate. In-scene checkerboards of known cell
geometry give the factor that undoes it. Reflectance carries two boards in the
frame, measured independently and pooled — boards on different planes disagree,
and that disagreement is a lower bound on the correction's error off the plane
it was measured on. Transmittance measures one shared board per calibration set.

Two things the measurement cannot determine are explicit configuration values
rather than silent assumptions: the target's true cell aspect, and the fact that
the kernels do not sit on the boards' plane. Both are echoed in every run log.

### 5.4 Provenance and caching

Calibration frames resolve to a shared set or a day-specific override, and every
run logs which it used, with file count and newest modification time — a
silently stale dark is invisible in the output and ruins a day of data. Frames
and the geometry factor are cached under a key derived from the *contents* of
the calibration directory (name, size, mtime of every file), so any change
invalidates the cache and a stale frame cannot be reused silently.

The corrections were validated as **bit-identical** to the earlier dry-run
scripts they replaced, on the same captures, before being adopted.

---

## 6. Geometry and kernel indexing

Corrected cubes are not enough: a kernel has to be *named*, and the name has to
mean the same physical seed across days, modes and both faces of the plate. That
is `gridfit`.

The plate is a rigid lattice, so the two ArUco fiducials pin it: detect the
fiducials, solve the plate pose, and project the full 4 × 7 lattice. Because the
plate is physically turned over, the ventral view is a **mirror** of the dorsal
one; the pose search is allowed to reverse handedness so `R2C6` names the same
well from either side.

The lattice is regenerated in full rather than detected well by well, so a cell
clipped by the rim or hidden under a fiducial still gets placed and flagged
rather than silently dropped.

**A fit may refuse.** Each capture's fit is `pass`, `needs_review` or `fail`, and
`fail` means the cells must not be used. A lattice off by one cell still draws a
clean, convincing grid while attributing every spectrum to the wrong kernel — so
the fit is allowed to say no. **All 300 captures in this collection fit `pass`.**

An independent consistency pass then checks each dish's captures against one
another: fiducial ids must name the same face every time, every dorsal capture
in the collection must come out the same handedness, the lattice pitch must not
move, and each well's contents must match the same well in that dish's other
captures better than any relabelling of the plate would. That last check uses no
fiducial at all, so it is independent evidence about the labelling rather than a
restatement of it.

---

## 7. Annotation

Kernel outlines are hand-drawn, not detected. One preview PNG per capture was
generated and annotated with a SAM-assisted tool, exported as COCO.

**The frame contract.** The preview is the capture's own array indexing,
untouched: `png[i, j] ↔ cube[i, j, :]`, both `(640, n_lines)`. A mask decoded
from the PNG therefore indexes the cube with no rotation, flip, transpose or
rescale. This is deliberately *not* gridfit's frame, which rotates every plane
90° so the dish reads upright.

The contract is **measured, not asserted**: on every run the preview is rebuilt
from the cube in cube layout, by its own independent code path, and required to
be bit-identical. A rotation, transpose, flip, sub-pixel shift, rescale or
wrong-capture mixup all fail it. (It once failed on a single pixel of one
capture — float64 versus float32 accumulation of the band mean landing on a
grey-level boundary, not a frame error.)

**Export:** 300 images, **5,456 masks**, one category (`KERNEL`).

Three properties of this particular export needed handling. Every annotation
carries `iscrowd: 1`, which most COCO tooling silently drops — normalised to 0
on load. Segmentations are uncompressed RLE, not the compressed form
`pycocotools` expects. And 7.5% of masks carry a detached speck a few dozen
pixels across, left over from the interactive tool; only the largest connected
component is kept, and what was dropped is recorded per mask rather than
discarded.

---

## 8. The modelling dataset

`build_dataset.py` turns cubes + masks + lattice into arrays a model can index.

**Assignment must be a bijection.** Every mask is assigned to exactly one
lattice cell by pixel overlap, and the build refuses to continue unless that
assignment is one-to-one. Get it wrong by one cell and every spectrum is
attributed to the wrong kernel while every overlay still looks perfect. Across
all 4,381 in-scope masks: zero off-lattice, zero straddling two cells, zero
double assignments, median containment 1.000, minimum 0.9915.

**One row per view.** Each row is one kernel in one capture.

| | |
|---|---|
| in-scope captures (`day1` + `day9`, both modes, both sides) | 200 |
| × 22 wells | 4,400 |
| − wells with no mask | 19 |
| **rows** | **4,381** |

The 19 gaps are exactly accounted for: the two permanently empty wells appear in
all 8 of their views (16), plus three kernels the annotator missed in exactly one
view each (`dish13_R3C2`, `dish3_R1C2`, `dish8_R2C0`).

**What is stored.** Each row is warped onto a fixed **128 × 64 patch** over
**192 bands** (indices 8–199, ≈ 928.7–1613.9 nm — the range valid in every
capture). A homography from the cell's four corners does four jobs at once: it
removes the plate's rotation, removes the dorsal/ventral mirror, puts the two
modes on a common grid, and makes an 0 h patch pixel-comparable with an 8 h one.
The mask goes through the *same* homography, so patch and mask are registered by
construction rather than by assumption.

Stored values are per-pixel **pseudo-absorbance**, `A = −log10(x)`, floored at
1e-4 (corrected transmittance reaches −0.0005 and a reflectance voxel can be
exactly 0 after dark subtraction; both make the logarithm undefined).
Saturated voxels are invalidated *before* the logarithm — a railed voxel carries
no information about how bright it really was.

SNV is deliberately **not** baked in. It is applied at load time, so changing
preprocessing is a flag rather than an hour-long rebuild.

Outputs: `patches.npy` (4381 × 128 × 64 × 192 float16, 13.78 GB), `masks.npy`,
`spectra.npy` (the mask-mean spectrum per view), and `index.csv`.

**The gate.** `verify_dataset.py` runs 24 checks and nothing trains unless all
pass. They exist because each failure is invisible downstream — a mirrored
patch, a mask attributed to the neighbouring well, or a spectrum that no longer
matches the pixels it summarises all produce a clean loss curve and a plausible
confusion matrix. Among them: the RLE decoder reproduces the exporter's own area
and bbox; the cube slice agrees with gridfit's reference implementation; stored
cell assignments match a fresh re-derivation; spectra reproduce *exactly* from
patches and masks; no warped mask touches the patch border; and no kernel or
dish spans train and test. **All 24 pass.**

---

## 9. Labels and the measured clock

### 9.1 Three states, not two

The scoring sheet has one row per dish and one column per kernel index 0–21.

| cell | means | treatment |
|---|---|---|
| `1`–`5` | first seen germinated at that scoring visit | an interval — see below |
| `-1` | scored, never germinated | a real observation — **kept** |
| blank | that dish had not been scored yet | a missing label — **dropped** |

Reading blank as `-1` would train the model to predict dormancy from unscored
plates; reading `-1` as blank would discard the kernels the whole question is
about. All 548 kernels are now scored, so no blanks remain, but the parser still
enforces the distinction.

**Distribution across all 548 kernels:** visit 1 → 101, visit 2 → 236, visit 3 →
33, visit 4 → 12, visit 5 → 8, never → 158. (The 158 includes prospect2's 110.)

### 9.2 A label is an interval, not a time

"Visit 2" means the kernel was seen ungerminated at visit 1 and germinated at
visit 2. That is a bracket, not an event time.

### 9.3 The nominal clock is wrong, and the real one is per dish

The protocol says captures at 0/8/24 h and scoring at 24/48/72/96/120 h. Neither
is what happened. The real timeline was reconstructed from two independent
sources — file modification times pulled off the acquisition machine, and EXIF
`DateTimeOriginal` on the scoring photographs.

**There is no single t = 0.** The day-1 session took **8.3 hours** to work
through 25 plates: dish 0 was scanned at 10:55 and dish 24 at 19:14. Each dish's
zero is therefore its own dry, pre-wetting scan, and everything else is measured
from that. The per-dish lags are stable precisely because the operator worked
the plates in the same order every time.

| clock | nominal | measured (mean) | range across dishes |
|---|---|---|---|
| first capture | 0 h | 0.02 h | 0.00 – 0.22 |
| second capture | 8 h | **7.59 h** | 6.91 – 8.95 |
| scoring visit 1 | 24 h | **22.61 h** | 21.37 – 25.99 |
| scoring visit 2 | 48 h | 48.39 h | 45.43 – 53.39 |
| scoring visit 3 | 72 h | 72.02 h | 69.02 – 77.05 |
| scoring visit 4 | 96 h | 95.65 h | 92.65 – 100.77 |
| scoring visit 5 | 120 h | **123.69 h** | 120.93 – 128.58 |

So germination intervals are **21.4–28.2 h wide** (median 24.5), and a
never-germinated kernel is censored at its own dish's last visit — **120.9 to
128.6 h**, not 120.

Two caveats. `t0` is the dry scan, which happened immediately *before* wetting,
so it is a lower bound on the wetting moment by however long it took to carry
the plate to the bench. And transmittance capture times were never collected:
those views inherit the reflectance timestamp for the same (day, dish, side),
flagged as a proxy in the data. The error is bounded by one dish slot, ~14 min —
a dish's two reflectance scans are ~1.2 min apart while consecutive dishes are
~13.7 min apart — which is immaterial against a 7.6 h offset and 21–28 h
brackets.

This clock is recorded in `timing.json`, which is a **hand-acquired input, not a
build product**: the tooling that collected it has been removed, and rebuilding
it means returning to the acquisition machine.

---

## 10. Excluding prospect2

**All 110 of prospect2's kernels never germinated.** It is excluded from every
model, split and figure.

It was dropped rather than kept as a class because a variety with a single
outcome cannot teach anything about germination and actively corrupts what it is
asked. It is a fifth of the data sitting entirely on one side of the target, and
since variety is confounded with dish and folds hold out whole dishes, a model
that learned nothing but "is this prospect2?" would score well on germination
*and survive cross-validation*. Measured before the exclusion: pooled
never-germinated balanced accuracy 0.868, but only 0.656 within the mixed
varieties — a 0.21 gap that was plate recognition.

The exclusion is applied in exactly one place (`Index.load`) that every trainer,
split builder and figure passes through, and it announces what it dropped rather
than doing so silently.

**Scope after exclusion: 438 kernels, 3,502 views, 20 dishes, 4 varieties.**
Of the 438, **390 germinated and 48 never did — an 11.0% minority.**

---

## 11. What the data looks like before modelling

### 11.1 Germination timing

Among the 438 kernels in scope, 89% germinate, with a sharp mode at the second
scoring visit: 101 kernels by ~22.6 h, 236 more by ~48.4 h, then a thin tail of
33 / 12 / 8, and 48 never.

The four varieties differ in **timing, not rate**. Germination rates are flat at
87–91%. But `unknown` has only 4% up at the first visit against 26–33% for the
other three, then catches up completely by 72 h — a delayed onset of roughly
5 hours, not a different process.

### 11.2 Unsupervised structure (PCA)

Principal component analysis was run on the mask-mean spectra, one row per
kernel, separately per mode and capture time (reflectance and transmittance have
different radiometry; combining them makes the first component "which camera").
Three components carry 93–98% of the variance.

The decisive measurement is how much of each component's variance sits *between*
groups. Because variety is confounded with dish, the informative comparison is
dish against variety, and both again after removing every between-variety
difference:

| | dish | variety | germinated | dish \| variety | germ. day \| variety |
|---|---|---|---|---|---|
| reflectance 0 h | 0.575 | 0.347 | 0.044 | **0.369** | 0.022 |
| reflectance 8 h | 0.388 | 0.258 | 0.041 | 0.189 | **0.155** |
| reflectance 8 h − 0 h | 0.506 | 0.170 | 0.009 | **0.405** | 0.076 |
| transmittance 8 h | 0.230 | 0.162 | 0.081 | 0.200 | 0.079 |

Four things follow, and all four shaped the modelling:

- **The plate is the largest single source of spectral variation.** Dish beats
  variety everywhere, and the pure plate effect (after removing variety) reaches
  0.41. This is the standing threat to every result in the project.
- **There is no unsupervised germination axis.** Germinated-vs-never never
  exceeds 0.081 of any leading component. This is expected — PCA has never heard
  of germination — and is *not* evidence that germination is unpredictable.
- **Reflectance at 8 h is the one cell with germination signal** that survives
  the variety control (0.155). Nothing else is close.
- **Differencing 8 h − 0 h was a dead end.** It looked like the natural
  imbibition contrast and is the *most* plate-dominated view of the six:
  differencing cancels the kernel and keeps whatever drifted between sessions.

A second PCA over individual voxels found that **48–58% of all reflectance voxel
variance lives in the outermost two pixels of each mask** — partial-volume
mixing of grain and well at the mask boundary, visible as a ring on the score
maps. Transmittance loses only 14–15% to the same test. This is a measured case
for eroding the masks, which has not yet been acted on because it requires a
full dataset rebuild.

---

## 12. Modelling: germination

### 12.1 Setup

**Target.** Will this kernel germinate by the last scoring visit? One bit per
kernel. 390 / 48.

**Unit of prediction is the kernel, not the view.** A kernel has up to 8 views,
which are correlated repeat looks at the same seed; treating them as independent
rows would inflate every count and every significance test eightfold. Views are
averaged over the two **faces** only — never across modes (different radiometry)
and never across capture times (0 h is a dry seed and 8 h an imbibed one, two
physical states). Each candidate representation is therefore exactly one row per
kernel: **438 × 192**.

**Model.** PLS-DA. At 438 samples and 192 heavily collinear predictors this is
the model to beat; the component count is the regularisation.

**Preprocessing.** SNV, then optionally a first Savitzky–Golay derivative.
Neither fits any parameter across rows, so neither can leak between folds.

**Cross-validation holds out whole dishes** — five folds. A kernel-level split
would let the model memorise the plate and score near-perfectly while learning
nothing about barley.

**Folds are stratified on germination rate *and* variety.** This was a fix: rate
alone produced a fold holding three dishes of one variety and zero of two
others. A Latin square now assigns, within each variety, the rate-ranked dishes
to folds with a per-variety offset, so **every fold holds exactly one dish of
each of the four varieties** (22 kernels each) with 7–12 never-germinated. The
same scheme is used for the inner folds.

**Three things are chosen inside each training fold, never on the test fold:**
the representation (mode × capture time × derivative), the component count — by
inner AUC, which is threshold-free and so is not entangled with an operating
point chosen on the same data — and then the decision threshold, by inner
balanced accuracy. The threshold matters as much as the model: PLS-DA's natural
argmax is effectively a 0.5 cut, which on an 89/11 split says yes to nearly
everything.

The whole outer cross-validation is **repeated 5×** with fresh stratified splits.

### 12.2 Results

| | nested (procedure) | representation fixed a priori |
|---|---|---|
| AUC | **0.716 ± 0.025** | **0.765 ± 0.013** |
| within-dish AUC | 0.761 ± 0.032 | **0.813 ± 0.014** |
| within-variety AUC | 0.745 | 0.767 |
| permutation floor | 0.491 | 0.469 |
| AP (never-germinated) | 0.432 | 0.419 (chance = 0.110) |
| balanced accuracy | 0.641 | 0.661 |
| sensitivity / specificity | 0.656 / 0.625 | 0.697 / 0.625 |

The **nested** column lets the model choose its own representation inside each
fold and is unbiased with respect to that choice. The **fixed** column pins it
to reflectance at 8 h with no derivative.

**Fix the representation; do not let the model choose it.** The nested selector
is *worse* — 0.716 against 0.765. With only 16 training dishes the inner AUC
estimate cannot reliably choose among fourteen candidates: it took `refl_8h` in
three folds and the 768-column combined representation in two, and those two
folds scored 0.614 and 0.844. Fixing the representation a priori costs no test
information, because the evidence for it is not a test score: the 0 h/8 h gap is
systematic across all four mode × derivative combinations, and the PCA reached
the same conclusion unsupervised. The derivative makes no difference (0.765 vs
0.768).

**Imbibition is the effect.**

| representation | AUC | within-dish AUC |
|---|---|---|
| reflectance 8 h, deriv 1 | 0.768 ± 0.008 | 0.814 |
| reflectance 8 h, deriv 0 | 0.765 ± 0.013 | 0.813 |
| transmittance 8 h, deriv 1 | 0.752 ± 0.020 | 0.774 |
| transmittance 8 h, deriv 0 | 0.741 ± 0.014 | 0.772 |
| transmittance 0 h, deriv 1 | 0.622 ± 0.014 | 0.642 |
| reflectance 0 h, deriv 1 | 0.610 ± 0.012 | 0.643 |
| reflectance 0 h, deriv 0 | 0.603 ± 0.015 | 0.637 |
| transmittance 0 h, deriv 0 | 0.610 ± 0.011 | 0.630 |

Every 8 h representation beats every 0 h one by 0.10–0.17 AUC. Combining blocks
hurts: 768 columns for 438 rows overfits.

### 12.3 Controls

- **Within-dish AUC exceeds pooled AUC** (0.813 vs 0.765). Comparing only
  kernels that shared a plate, where no plate-level artefact can contribute,
  performance is *higher*. The model is reading grain, not dish.
- **Permutation floors sit at 0.467–0.522** across all fourteen
  representations. The pipeline is not leaking.
- **Kernel size predicts nothing** — mask area alone gives AUC 0.546.
- **Cluster bootstrap over dishes**: AUC 0.741, 95% CI **[0.644, 0.843]**.
  Above chance beyond doubt; above 0.7 is *not* established.

### 12.4 The per-variety result

| variety | never | AUC | within-dish AUC | 95% CI (pooled) |
|---|---|---|---|---|
| prospect1 | 10 | **0.983** | **0.990** | [0.969, 1.000] |
| laureate1 | 13 | 0.782 | 0.815 | [0.705, 0.849] |
| laureate2 | 14 | 0.817 | 0.806 | [0.657, 0.960] |
| **unknown** | 11 | **0.486** | 0.534 | **[0.324, 0.628]** |

`prospect1`'s never-germinated kernels are almost perfectly separable, and the
within-dish figure of 0.990 shows this is not a plate effect. `unknown` is at
chance and its interval sits squarely across 0.5.

A quarter of the data is dragging the headline down while three quarters does
better than it looks. `unknown` is also the variety with the ~5 h longer
germination lag. Whether those are the same fact is the most interesting open
question in the project.

### 12.5 Practical framing

AUC is not directly actionable. Ranking the kernels worst-first and flagging the
bottom slice:

| flag the worst | never-germinated found | precision | vs 11% base rate | recall |
|---|---|---|---|---|
| 5% (20 kernels) | 14 | 0.70 | **6.4×** | 0.29 |
| 9% (40) | 17 | 0.42 | 3.9× | 0.35 |
| 14% (60) | 21 | 0.35 | 3.2× | 0.44 |
| 20% (88) | 23 | 0.26 | 2.4× | 0.48 |

"Discard the worst 5% and 70% of what you discard genuinely never came up" is a
claim that can be acted on. Per-seed condemnation is not supported: at the
balanced operating point, precision on the never-germinated call is ~20–26%.

### 12.6 A signal the model is currently throwing away

Single-number baselines were tested. Kernel size gives 0.546 — nothing. But the
**standard deviation of the raw absorbance spectrum at 8 h gives 0.697 pooled /
0.721 within-dish on its own.**

SNV removes exactly that: it divides each spectrum by its own standard deviation
before the model sees it. So the model cannot be using this — it reaches 0.765
from spectral *shape* alone with that information discarded. Physically the
magnitude of absorbance is plausibly water uptake, which is what an imbibing
seed does and a non-viable one does not. Two partly independent signals, one
currently unused; restoring the pre-SNV scale as extra features is the cheapest
available improvement.

### 12.7 Choosing metrics on an 89/11 problem

Three widely used metrics rank this model *below* a model that answers "yes" to
everything: accuracy (0.689 vs 0.890), F1 on the germinated class (0.800 vs
0.942) and sensitivity (0.697 vs 1.000). Average precision on the *majority*
class is equally useless — its chance level is 0.89.

What is reported instead: **AUC** (threshold-free, chance 0.5 regardless of
prevalence), **average precision on the minority class** quoted against its
0.110 prevalence, **sensitivity and specificity as a pair** rather than merged,
and enrichment for the practical case. Matthews correlation coefficient (0.213,
chance 0) is the defensible single scalar if one is demanded.

With 48 never-germinated kernels, specificity moves in steps of 1/48 = 2.1
percentage points; differences smaller than that are not differences.

---

## 13. Modelling: variety

The second track — identify the cultivar from the spectra — exists and runs, but
is **a baseline only**: no tuning, no controls, no CNN.

Four classes after the prospect2 exclusion, chance 0.250. PLS-DA on reflectance,
dish-grouped folds: **0.685 per-kernel balanced accuracy** (0.627 per view). The
confusion is dominated by the laureate pair, which is expected — `laureate1` and
`laureate2` are related cultivars.

This number should be treated with more suspicion than the germination result,
because variety is perfectly confounded with dish and no within-dish control is
possible for a target that is constant within a dish. The dish-identity control
task exists for exactly this reason and has not been run.

---

## 14. Limitations

Ordered by how much they should temper the conclusions.

1. **One experiment.** One growth run, one plate design, one rig, one operator.
   Nothing here has been validated on an independent batch, and no held-out test
   set exists — every number is cross-validated.
2. **48 never-germinated kernels.** The entire minority class. All germination
   conclusions rest on them.
3. **One variety fails.** `unknown` is at chance. Whether the method generalises
   to a new cultivar is genuinely unknown, and the one variety here that was not
   pre-selected is the one it fails on.
4. **The reflectance white reference clips**, over roughly bands 21–130 at up to
   79%, making reflectance read high in that range for every kernel. Not
   recoverable after capture. Reflectance at 8 h is also the configuration
   carrying the result — the two facts coexist and the interaction has not been
   quantified.
5. **Wavelengths are assumed**, not calibrated. Band indices are exact;
   nanometre values inherit a linear 900–1700 nm assumption.
6. **The plate effect is large.** Within-dish AUC controls for it in the
   germination result. Nothing controls for it in the variety result.
7. **Mask boundaries contribute half the reflectance voxel variance**, and
   erosion has not been applied.
8. **Selection.** The representation was fixed using evidence from this same
   dataset (unsupervised, and systematic across configurations, but the same
   dataset). The nested number, 0.716, is the estimate that carries no such
   caveat.
9. **`day2` (24 h) was captured and corrected but never annotated**, so a third
   of the imaging is unused.
10. **Transmittance capture times are a flagged proxy**, bounded at ~14 min.

---

## 15. Reproducing this

The repository tracks code and small irreplaceable inputs — 159 files, ~10 MB.
It does not track capture data.

**Not in the repository:**

| | size | needed for | regenerable |
|---|---|---|---|
| `real_data/` — corrected cubes | 163 GB | building the dataset, the gate, mask review | **no** |
| `modeling_pipeline/dataset/` | 13 GB | all modelling | yes, from the cubes, ~4 min |
| `grid_view/` — the fitted lattice | 348 MB | building the dataset | yes, from the cubes |
| `preview_coco/` — annotation renders | 75 MB | nothing at runtime | yes |
| `labels/germination_photos/` | 464 MB | nothing now — EXIF already extracted | **no** (primary evidence) |

**In the repository and irreplaceable:** the CVAT mask export
(`instances_default.json`, 5,456 hand-drawn masks), the scored labels
(`germination_label_annotator.xlsx`), and `timing.json`.

**The shortcut:** copying `dataset/` (13 GB) is enough for all modelling — both
trainers, both PCA scripts and the label histograms read only that, `timing.json`
and the label sheet. Only three scripts touch raw cubes. On that setup
`verify_dataset.py` cannot run, which should be stated in anything reported from
it.

Python 3.12, two interpreters by design: the system interpreter (numpy, scipy,
opencv, matplotlib) runs the pipeline and exploratory scripts; a virtualenv
adding scikit-learn and torch runs the trainers.

---

## 16. Repository map

```
germination_pipeline/          the repository root
PROJECT_REPORT.md            this document
CLAUDE.md                    orientation for agentic tooling
preprocessing_pipeline/
  collection_pipeline/       THE production path
    config.py                four per-capture variables + rig configuration
    process.py               one capture, start to finish
    common.py                stitching, raw archive, calibration cache, geometry
    reflectance_scripts/     the reflectance correction
    transmittance_scripts/   the transmittance correction
    gridfit/                 plate model, fiducial detection, lattice fit
    grid_index.py            fit every capture -> grid_view/
    loadstich/               the Mono12Packed codec
  dryrun_scripts/            the earlier pipeline the above was validated against
  <other>/                   superseded exploratory pipelines, kept for reference
modeling_pipeline/
  build_dataset.py           cubes + masks + lattice -> dataset/
  verify_dataset.py          the 24-check gate
  train_germination.py       the binary germination model
  train_pls.py               PLS-DA, the variety track
  train_cnn.py               ResNet-18 with a spectral stem, either track
  barley/                    library: index, labels, timing, splits, metrics, models
  explore/                   mask review, label histograms, the two PCA scripts
  reporting/                 figure style and report assembly
  timing.json                the measured clock (input, not a build product)
  germination_label_annotator.xlsx    the scored labels (input)
  annotations/               the COCO mask export (input)
  exploration/               finished figures + their summary.md files
```

Every generated folder carries a `summary.md` restating every number in its
figures. The most useful entry points for a reader are
`modeling_pipeline/reports/germ5/summary.md` (the germination model),
`modeling_pipeline/exploration/pca/mean_spectra/summary.md` (the confound
analysis) and `modeling_pipeline/exploration/germination/summary.md` (the
labels).
