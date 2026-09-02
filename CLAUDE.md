# CLAUDE.md

`README.md` is the map and the run commands. `PROJECT_REPORT.md` is the full
account of the experiment and the results. Each half has its own README:
`preprocessing_pipeline/README.md`, `modeling_pipeline/README.md`.

This file is only the things that will bite you.

## Things that will bite you

- **`cells.json` records absolute paths** from the machine that fitted the grid.
  Read a capture directory through `config.local_capture_dir()`, never
  `Path(rec["capture"]["source"])`, or the code only works on one machine.
- **The folder names `day1`, `day9`, `day2` are rig slot labels**, not calendar
  days: they mean 0 h, 8 h and 24 h after wetting. Never sort or join on them —
  join on `hours` or `kernel_uid`. "Day" also names the five assay scoring
  visits, which is a different clock again.
- **Variety is perfectly confounded with dish.** Every fold holds out whole
  dishes, always. Any claim about variety needs a dish-level control.
- **In the label sheet, `-1` and blank are not the same.** `-1` is a real
  observation (scored, never germinated) and is kept; blank is a missing label
  and is dropped. Merging them teaches the model to predict dormancy from
  unscored plates.
- **Never put reflectance and transmittance in one matrix**, and never average
  0 h with 8 h. Different radiometry; dry seed versus imbibed seed. Averaging
  over *sides* is fine.
- **Wavelengths are assumed, not calibrated.** 224 bands treated as linear over
  900–1700 nm; no calibration file exists. Band indices are exact, nanometres
  are not.
- **The reflectance white reference clips** over roughly bands 21–130, up to
  79%, making reflectance read high there. Not recoverable after capture.
- **`train_pls.py --task germination` is not `train_germination.py`.** The
  generic trainer takes the argmax of a one-hot PLS fit — an implicit 0.5
  threshold, which on an 89/11 split says yes to nearly everything. Use
  `train_germination.py` for the binary target.
- **`config.MASK_ERODE_PX` is 1, but the prose says erosion is off.** Unresolved
  — `dataset/build_meta.json` records what the published dataset actually used
  and is the authority. Reconcile before quoting it or anything downstream.
- `.gitignore` note: git does **not** support trailing comments on a pattern
  line. A pattern written as `data/  # big` silently matches nothing, which is
  why every comment in these files is on its own line.
