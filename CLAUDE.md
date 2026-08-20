# CLAUDE.md — repository root

Guidance for Claude Code working anywhere in this repository.
**`PROJECT_REPORT.md` is the human-facing account of the whole project** — read
it first for what was collected, corrected, labelled and modelled, and what the
results support. This file is the operating manual.

## What this is

Predicting barley germination from near-infrared hyperspectral images. 25 petri
dishes, 548 kernels, imaged with a Specim push-broom line-scan camera in two
optical modes from both faces, before wetting and 8 h after, then grown and
scored for germination over five days.

Two halves, and they are separate projects with separate conventions:

```
preprocessing_pipeline/    raw camera lines -> corrected cubes -> a fitted plate lattice
modeling_pipeline/         cubes + hand-drawn masks -> a dataset -> models
PROJECT_REPORT.md          the full write-up
```

**`modeling_pipeline/` has its own `CLAUDE.md` and `README.md`. Read those
before working in it** — it has a different dependency set, its own
conventions, and several settled decisions that are expensive to rediscover.

## Scope: two modelling tracks, and only two

| | PLS | CNN |
|---|---|---|
| **variety typing** — 4 cultivars | `train_pls.py --task variety` | `train_cnn.py --task variety` |
| **germination** — binary, 89/11 | `train_germination.py` | `train_cnn.py --task germination` |

A germination-*time* regressor was built and deleted; timing is out of scope for
models. Do not add a third track without being asked.

## The data is not in the repo

The repository tracks code plus small irreplaceable inputs — about 160 files,
~10 MB. It tracks **no capture data**.

| | size | where | regenerable |
|---|---|---|---|
| `real_data/` corrected cubes | 163 GB | beside the checkout | **no** |
| `modeling_pipeline/dataset/` | 13 GB | generated | yes, ~4 min, needs the cubes |
| `grid_view/` fitted lattice | 348 MB | generated | yes, needs the cubes |
| `labels/germination_photos/` | 464 MB | beside the checkout | **no** — evidence for the labels |

**Tracked and irreplaceable:** `modeling_pipeline/annotations/instances_default.json`
(5,456 hand-drawn masks), `germination_label_annotator.xlsx` (the scored labels),
`timing.json` (the measured capture clock; its acquisition tooling was removed).
Never delete these and never move them into a gitignored folder.

**The shortcut:** copy `dataset/` and all modelling works without the 163 GB of
cubes. Only `build_dataset.py`, `verify_dataset.py` and
`explore/review_masks.py` read raw cubes.

`.gitignore` note: git does **not** support trailing comments on a pattern line.
Every comment in the root `.gitignore` is on its own line for that reason; a
pattern written as `data/  # big` silently matches nothing.

## Things that will bite you

- **`cells.json` records absolute paths** from the machine that fitted the grid.
  Read a capture directory through `config.local_capture_dir()`, never
  `Path(rec["capture"]["source"])`, or the code only works on one machine.
- **Wavelengths are assumed, not calibrated.** 224 bands treated as linear over
  900–1700 nm; no calibration file exists. Band indices are exact, nanometres
  are not.
- **The folder names `day1`, `day9`, `day2` are rig slot labels**, not calendar
  days: they mean 0 h, 8 h and 24 h after wetting. Never sort or join on them —
  join on `hours` or `kernel_uid`.
- **Variety is perfectly confounded with dish.** Every fold holds out whole
  dishes, always. Any claim about variety needs a dish-level control.
- **The reflectance white reference clips** over roughly bands 21–130, up to
  79%, making reflectance read high there. Not recoverable after capture.
- **`SpecimStitcher.load_lines()` deletes its source directory on success.**
  Nothing current calls it; do not start.

## Running

```bash
# preprocessing (one capture at a time; edit the four vars in config.py first)
cd preprocessing_pipeline/collection_pipeline && python3 process.py
python3 grid_index.py                    # fit the plate lattice on every capture

# modelling
cd modeling_pipeline
python3 build_dataset.py                 # cubes + masks + lattice -> dataset/
python3 verify_dataset.py                # the 24-check gate; must be green
python3 -m barley.splits                 # variety folds
python3 -m barley.germination            # germination folds
.venv/bin/python train_germination.py    # the binary model
```

## Conventions

- One script per question, `argparse --help` on all of them, output to a folder
  carrying its own `summary.md` that restates every number in its figures.
- Anything hand-acquired and unreproducible lives at a repo root and is
  versioned. Anything a script can rebuild lives in a gitignored folder.
- Quality gates default to **off**, so what is excluded is always an explicit
  choice recorded in the run rather than a silent default.
