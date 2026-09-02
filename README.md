# Predicting barley germination from hyperspectral images

Twenty-five petri dishes of barley — 548 kernels, four cultivars — imaged with a
Specim near-infrared push-broom camera in two optical modes, from both faces,
before wetting and eight hours after. The kernels were then grown for five days
and scored by eye. **Do the spectra taken eight hours after wetting predict
which kernels will germinate, hours before any visible change?**

Yes, on three of the four varieties.

| | value | reference |
|---|---|---|
| AUC, unbiased (nested selection) | **0.716 ± 0.025** | 0.500 = chance |
| AUC, representation fixed a priori | **0.765 ± 0.013** | 0.500 |
| AUC comparing only kernels from the same dish | **0.813 ± 0.014** | 0.500 |
| Permutation floor (labels shuffled between dishes) | 0.469 | — |

**`PROJECT_REPORT.md` is the full account** — what was collected, corrected,
labelled and modelled, every control, and what the numbers do and do not
support. Read it first. This file is the map.

## Layout

Two halves, with different conventions and different dependencies. Each has its
own README.

```
preprocessing_pipeline/   raw camera lines -> corrected cubes -> a fitted plate lattice
modeling_pipeline/        cubes + hand-drawn masks -> a dataset -> models
PROJECT_REPORT.md         the write-up
```

## Scope: two modelling tracks, and only two

| | PLS | CNN |
|---|---|---|
| **variety typing** — 4 cultivars | `train_pls.py --task variety` | `train_cnn.py --task variety` |
| **germination** — binary, 89/11 | `train_germination.py` | `train_cnn.py --task germination` |

A germination-*time* regressor was built and deleted; timing is out of scope for
models, though `barley/timing.py` still holds the measured intervals.

## The data is not in the repo

The repository tracks code plus small irreplaceable inputs — 101 files, ~10 MB.
It tracks **no capture data**.

| | size | where | regenerable |
|---|---|---|---|
| `real_data/` corrected cubes | 163 GB | beside the checkout | **no** |
| `modeling_pipeline/dataset/` | 13 GB | generated | yes, ~5 min, needs the cubes |
| `preprocessing_pipeline/grid_view/` | 348 MB | generated | yes, needs the cubes |
| `labels/germination_photos/` | 464 MB | beside the checkout | **no** — evidence for the labels |

**Tracked and irreplaceable:** `modeling_pipeline/annotations/instances_default.json`
(5,456 hand-drawn masks), `germination_label_annotator.xlsx` (the scored labels),
`timing.json` (the measured capture clock; its acquisition tooling was removed).
Never delete these and never move them into a gitignored folder.

**The shortcut:** copy `dataset/` and all modelling works without the 163 GB of
cubes. Only `build_dataset.py`, `verify_dataset.py` and `explore/review_masks.py`
read raw cubes.

## Running

```bash
# preprocessing -- one capture at a time; edit the four vars in config.py first
cd preprocessing_pipeline && python3 process.py
python3 grid_index.py                    # fit the plate lattice on every capture

# modelling
cd modeling_pipeline
python3 build_dataset.py                 # cubes + masks + lattice -> dataset/
python3 verify_dataset.py                # the 24-check gate; must be green
python3 -m barley.splits                 # variety folds
python3 -m barley.germination            # germination folds
.venv/bin/python train_germination.py    # the binary model
```

Python 3.12, two interpreters by design: the system interpreter (numpy, scipy,
opencv, matplotlib) runs the pipeline and the exploratory scripts; a virtualenv
adding scikit-learn and torch runs the trainers.

## Conventions

- One script per question, `argparse --help` on all of them, output to a folder
  carrying its own `summary.md` that restates every number in its figures.
- Anything hand-acquired and unreproducible lives at a repo root and is
  versioned. Anything a script can rebuild lives in a gitignored folder.
- Quality gates default to **off**, so what is excluded is always an explicit
  choice recorded in the run rather than a silent default.
