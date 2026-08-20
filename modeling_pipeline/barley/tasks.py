"""What we are predicting.

Two tracks, and only two:

    variety      which of the four cultivars is this kernel?   (4 classes)
    germination  will it come up at all?                        (2 classes)

plus `dish`, which is not a target but the control that says whether either of
the above is being read off the plate rather than the grain.

The modularity contract of this package: changing what is predicted changes
THIS FILE ONLY. The index, the transforms, the datasets, the splits and the
models all take a Task and never ask what it means.

A Task turns index rows into integer class labels, and says which column groups
rows that must not be split across train and test.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

import config


@dataclass
class Task:
    name: str
    classes: list                     # class_id -> human-readable name
    label_fn: Callable                # Index -> int array of class ids, -1 = unlabelled
    group_col: str = "dish"
    description: str = ""

    def labels(self, idx):
        y = np.asarray(self.label_fn(idx), np.int64)
        if len(y) != len(idx):
            raise SystemExit(f"task {self.name}: {len(y)} labels for {len(idx)} rows")
        return y

    def groups(self, idx):
        return idx[self.group_col]

    def usable(self, idx):
        """Rows with a label. Unlabelled rows are dropped, loudly, by callers."""
        return self.labels(idx) >= 0


# ------------------------------------------------------------------ variety --
# Deliberately the easy task: it exercises the whole chain -- masks, patches,
# spectra, splits, model, metrics -- with a label that needs no external file.
#
# It is also perfectly confounded with dish. Every kernel of variety 1 is in
# dishes 0-4 and nowhere else, so any dish-level artifact (plate, illumination,
# capture session) is a *perfect* variety predictor. This is why group_col is
# "dish" and why evaluate.py runs the dish-identity control: a random split here
# would score near-perfect while learning nothing about barley.
# Class ids are positions in `config.kept_varieties()`, not the raw variety
# number. An excluded variety must not leave a hole in the numbering: PLS would
# get a one-hot column that is always zero -- a class it can predict but never
# get right -- chance level would be computed off the wrong class count, and
# every confusion matrix would carry an empty row.
def _variety_label(idx):
    remap = {v: i for i, v in enumerate(config.kept_varieties())}
    return np.array([remap.get(int(v), -1) for v in idx["variety"]], np.int64)


VARIETY = Task(
    name="variety",
    classes=[config.VARIETY_NAMES[v - 1] for v in config.kept_varieties()],
    label_fn=_variety_label,
    group_col="dish",
    description=", ".join(
        f"{config.VARIETY_NAMES[v - 1]} (dishes {(v - 1) * config.DISHES_PER_VARIETY}-"
        f"{v * config.DISHES_PER_VARIETY - 1})"
        for v in config.kept_varieties())
    + (f"; {', '.join(config.excluded_names())} excluded"
       if config.EXCLUDED_VARIETIES else ""),
)


# ----------------------------------------------------------------- controls --
# Not a scientific target -- a control. If a model predicts dish about as well
# as it predicts variety, it is reading the plate, not the grain.
def _dish_label(idx):
    remap = {d: i for i, d in enumerate(config.kept_dishes())}
    return np.array([remap.get(int(d), -1) for d in idx["dish"]], np.int64)


DISH = Task(
    name="dish",
    classes=[f"dish{d}" for d in config.kept_dishes()],
    label_fn=_dish_label,
    group_col="kernel_uid",     # dish cannot group its own prediction
    description="control: can the model identify the dish itself?",
)


# -------------------------------------------------------------- germination --
def _germinated(idx):
    """Did each kernel ever germinate? -> (N,) 0/1, -1 where unscored.

    One label file, one parser: `barley.germination`. Imported lazily so this
    module has no import-time dependency on it.

    Note "day" is overloaded in this project: the capture folders day1/day9/day2
    are slots meaning 0 h / 8 h / 24 h, while the germination photos are
    calendar days 1-5 of the assay. Join on kernel_uid, never on a day column.
    """
    from . import germination
    # germination._table() rather than a fresh parse: it is cached, and its orphan
    # check is against the FULL index. Checking labels against `idx` would call
    # every kernel outside the current selection an orphan the moment a run
    # selects one dish, or config excludes a variety.
    first = germination.first_day_array(idx, germination._table())
    # -1 for an unscored kernel, which datasets._resolve drops. 0 means scored
    # and not up by `day`, which includes the never-germinators.
    return np.array([-1 if d is germination.UNSCORED else int(d is not None)
                     for d in first], np.int64)


GERMINATION = Task(
    name="germination",
    classes=["not_germinated", "germinated"],
    label_fn=_germinated,
    group_col="dish",
    description="did it ever germinate, by the last scoring visit? "
                "(measured horizon 120.9-128.6 h per dish, not the nominal 120)",
)


def restrict(task, keep):
    """A Task over a subset of the classes, with labels renumbered contiguously.

    `keep` is class ids in the original task. Rows of any other class get label
    -1, which barley.datasets drops (loudly) when it resolves the selection.

    Renumbering matters: leaving the ids sparse would give PLS a one-hot column
    that is always zero -- a class the model can still predict but never get
    right -- and leave an empty row in every confusion matrix. Chance level
    follows len(classes), so it moves on its own.
    """
    keep = sorted(int(k) for k in keep)
    bad = [k for k in keep if not 0 <= k < len(task.classes)]
    if bad:
        raise SystemExit(f"{task.name} has no class {bad}; "
                         f"valid are 0..{len(task.classes) - 1}")
    if len(keep) < 2:
        raise SystemExit("keep at least two classes")
    remap = {old: new for new, old in enumerate(keep)}

    def label_fn(idx):
        y = np.asarray(task.label_fn(idx), np.int64)
        return np.array([remap.get(int(v), -1) for v in y], np.int64)

    kept = [task.classes[k] for k in keep]
    return Task(name=task.name, classes=kept, label_fn=label_fn,
                group_col=task.group_col,
                description=f"{task.description} (restricted to {', '.join(kept)})")


ALL = {t.name: t for t in (VARIETY, DISH, GERMINATION)}


def get(name):
    if name not in ALL:
        raise SystemExit(f"unknown task {name!r}; have {sorted(ALL)}")
    return ALL[name]
