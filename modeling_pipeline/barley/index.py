"""The dataset index, and selecting rows from it.

One row per (kernel, mode, side, hours) view. Deliberately plain numpy and the
stdlib csv module rather than pandas: the index is a few thousand rows of flat
columns, this keeps the ingest and verification path dependency-free, and a
`Selection` reads no worse than a chain of boolean masks.
"""
import csv
from dataclasses import dataclass, field

import numpy as np

import config

def _timing():
    # Imported lazily: barley.timing reads config.DATASET, and a circular import
    # here would break `python3 -m barley.splits`.
    from . import timing
    return timing


INT_COLS = {"row", "dish", "cell_index", "variety", "day_folder", "coco_ann_id",
            "coco_image_id", "mask_px_raw", "mask_px", "patch_mask_px",
            "n_components_raw", "dropped_px"}
FLOAT_COLS = {"hours", "containment", "clipped_frac", "sat_frac", "nan_frac"}


class Index:
    """Column-oriented view of index.csv. `idx["dish"]` is an int array."""

    def __init__(self, cols, path=None):
        self.cols = cols
        self.path = path

    @classmethod
    def load(cls, path=None, include_excluded=False, quiet=False):
        """The index, minus the varieties config has ruled out.

        The exclusion lives here rather than in each trainer because every
        consumer -- trainers, split builders, datasets, figures -- comes through
        this one method, and a variety that is axed in four places and forgotten
        in the fifth is worse than one that was never axed. It is announced on
        load rather than applied silently: a run that quietly trains on 80% of
        the plate is exactly the kind of thing nobody notices.

        `include_excluded=True` is for tools that inspect the built file as
        built -- the mask review, the label parser's cell map -- rather than
        tools that model with it.

        Two columns are attached that are not in index.csv: `capture_time` and
        `capture_hours`, the MEASURED clock from `dataset/timing.json`. They
        live outside the csv because they are a fact about the captures, not
        about the rows, and correcting them must not cost a 14 GB rebuild.
        `hours` remains the nominal slot (0/8/24) that every existing selection
        and split is written against.
        """
        path = path or config.INDEX_CSV
        if not path.exists():
            raise SystemExit(f"no index at {path} -- run build_dataset.py first")
        rows = list(csv.DictReader(path.open()))
        if not rows:
            raise SystemExit(f"{path} is empty")
        cols = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            if k in INT_COLS:
                cols[k] = np.array([int(v) for v in vals], np.int64)
            elif k in FLOAT_COLS:
                cols[k] = np.array([float(v) for v in vals], np.float64)
            else:
                cols[k] = np.array(vals, object)
        idx = cls(cols, path)
        if include_excluded or not config.EXCLUDED_VARIETIES:
            return _timing().attach(idx, quiet)
        drop = np.isin(idx["variety"], list(config.EXCLUDED_VARIETIES))
        if drop.any() and not quiet:
            print(f"  excluding {', '.join(config.excluded_names())}: "
                  f"{int(drop.sum())} views, "
                  f"{len(set(idx['kernel_uid'][drop].tolist()))} kernels "
                  f"(config.EXCLUDED_VARIETIES)")
        return _timing().attach(idx.take(~drop), quiet)

    def __len__(self):
        return len(next(iter(self.cols.values())))

    def __getitem__(self, key):
        return self.cols[key]

    def __contains__(self, key):
        return key in self.cols

    def take(self, where):
        """-> a new Index holding only the rows where `where` is True."""
        where = np.asarray(where)
        if where.dtype != bool:
            keep = np.zeros(len(self), bool)
            keep[where] = True
            where = keep
        return Index({k: v[where] for k, v in self.cols.items()}, self.path)

    @property
    def rows(self):
        """Row numbers into patches.npy / masks.npy / spectra.npy."""
        return self.cols["row"]

    def describe(self):
        from collections import Counter
        return (f"{len(self)} views | "
                f"{len(set(self['kernel_uid']))} kernels | "
                f"{len(set(self['dish'].tolist()))} dishes | "
                f"modes {dict(Counter(self['mode'].tolist()))} | "
                f"hours {dict(Counter(self['hours'].tolist()))}")


@dataclass
class Selection:
    """Which views a task should train on.

    Every field is optional and ANDed. `None` means "do not filter on this".
    Quality gates default to off so that what is excluded is always an explicit
    choice recorded in the run, never a silent default.
    """
    mode: str = None
    side: str = None
    hours: float = None
    dish: list = None
    variety: list = None
    max_nan_frac: float = None
    max_sat_frac: float = None
    min_mask_px: int = None
    extra: list = field(default_factory=list)   # [(column, value_or_list)]

    def apply(self, idx):
        keep = np.ones(len(idx), bool)
        for col, want in [("mode", self.mode), ("side", self.side),
                          ("hours", self.hours), ("dish", self.dish),
                          ("variety", self.variety), *self.extra]:
            if want is None:
                continue
            want = want if isinstance(want, (list, tuple, set, np.ndarray)) else [want]
            keep &= np.isin(idx[col], list(want))
        if self.max_nan_frac is not None:
            keep &= idx["nan_frac"] <= self.max_nan_frac
        if self.max_sat_frac is not None:
            keep &= idx["sat_frac"] <= self.max_sat_frac
        if self.min_mask_px is not None:
            keep &= idx["patch_mask_px"] >= self.min_mask_px
        return idx.take(keep)

    def describe(self):
        bits = [f"{k}={v}" for k, v in vars(self).items() if v not in (None, [], ())]
        return ", ".join(bits) or "all views"
