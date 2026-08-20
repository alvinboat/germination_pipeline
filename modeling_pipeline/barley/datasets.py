"""Turning (index, task, selection) into arrays a model can consume.

Two views of the same dataset:

  SpectraDataset  (N, B) kernel-mean spectra -- the PLS input. Reads spectra.npy
                  only, a few MB, so spectral work never touches the 14 GB of
                  patches.
  PatchDataset    (B, H, W) masked patches -- the CNN input. Memory-maps
                  patches.npy and applies the transform per item.

Neither knows what is being predicted. They take a Task, ask it for labels and
groups, and hand back arrays. Swapping variety for germination changes nothing
here.
"""
import numpy as np

import config
from . import index as index_mod
from . import transforms as T


def _resolve(idx, selection, task):
    idx = (selection.apply(idx) if selection is not None else idx)
    y = task.labels(idx)
    if (y < 0).any():
        n = int((y < 0).sum())
        print(f"  dropping {n} unlabelled view(s) for task {task.name}")
        idx = idx.take(y >= 0)
        y = task.labels(idx)
    if not len(idx):
        raise SystemExit(f"selection {selection.describe() if selection else 'all'} "
                         f"leaves no rows for task {task.name}")
    return idx, y


class SpectraDataset:
    """Kernel-mean spectra, ready for PLS or any sklearn estimator."""

    def __init__(self, task, selection=None, transform=None, idx=None,
                 spectra_path=None):
        idx = idx if idx is not None else index_mod.Index.load()
        self.idx, self.y = _resolve(idx, selection, task)
        self.task = task
        self.transform = transform or T.spectrum_pipeline()
        raw = np.load(spectra_path or config.SPECTRA_NPY)
        self.X_raw = raw[self.idx.rows]
        self.X = np.asarray(self.transform(self.X_raw), np.float32)
        self.groups = self.task.groups(self.idx)
        self.kernels = self.idx["kernel_uid"]

    def __len__(self):
        return len(self.y)

    def fold(self, train_mask, test_mask):
        return (self.X[train_mask], self.y[train_mask],
                self.X[test_mask], self.y[test_mask])

    def describe(self):
        from collections import Counter
        return (f"{self.X.shape[0]} views x {self.X.shape[1]} bands | "
                f"classes {dict(sorted(Counter(self.y.tolist()).items()))} | "
                f"transform {getattr(self.transform, 'spec', {})}")


class PatchDataset:
    """Masked kernel patches. Usable as a torch Dataset; torch is not imported
    here, so the class can be inspected and tested without it installed."""

    def __init__(self, task, selection=None, transform=None, idx=None,
                 augment=None, patches_path=None, masks_path=None):
        idx = idx if idx is not None else index_mod.Index.load()
        self.idx, self.y = _resolve(idx, selection, task)
        self.task = task
        self.transform = transform or T.patch_pipeline()
        self.augment = augment
        self.patches = np.load(patches_path or config.PATCHES_NPY, mmap_mode="r")
        self.masks = np.load(masks_path or config.MASKS_NPY, mmap_mode="r")
        self.groups = self.task.groups(self.idx)
        self.kernels = self.idx["kernel_uid"]
        self._rows = self.idx.rows

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        r = int(self._rows[i])
        x = self.transform(np.asarray(self.patches[r], np.float32),
                           np.asarray(self.masks[r]))
        if self.augment is not None:
            x = self.augment(x)
        return np.ascontiguousarray(x), int(self.y[i])

    def subset(self, mask):
        """A view over a boolean row selection, sharing the same memmaps."""
        out = object.__new__(PatchDataset)
        out.__dict__.update(self.__dict__)
        out.idx = self.idx.take(mask)
        out.y = self.y[mask]
        out.groups = self.groups[mask]
        out.kernels = self.kernels[mask]
        out._rows = out.idx.rows
        return out

    @property
    def n_channels(self):
        return self[0][0].shape[0]

    def describe(self):
        from collections import Counter
        c, h, w = self[0][0].shape
        return (f"{len(self)} views x ({c}, {h}, {w}) | "
                f"classes {dict(sorted(Counter(self.y.tolist()).items()))} | "
                f"transform {getattr(self.transform, 'spec', {})}")


def flip_augment(rng=None, p=0.5):
    """Horizontal and vertical flips.

    Label-preserving here specifically because the warp already normalised
    orientation: every patch has the plate's row axis down its rows and the
    dorsal/ventral mirror removed, so a flipped kernel is a kernel that could
    have been placed the other way round in its well.
    """
    rng = rng or np.random.default_rng(config.SEED)

    def f(x):
        if rng.random() < p:
            x = x[:, ::-1, :]
        if rng.random() < p:
            x = x[:, :, ::-1]
        return np.ascontiguousarray(x)
    return f
