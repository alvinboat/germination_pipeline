"""One row per kernel: the matrix every kernel-level analysis is built on.

The index holds eight views of each kernel -- two modes, two faces, two capture
times. For a question asked about the KERNEL rather than the measurement, eight
rows per kernel is pseudo-replication: the folds would still be honest, but
every count, every variance and every significance test would be inflated
eightfold.

So the views are collapsed. Not all eight into one row:

* reflectance and transmittance are different radiometry with different dynamic
  range, and belong in different matrices;
* 0 h is the dry seed and 8 h is the imbibed one -- two physical states, not two
  looks at one.

Averaging is therefore over SIDES only, giving one block per (mode, hours), each
exactly one row per kernel and all four sharing a row order. A kernel the
annotator missed on one face becomes a one-sided mean rather than a dropped
kernel, and is reported.
"""
from collections import defaultdict

import numpy as np

import config


def side_averaged(idx, spectra):
    """-> dict with `kernels`, `dish`, `variety`, `blocks`, `one_sided`.

    `blocks` is {(mode, hours): (n_kernels, N_BANDS) float32}, row-aligned.
    """
    by, meta = defaultdict(dict), {}
    for i in range(len(idx)):
        uid = str(idx["kernel_uid"][i])
        cell = (str(idx["mode"][i]), float(idx["hours"][i]))
        by[cell].setdefault(uid, []).append(spectra[int(idx["row"][i])])
        meta[uid] = (int(idx["dish"][i]), int(idx["variety"][i]))

    cells = sorted(by)
    kernels = sorted(set.intersection(*(set(by[c]) for c in cells)))
    dropped = sorted(set.union(*(set(by[c]) for c in cells)) - set(kernels))
    one_sided = sorted(u for u in kernels
                       if any(len(by[c][u]) < len(config.SIDES) for c in cells))
    return {
        "kernels": np.array(kernels, object),
        "dish": np.array([meta[u][0] for u in kernels]),
        "variety": np.array([meta[u][1] for u in kernels]),
        "blocks": {c: np.stack([np.mean(by[c][u], axis=0) for u in kernels])
                   for c in cells},
        "one_sided": one_sided,
        "dropped": dropped,
    }
