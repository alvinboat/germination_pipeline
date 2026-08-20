"""What a training run leaves behind.

Every run writes two files:

    runs/<name>.json    config, per-fold records, pooled metrics, confusion
    runs/<name>.npz     one row per scored view: dataset_row, fold, y_true,
                        y_pred, decision scores, kernel_uid

Keeping the per-view predictions is what lets reporting be a separate step.
A figure can be added, restyled or corrected months later without retraining,
and a per-dish breakdown does not have to be anticipated by the trainer.
"""
import json

import numpy as np

import config


def save(name, meta, pred):
    """-> (json path, npz path)."""
    config.RUNS.mkdir(parents=True, exist_ok=True)
    j = config.RUNS / f"{name}.json"
    n = config.RUNS / f"{name}.npz"
    j.write_text(json.dumps(_finite(meta), indent=1, default=_plain))
    np.savez_compressed(n, **{k: np.asarray(v) for k, v in pred.items()})
    return j, n


def _finite(o):
    """nan/inf -> None, recursively.

    json.dumps emits a bare `NaN`/`Infinity` for these. Python reads that back,
    so it looks fine from here, but it is not valid JSON and anything else
    choking on a run file is a confusing way to find that out. An undefined
    metric -- AUC on a column with one class, a c-index over zero comparable
    pairs -- belongs in the file as null, which `_fmt` already renders as "-".
    """
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def load(path):
    """Accepts either of the pair, or the bare run name. -> (meta, pred)."""
    from pathlib import Path
    path = Path(path)
    if not path.exists() and (config.RUNS / f"{path.name}.json").exists():
        path = config.RUNS / f"{path.name}.json"
    stem = path.with_suffix("")
    meta = json.loads(stem.with_suffix(".json").read_text())
    npz = stem.with_suffix(".npz")
    pred = {k: v for k, v in np.load(npz, allow_pickle=False).items()} \
        if npz.exists() else {}
    return meta, pred, stem.name


def _plain(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        # json.dumps writes a bare NaN/Infinity, which Python reads back but
        # which is not valid JSON for anything else. A metric that is undefined
        # (AUC on a single-class column) belongs in the file as null.
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
