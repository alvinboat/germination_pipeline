"""Cross-validation folds, held out by whole dish.

WHY DISH AND NOT KERNEL
Variety is perfectly confounded with dish: every kernel of variety 1 is in
dishes 0-4 and nowhere else. So any dish-level artifact -- plate, illumination,
capture session, stage speed -- is a *perfect* variety predictor. Split at the
kernel level and the model can memorise the dish and score near-perfect while
having learned nothing about barley.

Holding out whole dishes makes the test set a dish the model has never seen, so
the only way to be right is to have learned something that transfers between
dishes of the same variety. That is a harder number and it is the honest one.

The same argument applies, more weakly, to germination -- kernels in a dish
share a plate and a capture -- so dish grouping is the default there too.

Folds are written to dataset/splits.json and read by every model, so PLS and CNN
numbers are comparable rather than coincidentally similar.
"""
import json
from collections import defaultdict

import numpy as np

import config


def stratified_group_folds(groups, strata, n_folds, seed=config.SEED):
    """Assign whole groups to folds, keeping each stratum spread evenly.

    With 25 dishes, 5 varieties and 5 folds this puts exactly one dish of each
    variety in each fold. It degrades gracefully if the counts stop being so
    tidy: groups are dealt round-robin within a stratum, largest stratum first.
    """
    rng = np.random.default_rng(seed)
    by_stratum = defaultdict(list)
    for g in sorted(set(groups.tolist())):
        s = strata[groups == g]
        if len(set(s.tolist())) != 1:
            raise SystemExit(f"group {g!r} spans strata {sorted(set(s.tolist()))} -- "
                             "a group must map to exactly one stratum")
        by_stratum[s[0]].append(g)

    fold_of = {}
    for stratum in sorted(by_stratum, key=lambda k: (-len(by_stratum[k]), str(k))):
        members = list(by_stratum[stratum])
        rng.shuffle(members)
        for i, g in enumerate(members):
            fold_of[g] = i % n_folds
    return fold_of


def build(idx, task, n_folds=config.N_FOLDS, seed=config.SEED):
    """-> {"folds": [{"train": [...], "test": [...]}, ...], ...} keyed on kernel_uid.

    Folds are stored as kernel ids, not row numbers, so a rebuild of the dataset
    (new masks, different row order) does not silently invalidate them.
    """
    groups = idx[task.group_col]
    y = task.labels(idx)
    keep = y >= 0
    if not keep.all():
        idx, groups, y = idx.take(keep), groups[keep], y[keep]

    fold_of = stratified_group_folds(groups, y, n_folds, seed)
    assigned = np.array([fold_of[g] for g in groups.tolist()])

    kernels = idx["kernel_uid"]
    folds = []
    for k in range(n_folds):
        te = sorted(set(kernels[assigned == k].tolist()))
        tr = sorted(set(kernels[assigned != k].tolist()))
        folds.append({"train": tr, "test": te})
    return {
        "task": task.name,
        "group_col": task.group_col,
        "n_folds": n_folds,
        "seed": seed,
        "group_fold": {str(g): int(f) for g, f in sorted(fold_of.items(), key=lambda x: str(x[0]))},
        "folds": folds,
    }


def save(spec, path=None):
    path = path or config.SPLITS_JSON
    path.write_text(json.dumps(spec, indent=1))
    return path


def load(path=None):
    path = path or config.SPLITS_JSON
    if not path.exists():
        raise SystemExit(f"no splits at {path} -- run: python3 -m barley.splits")
    return json.loads(path.read_text())


def row_folds(idx, spec):
    """-> list of (train_rows, test_rows) as boolean masks over `idx`."""
    out = []
    kernels = idx["kernel_uid"]
    for f in spec["folds"]:
        tr = np.isin(kernels, list(f["train"]))
        te = np.isin(kernels, list(f["test"]))
        out.append((tr, te))
    return out


def main():
    import argparse
    from . import index as index_mod
    from . import tasks as tasks_mod

    p = argparse.ArgumentParser(description="write dataset/splits.json")
    p.add_argument("--task", default="variety")
    p.add_argument("--folds", type=int, default=config.N_FOLDS)
    args = p.parse_args()

    idx = index_mod.Index.load()
    task = tasks_mod.get(args.task)
    spec = build(idx, task, args.folds)
    path = save(spec)

    print(f"{args.task}: {args.folds} folds grouped by {task.group_col}")
    for i, f in enumerate(spec["folds"]):
        dishes = sorted({k.split("_")[0] for k in f["test"]})
        print(f"  fold {i}: {len(f['train'])} train / {len(f['test'])} test kernels "
              f"| held out {', '.join(dishes)}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
