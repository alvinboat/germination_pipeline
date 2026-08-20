"""PLS-DA on kernel-mean spectra, scored by dish-grouped cross-validation.

    python3 train_pls.py --task variety --mode reflectance
    python3 train_pls.py --task variety --mode transmittance --deriv 1
    python3 train_pls.py --task variety --controls        # + permutation + dish
    python3 train_pls.py --task variety --sweep           # every mode/side/hour

Reads dataset/spectra.npy only, so it never touches the 14 GB of patches.
Every run writes runs/pls_<task>_<tag>.json with its full configuration, the
per-fold scores and the confusion matrix.
"""
import argparse
import json
import time

import numpy as np

import config
from barley import datasets, index as index_mod, metrics, runlog, splits, tasks
from barley import transforms as T
from barley.models import pls as pls_mod


def run_cv(ds, spec, n_classes, grid, seed=0, verbose=True):
    """-> (per-fold records, pooled y_true, pooled y_pred, pooled scores)."""
    folds = splits.row_folds(ds.idx, spec)
    y_true = np.full(len(ds), -1)
    y_pred = np.full(len(ds), -1)
    y_score = np.zeros((len(ds), n_classes))
    fold_id = np.full(len(ds), -1)
    records = []

    for k, (tr, te) in enumerate(folds):
        if not te.any() or not tr.any():
            continue
        n_comp, inner = pls_mod.choose_components(
            ds.X[tr], ds.y[tr], ds.groups[tr], n_classes, grid, seed=seed + k)
        model = pls_mod.PLSDA(n_comp, n_classes).fit(ds.X[tr], ds.y[tr])
        s = np.asarray(model.decision(ds.X[te]))
        p = s.argmax(1)
        y_true[te], y_pred[te], y_score[te] = ds.y[te], p, s
        fold_id[te] = k
        m = metrics.summarise(ds.y[te], p, n_classes)
        held = sorted({str(g) for g in ds.groups[te].tolist()})
        records.append({"fold": k, "n_components": int(n_comp), **m,
                        "held_out": held, "inner_scores": inner})
        if verbose:
            print(f"  fold {k}: n_comp={n_comp:2d}  balanced_acc={m['balanced_accuracy']:.3f}"
                  f"  acc={m['accuracy']:.3f}  n={m['n']}  held out {','.join(held)}")

    done = y_pred >= 0
    return records, y_true[done], y_pred[done], y_score[done], done, fold_id[done]


def evaluate(ds, spec, task, grid, seed=0, verbose=True):
    n_classes = len(task.classes)
    recs, yt, yp, ys, done, fid = run_cv(ds, spec, n_classes, grid, seed, verbose)
    view = metrics.summarise(yt, yp, n_classes)
    _, kt, kp, _ = metrics.aggregate_by_kernel(ds.kernels[done], yt, ys)
    kernel = metrics.summarise(kt, kp, n_classes)
    cm = metrics.confusion(yt, yp, n_classes)
    # Kept so a report can be built (or rebuilt, or restyled) without retraining.
    pred = {"dataset_row": ds.idx.rows[done], "fold": fid, "y_true": yt,
            "y_pred": yp, "score": ys, "kernel_uid": ds.kernels[done].astype(str)}
    return {"folds": recs, "per_view": view, "per_kernel": kernel,
            "confusion": cm.tolist()}, cm, pred


def permutation_floor(ds, spec, task, grid, n_rep=5, seed=0):
    """Shuffle labels *between groups*, preserving group structure.

    Shuffling per row would break the fact that a dish carries one label and
    would understate the floor. Permuting whole dishes keeps the confound and
    asks the honest question: how well does this pipeline score when the label
    carries no information?
    """
    n_classes = len(task.classes)
    rng = np.random.default_rng(seed)
    out = []
    for r in range(n_rep):
        groups = np.array([str(g) for g in ds.groups.tolist()])
        uniq = np.array(sorted(set(groups.tolist())))
        lab = {}
        for g in uniq:
            lab[g] = int(ds.y[groups == g][0])
        vals = list(lab.values())
        rng.shuffle(vals)
        shuffled = dict(zip(lab.keys(), vals))
        y_shuf = np.array([shuffled[g] for g in groups])

        shadow = object.__new__(type(ds))
        shadow.__dict__.update(ds.__dict__)
        shadow.y = y_shuf
        _, yt, yp, _, _, _ = run_cv(shadow, spec, n_classes, grid,
                                    seed=seed + r, verbose=False)
        out.append(metrics.balanced_accuracy(yt, yp, n_classes))
    return {"mean": float(np.mean(out)), "std": float(np.std(out)),
            "reps": [float(v) for v in out]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="variety")
    p.add_argument("--mode", choices=list(config.MODES) + ["both"], default="reflectance")
    p.add_argument("--side", choices=list(config.SIDES) + ["both"], default="both")
    p.add_argument("--hours", type=float, default=None, help="0, 8, or omit for both")
    p.add_argument("--deriv", type=int, default=0, choices=[0, 1, 2],
                   help="Savitzky-Golay derivative order after SNV")
    p.add_argument("--window", type=int, default=11)
    p.add_argument("--max-nan-frac", type=float, default=None)
    p.add_argument("--grid", type=int, nargs="+", default=list(pls_mod.DEFAULT_GRID))
    p.add_argument("--controls", action="store_true",
                   help="also run the permutation floor and the dish-identity control")
    p.add_argument("--sweep", action="store_true",
                   help="run every mode/side/hour combination and tabulate")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   metavar="N",
                   help="keep only these classes, 1-based (e.g. --classes 1 2 3 5 "
                        "drops variety4). Labels are renumbered; splits.json is "
                        "reused unchanged so the folds stay comparable.")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-report", action="store_true",
                   help="skip building reports/<run>/ at the end")
    args = p.parse_args()

    task = tasks.get(args.task)
    if args.classes:
        task = tasks.restrict(task, [c - 1 for c in args.classes])
        print(f"  restricted to {len(task.classes)} classes: "
              f"{', '.join(task.classes)}  (chance {1 / len(task.classes):.3f})")
    idx = index_mod.Index.load()
    spec = splits.load()
    if spec["task"] != task.name:
        print(f"  note: splits.json was built for task {spec['task']!r}; "
              f"grouping is by {spec['group_col']} either way")

    def make(mode, side, hours):
        sel = index_mod.Selection(
            mode=None if mode == "both" else mode,
            side=None if side == "both" else side,
            hours=hours, max_nan_frac=args.max_nan_frac)
        tf = T.spectrum_pipeline(deriv=args.deriv, window=args.window)
        return datasets.SpectraDataset(task, sel, tf, idx=idx), sel

    if args.sweep:
        print(f"task={task.name}  deriv={args.deriv}\n")
        print(f"{'mode':14s} {'side':8s} {'hours':>6s} {'n':>5s} "
              f"{'view_bal':>9s} {'kernel_bal':>11s}")
        rows = []
        for mode in config.MODES:
            for side in list(config.SIDES) + ["both"]:
                for hours in list(config.HOURS.values()) + [None]:
                    ds, sel = make(mode, side, hours)
                    res, _, _ = evaluate(ds, spec, task, args.grid, verbose=False)
                    rows.append({"mode": mode, "side": side, "hours": hours,
                                 "n": len(ds), **res["per_view"],
                                 "kernel_balanced": res["per_kernel"]["balanced_accuracy"]})
                    print(f"{mode:14s} {side:8s} {str(hours):>6s} {len(ds):5d} "
                          f"{res['per_view']['balanced_accuracy']:9.3f} "
                          f"{res['per_kernel']['balanced_accuracy']:11.3f}", flush=True)
        out = config.RUNS / f"pls_{task.name}_sweep.json"
        config.RUNS.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        print(f"\n-> {out}")
        return

    ds, sel = make(args.mode, args.side, args.hours)
    print(f"task     : {task.name} -- {task.description}")
    print(f"selection: {sel.describe()}")
    print(f"data     : {ds.describe()}")
    print(f"splits   : {spec['n_folds']} folds grouped by {spec['group_col']}\n")

    t0 = time.time()
    res, cm, pred = evaluate(ds, spec, task, args.grid)
    print(f"\nper view  : balanced_acc={res['per_view']['balanced_accuracy']:.3f} "
          f"acc={res['per_view']['accuracy']:.3f} "
          f"macro_f1={res['per_view']['macro_f1']:.3f}  (n={res['per_view']['n']})")
    print(f"per kernel: balanced_acc={res['per_kernel']['balanced_accuracy']:.3f} "
          f"acc={res['per_kernel']['accuracy']:.3f} "
          f"macro_f1={res['per_kernel']['macro_f1']:.3f}  (n={res['per_kernel']['n']})")
    print("\nconfusion (rows = truth):")
    print(metrics.format_confusion(cm, task.classes))

    controls = {}
    if args.controls:
        print("\ncontrols")
        floor = permutation_floor(ds, spec, task, args.grid)
        print(f"  permutation floor : {floor['mean']:.3f} +- {floor['std']:.3f} "
              f"(chance if labels meant nothing)")
        controls["permutation"] = floor

        dish_task = tasks.get("dish")
        dish_ds = datasets.SpectraDataset(dish_task, sel, ds.transform, idx=idx)
        dish_spec = splits.build(dish_ds.idx, dish_task, spec["n_folds"])
        d_res, _, _ = evaluate(dish_ds, dish_spec, dish_task, args.grid,
                                verbose=False)
        d_bal = d_res["per_view"]["balanced_accuracy"]
        print(f"  dish identity     : {d_bal:.3f} over {len(dish_task.classes)} dishes "
              f"(kernel-grouped, so this is 'can it fingerprint the plate?')")
        controls["dish_identity"] = d_res["per_view"]
        gap = res["per_view"]["balanced_accuracy"] - floor["mean"]
        print(f"\n  signal above chance: {gap:+.3f}")

    # The class restriction has to be part of the default name: without it a
    # --classes run silently overwrites the full run's artifacts.
    restriction = "_c" + "".join(str(c) for c in args.classes) if args.classes else ""
    tag = args.tag or (f"{args.mode}_{args.side}_"
                       f"{'all' if args.hours is None else int(args.hours)}h"
                       f"_d{args.deriv}{restriction}")
    name = f"pls_{task.name}_{tag}"
    j, n = runlog.save(name, {
        "task": task.name, "classes": task.classes, "model": "pls-da", "selection": sel.describe(),
        "transform": ds.transform.spec, "grid": args.grid,
        "splits": {"n_folds": spec["n_folds"], "group_col": spec["group_col"],
                   "seed": spec["seed"]},
        "results": res, "controls": controls,
        "seconds": round(time.time() - t0, 1),
    }, pred)
    print(f"\n-> {j}\n-> {n}")

    if not args.no_report:
        from reporting import session as report_session
        out_dir, made = report_session.build(
            json.loads(j.read_text()), pred, name)
        print(f"-> {out_dir}/summary.md  ({len(made)} figures)")


if __name__ == "__main__":
    main()
