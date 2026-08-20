"""ResNet-18 on masked kernel patches, scored by the same dish-grouped folds
that train_pls.py uses -- so the two numbers are comparable rather than
coincidentally similar.

    python3 train_cnn.py --task variety --mode reflectance
    python3 train_cnn.py --task variety --mode reflectance --stem pca3 --pretrained
    python3 train_cnn.py --task variety --fold 0 --epochs 5      # quick smoke test

Requires torch. Run it from the venv:
    .venv/bin/python train_cnn.py ...
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from barley import datasets, index as index_mod, metrics, runlog, splits, tasks
from barley import transforms as T
from barley.models import cnn as cnn_mod


class TorchWrap(torch.utils.data.Dataset):
    """Adapts barley.datasets.PatchDataset (plain numpy) to torch."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return torch.from_numpy(np.asarray(x, np.float32)), y


def run_fold(ds, tr_mask, te_mask, task, args, device, seed=0):
    torch.manual_seed(seed)
    train_ds = ds.subset(tr_mask)
    test_ds = ds.subset(te_mask)
    train_ds.augment = datasets.flip_augment(np.random.default_rng(seed)) \
        if args.augment else None
    test_ds.augment = None

    n_classes = len(task.classes)
    model = cnn_mod.build(ds.n_channels, n_classes, stem=args.stem,
                          pretrained=args.pretrained, dropout=args.dropout).to(device)
    if args.stem == "pca3":
        spectra = np.load(config.SPECTRA_NPY)[train_ds.idx.rows]
        model.fit_reduce(T.snv(spectra))

    counts = np.bincount(train_ds.y, minlength=n_classes).astype(np.float64)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) / n_classes,
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tl = DataLoader(TorchWrap(train_ds), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=len(train_ds) > args.batch)
    vl = DataLoader(TorchWrap(test_ds), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers)

    best = {"balanced_accuracy": -1.0}
    best_scores = None
    # Kept per epoch so the report can show whether the fold was still learning
    # or had started memorising its 20 training dishes.
    history = {"loss": [], "val_balanced": [], "val_accuracy": []}
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
        sched.step()

        model.eval()
        scores, ys = [], []
        with torch.no_grad():
            for x, y in vl:
                scores.append(model(x.to(device)).cpu().numpy())
                ys.append(y.numpy())
        s = np.concatenate(scores)
        yt = np.concatenate(ys)
        m = metrics.summarise(yt, s.argmax(1), n_classes)
        history["loss"].append(tot / len(train_ds))
        history["val_balanced"].append(m["balanced_accuracy"])
        history["val_accuracy"].append(m["accuracy"])
        if args.verbose:
            print(f"    epoch {ep + 1:3d}/{args.epochs}  loss={tot / len(train_ds):.4f}"
                  f"  val_balanced={m['balanced_accuracy']:.3f}", flush=True)
        if m["balanced_accuracy"] > best["balanced_accuracy"]:
            best, best_scores = m, s
    return best, best_scores, test_ds, history


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="variety")
    p.add_argument("--mode", choices=list(config.MODES) + ["both"], default="reflectance")
    p.add_argument("--side", choices=list(config.SIDES) + ["both"], default="both")
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--deriv", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--stem", choices=["direct", "pca3"], default="direct")
    p.add_argument("--pretrained", action="store_true",
                   help="ImageNet weights; only meaningful with --stem pca3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fold", type=int, default=None, help="run one fold only")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   metavar="N",
                   help="keep only these classes, 1-based (e.g. --classes 1 2 3 5 "
                        "drops variety4). Labels are renumbered; splits.json is "
                        "reused unchanged so the folds stay comparable.")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-report", action="store_true",
                   help="skip building reports/<run>/ at the end")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = tasks.get(args.task)
    if args.classes:
        task = tasks.restrict(task, [c - 1 for c in args.classes])
        print(f"  restricted to {len(task.classes)} classes: "
              f"{', '.join(task.classes)}  (chance {1 / len(task.classes):.3f})")
    idx = index_mod.Index.load()
    spec = splits.load()

    sel = index_mod.Selection(
        mode=None if args.mode == "both" else args.mode,
        side=None if args.side == "both" else args.side,
        hours=args.hours)
    ds = datasets.PatchDataset(task, sel, T.patch_pipeline(deriv=args.deriv), idx=idx)

    print(f"task     : {task.name} -- {task.description}")
    print(f"selection: {sel.describe()}")
    print(f"data     : {ds.describe()}")
    print(f"model    : resnet18 stem={args.stem} pretrained={args.pretrained} "
          f"in_ch={ds.n_channels} device={device}")
    print(f"splits   : {spec['n_folds']} folds grouped by {spec['group_col']}\n")

    folds = splits.row_folds(ds.idx, spec)
    n_classes = len(task.classes)
    t0 = time.time()
    records = []
    all_true, all_pred, all_score, all_kernels, all_rows, all_fold = [], [], [], [], [], []

    for k, (tr, te) in enumerate(folds):
        if args.fold is not None and k != args.fold:
            continue
        if not tr.any() or not te.any():
            continue
        held = sorted({str(g) for g in ds.groups[te].tolist()})
        print(f"  fold {k}: {int(tr.sum())} train / {int(te.sum())} test views "
              f"| held out {','.join(held)}", flush=True)
        best, scores, test_ds, history = run_fold(ds, tr, te, task, args, device,
                                                  seed=k)
        print(f"    -> balanced_acc={best['balanced_accuracy']:.3f} "
              f"acc={best['accuracy']:.3f}", flush=True)
        records.append({"fold": k, "held_out": held, **best, "history": history})
        all_true.append(test_ds.y)
        all_pred.append(scores.argmax(1))
        all_score.append(scores)
        all_kernels.append(test_ds.kernels)
        all_rows.append(test_ds.idx.rows)
        all_fold.append(np.full(len(test_ds.y), k))

    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    ys = np.concatenate(all_score)
    kk = np.concatenate(all_kernels)
    view = metrics.summarise(yt, yp, n_classes)
    _, kt, kp, _ = metrics.aggregate_by_kernel(kk, yt, ys)
    kernel = metrics.summarise(kt, kp, n_classes)
    cm = metrics.confusion(yt, yp, n_classes)

    print(f"\nper view  : balanced_acc={view['balanced_accuracy']:.3f} "
          f"acc={view['accuracy']:.3f} macro_f1={view['macro_f1']:.3f} (n={view['n']})")
    print(f"per kernel: balanced_acc={kernel['balanced_accuracy']:.3f} "
          f"acc={kernel['accuracy']:.3f} macro_f1={kernel['macro_f1']:.3f} "
          f"(n={kernel['n']})")
    print("\nconfusion (rows = truth):")
    print(metrics.format_confusion(cm, task.classes))

    restriction = "_c" + "".join(str(c) for c in args.classes) if args.classes else ""
    tag = args.tag or f"{args.mode}_{args.side}_{args.stem}{restriction}"
    name = f"cnn_{task.name}_{tag}"
    meta = {
        "task": task.name, "classes": task.classes, "model": "resnet18", "selection": sel.describe(),
        "args": {k: v for k, v in vars(args).items()},
        "transform": ds.transform.spec,
        "splits": {"n_folds": spec["n_folds"], "group_col": spec["group_col"]},
        "folds": records, "per_view": view, "per_kernel": kernel,
        "confusion": cm.tolist(), "seconds": round(time.time() - t0, 1),
    }
    pred = {"dataset_row": np.concatenate(all_rows),
            "fold": np.concatenate(all_fold),
            "y_true": yt, "y_pred": yp, "score": ys, "kernel_uid": kk.astype(str)}
    j, n = runlog.save(name, meta, pred)
    print(f"\n-> {j}\n-> {n}")

    if not args.no_report:
        from reporting import session as report_session
        out_dir, made = report_session.build(json.loads(j.read_text()), pred, name)
        print(f"-> {out_dir}/summary.md  ({len(made)} figures)")


if __name__ == "__main__":
    main()
