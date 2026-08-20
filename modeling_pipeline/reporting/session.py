"""Assemble one training session's report folder.

    reports/<run name>/
        summary.md              headline, per-fold, per-class, confusion, config
        01_fold_scores.png
        02_confusion.png
        03_per_class_recall.png
        04_per_dish.png
        05_spectra_by_class.png
        06_components.png        PLS only
        06_training_curves.png   CNN only

summary.md is not an afterthought. It is the table view every figure needs to
be accessible, it is the relief required by the light-mode contrast of three
categorical hues, and it is the thing that is still readable in a terminal or a
diff six months from now.
"""
import numpy as np

import config
from barley import index as index_mod, metrics, tasks
from . import figures as F
from . import style as S


def _fmt(v, n=3):
    return "-" if v is None else f"{v:.{n}f}"


def build(meta, pred, name, out_dir=None, mode="light"):
    """Build a report folder from a run's json + npz.

    This once dispatched on run kind, because the germination-time regressor
    shared no headline metric with a classifier. That model is gone and there is
    only one kind left, so the dispatch is a passthrough -- kept as the seam to
    reopen if a second kind ever comes back."""
    return build_classification(meta, pred, name, out_dir=out_dir, mode=mode)


def build_classification(meta, pred, name, out_dir=None, mode="light"):
    S.use(mode)
    out = (out_dir or config.REPORTS) / name
    out.mkdir(parents=True, exist_ok=True)

    # The run records its own class list: a run restricted with --classes has
    # fewer than the registered task, and looking the task up would give a
    # confusion matrix the wrong shape.
    classes = meta.get("classes") or tasks.get(meta["task"]).classes
    # train_pls nests its metrics under "results"; train_cnn writes them flat.
    core = meta.get("results", meta)
    folds = core.get("folds", [])
    # A run that does not produce a confusion matrix (or pooled per-view /
    # per-kernel blocks) still deserves whatever report it can support. These
    # were unguarded, so one such run raised KeyError before any figure was
    # written -- and under --all it aborted every later report too.
    cm = core.get("confusion")
    cm = np.asarray(cm, float) if cm is not None else None
    per_view = core.get("per_view")
    per_kernel = core.get("per_kernel")
    floor = ((meta.get("controls") or {}).get("permutation") or {}).get("mean")

    made = []
    if folds and per_view:
        made.append(F.fold_scores(folds, per_view["balanced_accuracy"],
                                  out / "01_fold_scores.png", floor=floor))
    if cm is not None:
        made.append(F.confusion(cm, classes, out / "02_confusion.png"))
        made.append(F.per_class_recall(cm, classes,
                                       out / "03_per_class_recall.png"))

    idx = index_mod.Index.load()
    by_row = {int(r): i for i, r in enumerate(idx.rows)}

    if pred:
        rows = pred["dataset_row"].astype(int)
        sel = np.array([by_row[r] for r in rows])
        dishes = idx["dish"][sel]
        varieties = pred["y_true"].astype(int)
        made.append(F.per_dish(dishes, varieties,
                               pred["y_true"] == pred["y_pred"], classes,
                               out / "04_per_dish.png"))

        # Separability of the classes in the spectra themselves -- independent of
        # whatever the model did with them.
        spectra = np.load(config.SPECTRA_NPY)[rows]
        from barley import transforms as T
        worst = _worst_pair(cm) if cm is not None else None
        made.append(F.spectra_by_class(
            T.snv(spectra), varieties, classes,
            _wavelengths(), out / "05_spectra_by_class.png", focus=worst))

    if any(f.get("inner_scores") for f in folds):
        made.append(F.pls_components(folds, out / "06_components.png"))
    if any(f.get("history") for f in folds):
        made.append(F.cnn_curves(folds, out / "06_training_curves.png"))

    made = [m for m in made if m]
    (out / "summary.md").write_text(
        _summary(meta, pred, name, cm, classes, per_view, per_kernel, floor, idx, made))
    return out, made


def _wavelengths():
    import json
    b = json.loads(config.BANDS_JSON.read_text())
    return np.asarray(b["wavelength_nm"], float)


def _worst_pair(cm):
    """The two classes most confused with each other -- the story of the run."""
    cm = np.asarray(cm, float)
    off = cm + cm.T
    np.fill_diagonal(off, -1)
    i, j = np.unravel_index(off.argmax(), off.shape)
    return [int(i), int(j)]


def _summary(meta, pred, name, cm, classes, per_view, per_kernel, floor, idx, made):
    L = [f"# {name}", ""]
    L += [f"`{meta.get('selection', '')}`  ",
          f"task **{meta['task']}** · {meta.get('transform', {})}", ""]

    if per_view and per_kernel:
        L += ["## Headline", "",
              "| | balanced accuracy | accuracy | macro F1 | n |",
              "|---|---|---|---|---|",
              f"| per view | **{_fmt(per_view['balanced_accuracy'])}** | "
              f"{_fmt(per_view['accuracy'])} | {_fmt(per_view['macro_f1'])} | "
              f"{per_view['n']} |",
              f"| per kernel | **{_fmt(per_kernel['balanced_accuracy'])}** | "
              f"{_fmt(per_kernel['accuracy'])} | {_fmt(per_kernel['macro_f1'])} | "
              f"{per_kernel['n']} |"]
        chance = 1.0 / len(classes)
        L += [f"| chance | {_fmt(chance)} | | | |"]
        if floor is not None:
            L += [f"| permutation floor | {_fmt(floor)} | | | |"]
        L += [""]
        if floor is not None:
            L += [f"Signal above the measured floor: "
                  f"**{per_view['balanced_accuracy'] - floor:+.3f}**", ""]

    folds = meta.get("results", meta).get("folds", [])
    if folds:
        ys = [f["balanced_accuracy"] for f in folds]
        L += ["## Folds", "",
              f"Held out by **{meta.get('splits', {}).get('group_col', 'dish')}**. "
              f"Spread {min(ys):.3f}–{max(ys):.3f}; "
              f"sd {np.std(ys):.3f}, so the pooled figure carries roughly "
              f"±{1.96 * np.std(ys) / np.sqrt(len(ys)):.3f} at 95%.", "",
              "| fold | balanced | accuracy | n | held out | detail |",
              "|---|---|---|---|---|---|"]
        for f in folds:
            detail = ""
            if "n_components" in f:
                detail = f"{f['n_components']} components"
            elif f.get("history"):
                detail = f"best epoch {int(np.argmax(f['history']['val_balanced'])) + 1}"
            held = f.get("held_out", [])
            try:                       # dish ids sort numerically, not as strings
                held = [str(v) for v in sorted(int(v) for v in held)]
            except ValueError:
                held = sorted(held)
            L.append(f"| {f['fold']} | {_fmt(f['balanced_accuracy'])} | "
                     f"{_fmt(f['accuracy'])} | {f['n']} | "
                     f"{', '.join(held)} | {detail} |")
        L += [""]

    if cm is not None:
        rows = cm.sum(1)
        rec = np.divide(np.diag(cm), np.where(rows > 0, rows, 1))
        L += ["## Per class", "", "| class | recall | n | most often mistaken for |",
              "|---|---|---|---|"]
        for c in np.argsort(rec):
            off = cm[c].copy()
            off[c] = -1
            other = classes[int(off.argmax())] if off.max() > 0 else "-"
            L.append(f"| {classes[c]} | {_fmt(rec[c], 3)} | {int(rows[c])} | "
                     f"{other} ({int(off.max()) if off.max() > 0 else 0}) |")
        L += [""]

        L += ["## Confusion", "", "Rows are truth, columns prediction.", "",
              "| | " + " | ".join(classes) + " |",
              "|---|" + "---|" * len(classes)]
        for i, c in enumerate(classes):
            L.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
        L += [""]

    if pred:
        prows = pred["dataset_row"].astype(int)
        by_row = {int(r): i for i, r in enumerate(idx.rows)}
        sel = np.array([by_row[r] for r in prows])
        dishes = idx["dish"][sel]
        ok = pred["y_true"] == pred["y_pred"]
        accs = sorted(((float(ok[dishes == d].mean()), int(d))
                       for d in sorted(set(dishes.tolist()))))
        L += ["## Hardest dishes", "", "| dish | accuracy | views |",
              "|---|---|---|"]
        for a, d in accs[:5]:
            L.append(f"| dish{d} | {_fmt(a)} | {int((dishes == d).sum())} |")
        L += ["", f"Best: dish{accs[-1][1]} at {_fmt(accs[-1][0])}.", ""]

    L += ["## Figures", ""]
    L += [f"- `{m.name}`" for m in made]
    L += ["", "## Run", "", "```json",
          _json_block(meta), "```", ""]
    return "\n".join(L)


def _json_block(meta):
    import json
    keep = {k: v for k, v in meta.items()
            if k not in ("folds", "confusion", "results", "controls")}
    return json.dumps(keep, indent=1)


def dataset_report(out_dir=None, mode="light"):
    """A one-off overview of the dataset itself, independent of any run."""
    S.use(mode)
    out = (out_dir or config.REPORTS) / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    idx = index_mod.Index.load()
    png = F.dataset_overview(idx, out / "overview.png")

    kernels = len(set(idx["kernel_uid"].tolist()))
    L = ["# dataset", "", f"`{idx.describe()}`", "",
         "| | |", "|---|---|",
         f"| views (rows) | {len(idx)} |",
         f"| kernels | {kernels} |",
         f"| dishes | {len(set(idx['dish'].tolist()))} |",
         f"| bands | {config.N_BANDS} ({config.wavelengths()[0]:.0f}"
         f"–{config.wavelengths()[-1]:.0f} nm) |",
         f"| patch | {config.PATCH_H}x{config.PATCH_W} |", ""]
    L += ["## Views per kernel", "",
          "A kernel is measured 8 times: 2 modes x 2 sides x 2 time points. "
          "Those are repeat looks at one grain, not independent samples — which "
          "is why folds group by dish and results are also reported per kernel.",
          ""]
    for m in sorted(set(idx["mode"].tolist())):
        sel = idx["mode"] == m
        L.append(f"- **{m}**: {int(sel.sum())} views, "
                 f"median invalid fraction {np.median(idx['nan_frac'][sel]):.3f}, "
                 f"median kernel {int(np.median(idx['patch_mask_px'][sel]))} px")
    L += ["", "## Figures", "", f"- `{png.name}`", ""]
    (out / "summary.md").write_text("\n".join(L))
    return out, [png]
