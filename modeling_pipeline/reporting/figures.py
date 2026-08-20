"""One function per figure. Each takes plain arrays and writes a PNG.

Every figure here answers a question someone actually asked of this project:

  fold_scores        how much does the score move between held-out dish sets?
  confusion          which classes get mistaken for which?
  per_class_recall   is the headline hiding one class that does not work?
  per_dish           is a bad fold a bad variety, or one awkward dish?
  spectra_by_class   are these classes even separable in the spectra?
  pls_components     did component selection saturate, or is it truncated?
  cnn_curves         is it learning, or memorising the training dishes?

The numbers in all of them are restated as tables in summary.md, which is both
the accessibility twin and the relief required by the light-mode contrast of
three of the five categorical hues.
"""
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import config
from . import style as S


def short(name, width=4):
    """Compact axis label: leading letters + any trailing number.

    "prospect1" -> "pro1", "unknown" -> "unkn". Class names are real cultivar
    names now, too long for a heatmap tick, and truncating blindly would render
    prospect1 and prospect2 identically.
    """
    m = re.match(r"^([A-Za-z]+?)(\d*)$", str(name))
    if not m:
        return str(name)[:width]
    head, num = m.groups()
    if len(head) + len(num) <= width:
        return head + num
    return head[:max(1, width - len(num))] + num


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ------------------------------------------------------------ fold variance --
def fold_scores(folds, pooled, out, floor=None, metric="balanced accuracy",
                key="balanced_accuracy"):
    """Per-fold score with the pooled value and the permutation floor.

    The spread between folds IS the finding on a 25-dish dataset: a single
    random holdout could have reported either extreme. Only the extremes are
    direct-labelled -- a number on every point goes unread, and summary.md
    carries the full list.

    `key` selects which per-fold number to plot, so a run can show its
    integrated Brier here rather than writing that value into a field named
    balanced_accuracy -- a lie that would then be baked into a stored artifact.
    Say so in `metric`, including the direction, since Brier is lower-is-better.
    """
    ks = [f["fold"] for f in folds]
    ys = np.array([f[key] for f in folds], float)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    lo = min([floor] if floor is not None else []) if floor is not None else ys.min()
    base = min(lo, ys.min())
    ax.vlines(ks, base, ys, color=S.ink("axis"), lw=1.0, zorder=2)
    ax.plot(ks, ys, "o", color=S.series(0), markersize=8, zorder=3,
            markeredgecolor=S.ink("surface"), markeredgewidth=2)  # 2px surface ring

    S.reference_line(ax, float(pooled), "pooled")
    if floor is not None:
        S.reference_line(ax, float(floor), "permutation floor")

    for i in (int(ys.argmin()), int(ys.argmax())):
        ax.annotate(f"{ys[i]:.3f}", xy=(ks[i], ys[i]), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=S.ink("primary"))

    ax.set_xticks(ks)
    ax.set_xticklabels([f"fold {k}\n{_held(folds[i])}" for i, k in enumerate(ks)],
                       fontsize=8)
    ax.set_ylabel(metric)
    ax.set_xlim(min(ks) - 0.5, max(ks) + 0.75)
    ax.set_ylim(base - 0.03, max(ys.max(), pooled) + 0.09)
    S.hide_grid_x(ax)
    S.title(ax, f"{metric} by held-out dish set",
            f"spread {ys.min():.3f}-{ys.max():.3f} across {len(ys)} folds")
    return _save(fig, out)


def _held(f):
    d = sorted(int(x) for x in f.get("held_out", []))
    return "d" + ",".join(str(x) for x in d) if d else ""


# ----------------------------------------------------------------- confusion --
def confusion(cm, classes, out):
    """Row-normalised confusion as a one-hue sequential heatmap, counts inline.

    Sequential because the encoded quantity is a magnitude (recall). One hue,
    light to dark, with a scale legend -- never a rainbow.
    """
    cm = np.asarray(cm, float)
    n = len(classes)
    rows = cm.sum(1, keepdims=True)
    rec = np.divide(cm, np.where(rows > 0, rows, 1))

    fig, ax = plt.subplots(figsize=(1.15 * n + 2.6, 1.15 * n + 1.9))
    cmap = S.cmap_sequential()
    # pcolormesh rather than imshow so adjacent cells can carry a surface-coloured
    # gap between them -- separation by a gap, never by a drawn border.
    im = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), rec, cmap=cmap,
                       vmin=0, vmax=1, edgecolors=S.ink("surface"), linewidth=2)
    ax.grid(False)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    for i in range(n):
        for j in range(n):
            frac = rec[i, j]
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]:.0f}\n{frac:.0%}",
                    ha="center", va="center", fontsize=8.5, linespacing=1.35,
                    color=S.on_dark(_rgb_hex(cmap(frac))))

    ticks = np.arange(n) + 0.5
    ax.set_xticks(ticks, [short(c) for c in classes])
    ax.set_yticks(ticks, [short(c) for c in classes])
    ax.set_xlabel("predicted")
    ax.set_ylabel("truth")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("share of the true class", color=S.ink("secondary"), fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=S.ink("muted"), labelcolor=S.ink("secondary"), length=2)

    S.title(ax, "Confusion", "cell = count, and share of that true class")
    return _save(fig, out)


def _rgb_hex(rgba):
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in rgba[:3])


# ---------------------------------------------------------- per-class recall --
def per_class_recall(cm, classes, out):
    """One bar per class. Single hue: the axis already names the class, so
    colouring by it would re-encode what the chart shows."""
    cm = np.asarray(cm, float)
    rows = cm.sum(1)
    rec = np.divide(np.diag(cm), np.where(rows > 0, rows, 1))
    order = np.argsort(rec)
    chance = 1.0 / len(classes)

    fig, ax = plt.subplots(figsize=(7.0, 0.52 * len(classes) + 2.0))
    y = np.arange(len(classes))
    ax.barh(y, rec[order], height=0.5, color=S.series(0), zorder=2)
    ax.invert_yaxis()          # worst at the top, as the subtitle promises
    # Blank band above the first bar so the chance label has somewhere to sit
    # that is neither on a bar nor on another label.
    ax.set_ylim(len(classes) - 0.4, -1.05)
    ax.axvline(chance, color=S.ink("muted"), lw=1.0, zorder=3)
    # Blended transform: x in data, y in axes fraction, so the label sits at the
    # top of the plot instead of being clipped past the last bar.
    ax.annotate(f"chance {chance:.2f}", xy=(chance, 1.0),
                xycoords=ax.get_xaxis_transform(), xytext=(4, -10),
                textcoords="offset points", fontsize=8,
                color=S.ink("secondary"), va="top")

    for i, v in enumerate(rec[order]):
        # Outside the bar end, so a short bar never clips its own label.
        ax.annotate(f"{v:.2f}  n={int(rows[order][i])}", xy=(v, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=S.ink("secondary"))

    ax.set_yticks(y, [classes[i] for i in order])
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("recall (share of that class found)")
    S.hide_grid_y(ax)
    S.title(ax, "Recall by class", "sorted worst first")
    return _save(fig, out)


# --------------------------------------------------------------- per dish ----
def per_dish(dishes, varieties, correct, classes, out):
    """Accuracy of every dish, grouped by its class.

    A bad fold could be a hard variety or one awkward plate; this separates
    them. Single hue -- the x position already carries the class.
    """
    dishes = np.asarray(dishes)
    varieties = np.asarray(varieties)
    correct = np.asarray(correct, float)

    uniq_v = sorted(set(varieties.tolist()))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    rng = np.random.default_rng(0)

    labels, worst = [], []
    for xi, v in enumerate(uniq_v):
        ds = sorted(set(dishes[varieties == v].tolist()))
        accs = [correct[(dishes == d)].mean() for d in ds]
        jitter = rng.uniform(-0.16, 0.16, size=len(ds))
        ax.plot(np.full(len(ds), xi) + jitter, accs, "o", color=S.series(0),
                markersize=7, markeredgecolor=S.ink("surface"), markeredgewidth=2,
                zorder=3)
        m = float(np.mean(accs))
        ax.plot([xi - 0.30, xi + 0.30], [m, m], color=S.ink("secondary"), lw=1.6,
                zorder=2)
        labels.append(classes[v] if v < len(classes) else str(v))
        j = int(np.argmin(accs))
        worst.append((accs[j], ds[j], xi))

    # Label only the single weakest dish -- the one worth looking at.
    a, d, xi = min(worst)
    ax.annotate(f"dish{d}  {a:.2f}", xy=(xi, a), xytext=(10, -2),
                textcoords="offset points", fontsize=8.5, color=S.ink("primary"))

    ax.set_xticks(range(len(uniq_v)), labels)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.6, len(uniq_v) - 0.4)
    S.hide_grid_x(ax)
    S.title(ax, "Accuracy per dish", "one dot per dish; bar is the class mean")
    return _save(fig, out)


# ------------------------------------------------------- spectra separability --
def spectra_by_class(X, y, classes, wl, out, focus=None):
    """Class mean spectra, and the same means with the grand mean removed.

    Raw NIR means of related cultivars sit almost on top of each other, so the
    left panel usually looks like one curve -- that is itself the answer, and
    the right panel is where the differences become visible. Five series, so a
    legend carries identity; the focus classes are also direct-labelled.
    """
    X = np.asarray(X, float)
    wl = np.asarray(wl, float)
    present = [c for c in range(len(classes)) if (y == c).any()]
    means = {c: X[y == c].mean(0) for c in present}
    grand = X.mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))
    for c in present:
        col = S.series(c)
        axes[0].plot(wl, means[c], color=col, label=classes[c])
        axes[1].plot(wl, means[c] - grand, color=col, label=classes[c])

    if focus:
        for c in focus:
            if c in means:
                axes[1].annotate(classes[c], xy=(wl[-1], (means[c] - grand)[-1]),
                                 xytext=(5, 0), textcoords="offset points",
                                 fontsize=8.5, color=S.series(c), va="center")

    axes[1].axhline(0, color=S.ink("axis"), lw=1.0)
    for ax, t, sub in ((axes[0], "Class mean spectrum", "SNV pseudo-absorbance"),
                       (axes[1], "Deviation from the grand mean",
                        "where the classes actually differ")):
        ax.set_xlabel("wavelength (nm)")
        S.title(ax, t, sub)
    axes[0].set_ylabel("SNV absorbance")
    axes[0].legend(loc="upper right", ncols=2)
    return _save(fig, out)


# ------------------------------------------------------------ PLS internals --
def pls_components(folds, out, ylabel="inner-CV balanced accuracy"):
    """Inner-CV score against component count, one line per fold.

    Reads whether the chosen count is a real optimum or just the top of the
    grid -- the difference between "tuned" and "truncated".
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    any_line = False
    for i, f in enumerate(folds):
        sc = {int(k): v for k, v in (f.get("inner_scores") or {}).items()}
        if not sc:
            continue
        any_line = True
        ns = sorted(sc)
        ax.plot(ns, [sc[n] for n in ns], color=S.series(i), label=f"fold {f['fold']}")
        chosen = f.get("n_components")
        if chosen in sc:
            ax.plot([chosen], [sc[chosen]], "o", color=S.series(i), markersize=8,
                    markeredgecolor=S.ink("surface"), markeredgewidth=2, zorder=4)
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("PLS components")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", ncols=3)
    S.title(ax, "Component selection", "marker = the count each fold chose")
    return _save(fig, out)


# ------------------------------------------------------------ CNN internals --
def cnn_curves(folds, out, key="val_balanced", ylabel="balanced accuracy",
               best="max"):
    """Training loss and held-out score per epoch, in two stacked panels.

    `key`/`ylabel`/`best` let a run plot a different per-epoch metric here, where
    lower is better, instead of writing that value into a field named
    val_balanced -- which would bake a lie into a stored artifact.

    Two panels rather than two y-axes on one plot: a dual-axis chart invents a
    correlation by arbitrary scale alignment. Stacked and sharing x, the reader
    compares them honestly.

    The shape to look for is training loss still falling while the held-out
    score peaks and turns down -- the network memorising its 20 training dishes.
    """
    have = [f for f in folds if f.get("history")]
    if not have:
        return None
    # hspace has to clear a title AND a subtitle on the lower panel; the default
    # puts the lower title on top of the upper panel's axis.
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.6), sharex=True,
                             gridspec_kw={"hspace": 0.34})
    for i, f in enumerate(have):
        h = f["history"]
        ep = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["loss"], color=S.series(i), label=f"fold {f['fold']}")
        axes[1].plot(ep, h[key], color=S.series(i), label=f"fold {f['fold']}")
        j = int(np.argmax(h[key]) if best == "max" else np.argmin(h[key]))
        axes[1].plot([ep[j]], [h[key][j]], "o", color=S.series(i),
                     markersize=8, markeredgecolor=S.ink("surface"),
                     markeredgewidth=2, zorder=4)

    axes[0].set_ylabel("training loss")
    S.title(axes[0], "Training loss", "on the 20 training dishes")
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlabel("epoch")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))   # epochs are whole
    S.title(axes[1], "Held-out score", "marker = best epoch, which is the one kept")
    axes[1].legend(loc="best", ncols=3)
    return _save(fig, out)


# ------------------------------------------------------------------ dataset --
def dataset_overview(index, out):
    """Three things worth knowing before trusting any model on this dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    v = index["variety"]
    uniq = sorted(set(v.tolist()))
    kernels = [len(set(index["kernel_uid"][v == u].tolist())) for u in uniq]
    axes[0].bar([short(config.VARIETY_NAMES[u - 1]) for u in uniq], kernels, width=0.62, color=S.series(0),
                zorder=2)
    for i, n in enumerate(kernels):
        axes[0].annotate(str(n), xy=(i, n), xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=8.5, color=S.ink("secondary"))
    axes[0].set_ylabel("kernels")
    S.hide_grid_x(axes[0])
    S.title(axes[0], "Kernels per variety", f"{sum(kernels)} kernels total")

    for i, m in enumerate(sorted(set(index["mode"].tolist()))):
        sel = index["mode"] == m
        axes[1].hist(index["nan_frac"][sel], bins=40, histtype="step", lw=2.0,
                     color=S.series(i), label=m)
    axes[1].set_xlabel("invalid voxels per patch")
    axes[1].set_ylabel("views")
    # A fraction cannot be negative -- matplotlib's automatic padding around the
    # reflectance spike at 0 would otherwise imply it can.
    axes[1].set_xlim(0, float(index["nan_frac"].max()) * 1.05)
    axes[1].legend(loc="upper right")
    S.title(axes[1], "Invalid fraction", "over the cell box, not the kernel")

    axes[2].hist(index["patch_mask_px"], bins=40, histtype="step", lw=2.0,
                 color=S.series(0))
    axes[2].set_xlabel("kernel pixels per patch")
    axes[2].set_ylabel("views")
    S.title(axes[2], "Kernel size", f"patch is {128 * 64} px")
    return _save(fig, out)
