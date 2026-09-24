"""PCA of the mask-mean spectra: what varies, and whether it is the grain.

    python3 explore/pca_spectra.py             # -> exploration/pca/mean_spectra/

ONE ROW PER KERNEL, AND WHAT THAT COST
The eight views of a kernel are eight correlated rows, so they are averaged
down to one. But NOT all eight into one row:

* Reflectance and transmittance are different radiometry with different dynamic
  range. Averaged together -- or even put in one matrix uncentred -- PC1 becomes
  "which camera mode" and explains most of the variance and nothing else. They
  get separate PCAs.
* day1 is 0 h, shot BEFORE the seeds were wetted; day9 is 8 h, after eight hours
  of imbibition. Those are two physical states of the same kernel, not two looks
  at one state. Averaging them would throw away the imbibition contrast, which
  is the single axis most likely to carry germination signal.

So the averaging is over SIDES only -- dorsal and ventral, two faces of one
kernel in one state -- and the four resulting matrices are each exactly one row
per kernel, 438 x 192. A fifth view is added per mode: the 8 h - 0 h difference
spectrum, which is imbibition with the kernel's own baseline removed.

WHAT THE CONTROLS ARE FOR
Variety is perfectly confounded with dish: every kernel of a variety sits in
five consecutive dishes and nowhere else. A component that separates varieties
is therefore worth nothing on its own -- `controls.csv` reports, per component,
the share of its variance that sits between dishes beside the share that sits
between varieties. If the two are equal, the component is reading the plate.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                     # noqa: E402
from barley import index as index_mod, kernels as kmod, pca, germination  # noqa: E402
from reporting import style as S                  # noqa: E402

DERIVS = (0, 1)
N_PC = 6                # kept and reported; the scree figure shows all of them
N_PC_DIAG = 4           # components the T2/Q outlier map is built on


# --------------------------------------------------------------------- data --
def build_matrices(idx, spectra, labels):
    """-> {(mode, key): {"X", "kernels", "dish", "variety", "day"}}.

    `key` is "0h", "8h" or "delta". Rows are aligned across every matrix, so a
    kernel is the same row index everywhere and the four score sets can be
    compared point by point.
    """
    ks = kmod.side_averaged(idx, spectra)
    kernels = list(ks["kernels"])
    base = {"kernels": ks["kernels"], "dish": ks["dish"], "variety": ks["variety"],
            "day": [labels.get(u, germination.UNSCORED) for u in kernels]}
    out = {}
    for mode in config.MODES:
        lo, hi = sorted(config.HOURS.values())
        a, b = ks["blocks"][(mode, lo)], ks["blocks"][(mode, hi)]
        out[(mode, f"{lo:g}h")] = {"X": a, **base}
        out[(mode, f"{hi:g}h")] = {"X": b, **base}
        out[(mode, "delta")] = {"X": b - a, **base}
    return out, kernels, ks["dropped"], ks["one_sided"]


def outcome_arrays(day):
    """-> (germinated 0/1, first day with -1 for never) as plain int arrays."""
    ever = np.array([0 if d is None else 1 for d in day])
    first = np.array([-1 if d is None else int(d) for d in day])
    return ever, first


# ------------------------------------------------------------------ figures --
def fig_scree(results, out):
    fig, axes = plt.subplots(1, len(DERIVS), figsize=(11.4, 4.6), sharey=True)
    for ax, deriv in zip(np.atleast_1d(axes), DERIVS):
        cells = sorted({(m, k) for (d, m, k) in results if d == deriv}, key=str)
        for i, key in enumerate(cells):
            r = results[(deriv, *key)]
            cum = 100 * r["fit"]["cum_evr"][:10]
            ax.plot(np.arange(1, len(cum) + 1), cum, marker="o",
                    color=S.series(i), zorder=3)
            ax.annotate(f" {key[0][:5]} {key[1]}", xy=(len(cum), cum[-1]),
                        fontsize=7.5, color=S.series(i), va="center")
        ax.set_xlabel("components")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 101)
        S.title(ax, f"derivative {deriv}",
                "cumulative % of variance" if deriv == 0 else None)
    np.atleast_1d(axes)[0].set_ylabel("cumulative % of variance")
    fig.suptitle("how many components the spectra actually need",
                 x=0.02, ha="left", fontsize=13, color=S.ink("primary"))
    fig.tight_layout(rect=(0, 0, 0.87, 0.94))
    fig.savefig(out)
    plt.close(fig)


def _scatter(ax, x, y, values, palette, title, legend=None, seq=False):
    if seq:
        sc = ax.scatter(x, y, c=values, cmap=S.cmap_sequential(), s=14,
                        linewidths=0, zorder=3)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=S.ink("muted"))
    else:
        for i, v in enumerate(sorted(set(values.tolist()))):
            m = values == v
            ax.scatter(x[m], y[m], s=14, linewidths=0, zorder=3,
                       color=palette(i), label=str(legend[v] if legend else v))
        if legend is not None or len(set(values.tolist())) <= 6:
            ax.legend(fontsize=7, markerscale=1.2, loc="best")
    ax.grid(True)
    S.title(ax, title)


def fig_scores(res, data, deriv, mode, key, out):
    sc = res["scores"]
    ever, first = outcome_arrays(data["day"])
    evr = res["evr"] * 100
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0))
    lab = (f"PC1 ({evr[0]:.1f}%)", f"PC2 ({evr[1]:.1f}%)", f"PC3 ({evr[2]:.1f}%)")

    _scatter(axes[0, 0], sc[:, 0], sc[:, 1], data["variety"],
             lambda i: S.series(i), "coloured by variety",
             legend={v: config.VARIETY_NAMES[v - 1] for v in np.unique(data["variety"])})
    # The control, in the same coordinates: if this looks as clean as the panel
    # above it, the component is a plate signature wearing a variety label.
    _scatter(axes[0, 1], sc[:, 0], sc[:, 1], data["dish"], None,
             "coloured by dish — the confound control", seq=True)
    _scatter(axes[1, 0], sc[:, 0], sc[:, 1], ever, lambda i: S.series(i + 1),
             "germinated or never", legend={0: "never", 1: "germinated"})
    g = first > 0
    axes[1, 1].scatter(sc[~g, 0], sc[~g, 2], s=14, linewidths=0,
                       color=S.ink("grid"), zorder=2, label="never")
    s2 = axes[1, 1].scatter(sc[g, 0], sc[g, 2], c=first[g], cmap=S.cmap_sequential(),
                            s=16, linewidths=0, zorder=3)
    cb = fig.colorbar(s2, ax=axes[1, 1], fraction=0.046, pad=0.03)
    cb.set_label("germination day", fontsize=7.5, color=S.ink("secondary"))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=S.ink("muted"))
    axes[1, 1].legend(fontsize=7, loc="best")
    axes[1, 1].grid(True)
    S.title(axes[1, 1], "PC1 vs PC3, by germination day")

    for ax in axes.ravel()[:3]:
        ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
    axes[1, 1].set_xlabel(lab[0]); axes[1, 1].set_ylabel(lab[2])
    fig.suptitle(f"{mode} · {key} · derivative {deriv} · {res['n']} kernels",
                 x=0.02, ha="left", fontsize=13, color=S.ink("primary"))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out)
    plt.close(fig)


def fig_loadings(results, deriv, mode, out):
    wl = config.wavelengths()
    keys = [k for (d, m, k) in results if d == deriv and m == mode]
    fig, axes = plt.subplots(len(keys), 1, figsize=(9.6, 2.35 * len(keys)),
                             sharex=True)
    for ax, key in zip(np.atleast_1d(axes), sorted(keys, key=str)):
        r = results[(deriv, mode, key)]
        for a in range(4):
            ax.plot(wl, r["loadings"][a], color=S.series(a), lw=1.6, zorder=3,
                    label=f"PC{a + 1} ({r['evr'][a] * 100:.1f}%)")
        ax.axhline(0, color=S.ink("axis"), lw=0.8, zorder=1)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        S.title(ax, f"{key}")
    np.atleast_1d(axes)[-1].set_xlabel("wavelength (nm)")
    fig.suptitle(f"loadings — {mode}, derivative {deriv}",
                 x=0.02, ha="left", fontsize=13, color=S.ink("primary"))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out)
    plt.close(fig)


def fig_outliers(results, deriv, out):
    keys = sorted({(m, k) for (d, m, k) in results if d == deriv}, key=str)
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.0))
    for ax, (mode, key) in zip(axes.ravel(), keys):
        r = results[(deriv, mode, key)]
        ax.scatter(r["t2"], r["q"], s=14, linewidths=0, color=S.series(0), zorder=3)
        for cut, axis in ((np.percentile(r["t2"], 99), "v"),
                          (np.percentile(r["q"], 99), "h")):
            (ax.axvline if axis == "v" else ax.axhline)(
                cut, color=S.ink("muted"), lw=0.9, zorder=2)
        worst = int(np.argmax(r["q"]))
        ax.annotate(str(r["kernels"][worst]), xy=(r["t2"][worst], r["q"][worst]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7,
                    color=S.series(1))
        ax.set_xlabel(f"Hotelling T² ({N_PC_DIAG} PCs)")
        ax.set_ylabel("Q residual")
        ax.grid(True)
        S.title(ax, f"{mode[:5]} {key}")
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(f"outlier map, derivative {deriv} — lines are the 99th percentile",
                 x=0.02, ha="left", fontsize=13, color=S.ink("primary"))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=config.EXPLORATION / "pca" / "mean_spectra")
    args = p.parse_args()

    S.use("light")
    idx = index_mod.Index.load()
    spectra = np.load(config.SPECTRA_NPY)
    labels = germination.read_germination_labels(cell_map=germination.build_cell_map())
    mats, kernels, dropped, one_sided = build_matrices(idx, spectra, labels)
    print(f"{len(kernels)} kernels x 192 bands, side-averaged, "
          f"{len(mats)} matrices ({len(config.MODES)} modes x 0h/8h/delta)")
    if one_sided:
        print(f"  {len(one_sided)} kernel(s) averaged over one face only "
              f"(annotator missed the other): {', '.join(one_sided)}")
    if dropped:
        print(f"  dropped {len(dropped)} kernel(s) absent from a whole cell: "
              f"{', '.join(dropped)}")

    args.out.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for deriv in DERIVS:
        for (mode, key), data in sorted(mats.items(), key=str):
            Z, _, spec = pca.prepare(data["X"], deriv=deriv)
            fit = pca.fit(Z, N_PC)
            t2, q = pca.diagnostics(Z, fit, N_PC_DIAG)
            ever, first = outcome_arrays(data["day"])
            germ = first > 0
            r = {"fit": fit, "scores": fit["scores"], "loadings": fit["loadings"],
                 "evr": fit["evr"], "n": fit["n"], "t2": t2, "q": q,
                 "kernels": data["kernels"],
                 "eta_dish": pca.eta_squared(fit["scores"], data["dish"]),
                 "eta_variety": pca.eta_squared(fit["scores"], data["variety"]),
                 "eta_ever": pca.eta_squared(fit["scores"], ever),
                 "eta_day": pca.eta_squared(fit["scores"][germ], first[germ]),
                 # Within variety: variety is confounded with dish and
                 # correlated with germination day, so these are the only two
                 # columns that can be read as being about the grain.
                 "eta_dish_wv": pca.eta_squared(
                     pca.center_within(fit["scores"], data["variety"]), data["dish"]),
                 "eta_day_wv": pca.eta_squared(
                     pca.center_within(fit["scores"], data["variety"])[germ],
                     first[germ])}
            results[(deriv, mode, key)] = r
            for a in range(N_PC):
                rows.append({"deriv": deriv, "mode": mode, "view": key,
                             "pc": a + 1,
                             "evr_pct": round(100 * fit["evr"][a], 3),
                             "cum_evr_pct": round(100 * fit["cum_evr"][a], 2),
                             "eta2_dish": round(r["eta_dish"][a], 4),
                             "eta2_variety": round(r["eta_variety"][a], 4),
                             "eta2_germinated": round(r["eta_ever"][a], 4),
                             "eta2_germ_day": round(r["eta_day"][a], 4),
                             "eta2_dish_within_variety": round(r["eta_dish_wv"][a], 4),
                             "eta2_germ_day_within_variety": round(r["eta_day_wv"][a], 4)})
            fig_scores(r, data, deriv, mode, key,
                       args.out / f"scores_d{deriv}_{mode}_{key}.png")
        for mode in config.MODES:
            fig_loadings(results, deriv, mode,
                         args.out / f"loadings_d{deriv}_{mode}.png")
        fig_outliers(results, deriv, args.out / f"outliers_d{deriv}.png")
    fig_scree(results, args.out / "scree.png")

    with (args.out / "controls.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nmax over PC1-3 of the share of a component's variance that sits "
          "between groups")
    print(f"{'d':>2s} {'mode':13s} {'view':6s} {'PC1%':>6s} {'PC1-3%':>7s} | "
          f"{'dish':>6s} {'variety':>7s} {'germ':>6s} {'day':>6s} | "
          f"{'dish|var':>8s} {'day|var':>7s}")
    for (deriv, mode, key), r in sorted(results.items(), key=str):
        print(f"{deriv:2d} {mode:13s} {key:6s} {100 * r['evr'][0]:6.1f} "
              f"{100 * r['fit']['cum_evr'][2]:7.1f} | "
              f"{r['eta_dish'][:3].max():6.3f} {r['eta_variety'][:3].max():7.3f} "
              f"{r['eta_ever'][:3].max():6.3f} {r['eta_day'][:3].max():6.3f} | "
              f"{r['eta_dish_wv'][:3].max():8.3f} {r['eta_day_wv'][:3].max():7.3f}")
    write_summary(args.out, results, kernels, one_sided)
    print(f"\nfigures + controls.csv + summary.md -> {args.out}")
    return results


def write_summary(out, results, kernels, one_sided):
    def r(key):
        return results[key]
    L = ["# PCA of the mask-mean spectra", "",
         f"{len(kernels)} kernels x {config.N_BANDS} bands "
         f"({config.wavelengths()[0]:.0f}-{config.wavelengths()[-1]:.0f} nm), "
         "side-averaged so every matrix is exactly one row per kernel. Six "
         "matrices: two modes x {0 h, 8 h, 8 h - 0 h}. Two preprocessing "
         "variants: SNV, and SNV then a first derivative. Written by "
         "`python3 explore/pca_spectra.py`.", ""]
    if one_sided:
        L += [f"Two kernels are one-sided means, the annotator having missed the "
              f"other face: {', '.join(one_sided)}.", ""]
    L += ["## the controls, which decide whether any score plot means anything", "",
          "Each cell is the largest share, over PC1-PC3, of a component's "
          "variance that sits between the groups named. `dish|var` and "
          "`day|var` are computed after centring the scores within variety, so "
          "they are the parts that variety cannot explain.", "",
          "| d | mode | view | PC1 % | PC1-3 % | dish | variety | germ | day | "
          "dish\\|var | day\\|var |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(results, key=str):
        d, mode, view = key
        x = r(key)
        L.append(f"| {d} | {mode} | {view} | {100 * x['evr'][0]:.1f} | "
                 f"{100 * x['fit']['cum_evr'][2]:.1f} | "
                 f"{x['eta_dish'][:3].max():.3f} | {x['eta_variety'][:3].max():.3f} | "
                 f"{x['eta_ever'][:3].max():.3f} | {x['eta_day'][:3].max():.3f} | "
                 f"**{x['eta_dish_wv'][:3].max():.3f}** | "
                 f"**{x['eta_day_wv'][:3].max():.3f}** |")
    L += ["", "### how to read it", "",
          "* **PC1 takes 76-87% of the variance** in every matrix but one. After "
          "SNV, that is one dominant axis and a long thin tail — three "
          "components carry 93-98%.",
          "* **`dish` beats `variety` everywhere.** Variety is perfectly "
          "confounded with dish, so this is the decisive comparison: the leading "
          "components track the plate more closely than the cultivar. `dish|var` "
          "is the pure plate effect, and it reaches 0.41 on the reflectance "
          "delta — four tenths of a leading component's variance is which dish "
          "the kernel sat in, within a single variety.",
          "* **Nothing separates germinated from never.** `germ` never exceeds "
          "0.081. There is no unsupervised ever/never axis in these spectra, "
          "which is what the germination histograms would predict: 87-91% "
          "germinate, so the minority class is small and, on this evidence, not "
          "spectrally distinct.",
          "* **The one real germination signal is reflectance at 8 h.** `day` "
          "0.234 falls to `day|var` 0.155 once variety is removed — so about a "
          "third of it was variety — but 0.155 survives, and nothing else comes "
          "close. Eight hours of imbibition, seen in reflectance, is where to "
          "look.",
          "* **Transmittance is the cleaner mode.** Its plate effect is roughly "
          "half reflectance's (`dish|var` 0.107-0.306 against 0.187-0.409). It "
          "also carries less germination signal, so the two modes are not "
          "redundant.",
          "* **The delta is a disappointment.** 8 h - 0 h was the obvious place "
          "to look for imbibition, and it is the most dish-dominated matrix of "
          "the six. Differencing two captures cancels the kernel and keeps "
          "whatever drifted between sessions.",
          "* **The derivative changes little.** It shifts variance between "
          "components — reflectance 0 h PC1 drops from 81% to 34% — without "
          "changing any conclusion above. Both are kept; read the deriv-0 "
          "figures unless you want the loadings resolved.", "",
          "## figures", "",
          "* `scree.png` — cumulative variance, all six matrices, both derivatives.",
          "* `scores_d<k>_<mode>_<view>.png` — PC1/PC2 coloured by variety, by "
          "**dish** (the control, in the same coordinates), by germinated/never, "
          "and PC1/PC3 by germination day.",
          "* `loadings_d<k>_<mode>.png` — PC1-PC4 against wavelength.",
          "* `outliers_d<k>.png` — Hotelling T2 against Q residual. A high Q is "
          "a spectrum the components do not describe, which is what a mask that "
          "caught the retaining clip looks like; the worst kernel in each panel "
          "is named.", "",
          "`controls.csv` has every component, not just the best of PC1-3.", ""]
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
