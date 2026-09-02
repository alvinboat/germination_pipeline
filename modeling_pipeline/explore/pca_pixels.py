"""PCA of individual voxels: is there structure inside a kernel worth modelling?

    python3 explore/pca_pixels.py                   # -> exploration/pca/per_pixel/
    python3 explore/pca_pixels.py --px 60           # fewer pixels per view, quicker

THE QUESTION THIS ANSWERS AND THE MEAN SPECTRA CANNOT
`spectra.npy` is one mask-mean per view. Averaging over ~700 pixels is what
makes PLS possible and it is also, silently, a decision: everything that varies
ACROSS a kernel -- embryo against endosperm, the crease, a bruise -- is thrown
away before any model sees it. The CNN is the only path that could use it, and
it costs an order of magnitude more to train. So the number that matters here is
the variance split: of a component's total variance over all voxels, how much
sits BETWEEN kernels and how much WITHIN one. A component that is 95% between-
kernel is something the cheap mean spectrum already carries; a component with
real within-kernel variance is the case for per-pixel modelling, or the case
against it.

The row unit is the voxel, so unlike `pca_spectra.py` this does not average the
two faces -- it pools their pixels. Dorsal and ventral see different surfaces of
the same grain, and for a question about within-kernel structure that is more
coverage, not duplication.

SAMPLING
Every masked voxel of every view would be ~2.4 million rows per mode. Pixels are
sampled per view instead (`--px`, default 120), which keeps each matrix around
100k x 192 -- far past the point where another voxel moves a 192x192 covariance
-- and keeps every kernel equally represented rather than letting the biggest
masks dominate.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                     # noqa: E402
from barley import index as index_mod, pca, germination   # noqa: E402
from reporting import style as S                  # noqa: E402

DERIVS = (0, 1)
N_PC = 6
MAP_PC = 3          # components painted back onto a patch
MAP_KERNELS = 5     # example kernels per score-map figure


# --------------------------------------------------------------------- data --
ERODE_PX = (0, 2)       # boundary rings peeled off for the artefact check


def sample_voxels(idx, patches, masks, mode, hours, px, rng, erode=0):
    """-> (X, kernel_uid per row, dropped voxel count).

    Reads each view's patch once and keeps `px` masked voxels from it. Voxels
    with any non-finite band are dropped rather than imputed: in transmittance
    those are the railed open-beam pixels that `extract.cut` invalidated on
    purpose, and filling them in would put a made-up number into a component.
    """
    keep = (idx["mode"] == mode) & (idx["hours"] == hours)
    rows = idx.take(keep)
    out, uids, dropped = [], [], 0
    for i in range(len(rows)):
        r = int(rows["row"][i])
        m = np.asarray(masks[r])
        if erode:
            k = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
            m = cv2.erode(m.astype(np.uint8), k).astype(bool)
        flat = np.flatnonzero(m.ravel())
        if not len(flat):
            continue
        pick = rng.choice(flat, size=min(px, len(flat)), replace=False)
        v = np.asarray(patches[r], np.float32).reshape(-1, config.N_BANDS)[pick]
        ok = np.isfinite(v).all(axis=1)
        dropped += int((~ok).sum())
        out.append(v[ok])
        uids += [str(rows["kernel_uid"][i])] * int(ok.sum())
    return np.concatenate(out), np.array(uids, object), dropped


def patch_scores(patches, masks, row, res, deriv):
    """Project one whole patch's masked voxels. -> (scores image, mask)."""
    m = np.asarray(masks[row])
    v = np.asarray(patches[row], np.float32).reshape(-1, config.N_BANDS)
    flat = np.flatnonzero(m.ravel())
    ok = flat[np.isfinite(v[flat]).all(axis=1)]
    Z, _, _ = pca.prepare(v[ok], deriv=deriv)
    t = pca.project(Z, res)[:, :MAP_PC]
    img = np.full((config.PATCH_H * config.PATCH_W, MAP_PC), np.nan, np.float32)
    img[ok] = t
    return img.reshape(config.PATCH_H, config.PATCH_W, MAP_PC), m


# ------------------------------------------------------------------ figures --
def fig_scree(results, out):
    fig, axes = plt.subplots(1, len(DERIVS), figsize=(11.4, 4.6), sharey=True)
    for ax, deriv in zip(np.atleast_1d(axes), DERIVS):
        cells = sorted({(m, h) for (d, m, h, e) in results
                        if d == deriv and e == 0}, key=str)
        for i, key in enumerate(cells):
            cum = 100 * results[(deriv, *key, 0)]["fit"]["cum_evr"][:10]
            ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", color=S.series(i))
            ax.annotate(f" {key[0][:5]} {key[1]:g}h", xy=(len(cum), cum[-1]),
                        fontsize=7.5, color=S.series(i), va="center")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("components")
        ax.set_ylim(0, 101)
        S.title(ax, f"derivative {deriv}")
    np.atleast_1d(axes)[0].set_ylabel("cumulative % of voxel variance")
    fig.suptitle("components needed by individual voxels",
                 x=0.02, ha="left", fontsize=13, color=S.ink("primary"))
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    fig.savefig(out)
    plt.close(fig)


def fig_loadings(results, deriv, mode, out):
    wl = config.wavelengths()
    cells = sorted({h for (d, m, h, e) in results
                    if d == deriv and m == mode and e == 0})
    fig, axes = plt.subplots(len(cells), 1, figsize=(9.6, 2.5 * len(cells)),
                             sharex=True)
    for ax, h in zip(np.atleast_1d(axes), cells):
        r = results[(deriv, mode, h, 0)]
        for a in range(4):
            ax.plot(wl, r["fit"]["loadings"][a], color=S.series(a), lw=1.6,
                    label=f"PC{a + 1} ({r['fit']['evr'][a] * 100:.1f}%)")
        ax.axhline(0, color=S.ink("axis"), lw=0.8)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        S.title(ax, f"{h:g} h")
    np.atleast_1d(axes)[-1].set_xlabel("wavelength (nm)")
    fig.suptitle(f"voxel loadings — {mode}, derivative {deriv}",
                 x=0.02, ha="left", fontsize=13, color=S.ink("primary"))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def fig_maps(patches, masks, picks, res, deriv, title, out):
    """Scores painted back onto the patch. The only view of what a PC IS spatially.

    Row 0 is the grain itself, so a component can be read against the anatomy
    rather than against nothing -- which is how a boundary artefact gets told
    apart from a real embryo/endosperm contrast.
    """
    nrow, ncol = MAP_PC + 1, len(picks)
    cw = 1.35
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(cw * ncol + 0.85,
                                      cw * config.PATCH_H / config.PATCH_W * nrow + 0.7))
    for c, (uid, row, day) in enumerate(picks):
        img, m = patch_scores(patches, masks, row, res, deriv)
        p16 = np.asarray(patches[row], np.float32)
        with np.errstate(invalid="ignore"):
            grey = np.nanmean(np.where(np.isfinite(p16).any(2, keepdims=True),
                                       p16, 0.0), axis=2)
        ref = grey[m] if m.any() else grey
        lo, hi = np.nanpercentile(ref, [2, 98])
        panels = [np.where(m, np.clip((grey - lo) / max(hi - lo, 1e-9), 0, 1), np.nan)] \
            + [img[:, :, a] for a in range(MAP_PC)]
        # Crop to the mask. CELL_SCALE pads the whole well into the patch and the
        # grain is only about a tenth of that area, so an uncropped map is a
        # thumbnail of a kernel adrift in empty frame.
        ys, xs = np.nonzero(m)
        sl = (slice(max(ys.min() - 2, 0), ys.max() + 3),
              slice(max(xs.min() - 2, 0), xs.max() + 3))
        panels = [v[sl] for v in panels]
        m = m[sl]
        for a, v in enumerate(panels):
            ax = axes[a, c]
            if a == 0:
                ax.imshow(v, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                lim = np.nanpercentile(np.abs(v), 98)
                ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          interpolation="nearest")
            ax.contour(m.astype(float), levels=[.5], colors=[S.ink("primary")],
                       linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if a == 0:
                ax.set_title(f"{uid}\n{day}", fontsize=7, color=S.ink("secondary"))
            if c == 0:
                ax.set_ylabel("grain" if a == 0 else f"PC{a}", fontsize=8.5,
                              color=S.ink("secondary"))
    fig.suptitle(title, x=0.02, y=0.995, ha="left", va="top", fontsize=12,
                 color=S.ink("primary"))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.01,
                        wspace=0.04, hspace=0.1)
    fig.savefig(out, bbox_inches=None)
    plt.close(fig)


def fig_split(results, out, deriv=0):
    """Between-kernel share of each component's voxel variance, mask vs eroded."""
    cells = sorted({(m, h) for (d, m, h, e) in results if d == deriv}, key=str)
    fig, axes = plt.subplots(1, len(ERODE_PX), figsize=(12.4, 5.0), sharey=True)
    for ax, erode in zip(np.atleast_1d(axes), ERODE_PX):
        w = 0.8 / len(cells)
        for i, (mode, hours) in enumerate(cells):
            b = results[(deriv, mode, hours, erode)]["between"][:N_PC] * 100
            ax.bar(np.arange(N_PC) + i * w - 0.4, b, width=w * 0.92,
                   color=S.series(i), zorder=3,
                   label=f"{mode[:5]} {hours:g}h")
        ax.set_xticks(range(N_PC))
        ax.set_xticklabels([f"PC{a + 1}" for a in range(N_PC)])
        ax.set_ylim(0, 100)
        ax.axhline(50, color=S.ink("muted"), lw=1.0, zorder=2)
        S.hide_grid_x(ax)
        S.title(ax, "mask as drawn" if erode == 0 else f"mask eroded {erode} px")
    np.atleast_1d(axes)[0].set_ylabel(
        "% of the component's voxel variance that is BETWEEN kernels")
    np.atleast_1d(axes)[0].legend(fontsize=7.5)
    fig.suptitle("what the mask-mean throws away, and how much of it is the mask edge",
                 x=0.02, ha="left", fontsize=13, color=S.ink("primary"))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=config.EXPLORATION / "pca" / "per_pixel")
    p.add_argument("--px", type=int, default=120, help="voxels sampled per view")
    p.add_argument("--map-deriv", type=int, default=0,
                   help="derivative used for the score maps")
    args = p.parse_args()

    t0 = time.time()
    S.use("light")
    rng = np.random.default_rng(config.SEED)
    idx = index_mod.Index.load()
    patches = np.load(config.PATCHES_NPY, mmap_mode="r")
    masks = np.load(config.MASKS_NPY, mmap_mode="r")
    labels = germination.read_germination_labels(cell_map=germination.build_cell_map())
    args.out.mkdir(parents=True, exist_ok=True)

    # Sampled twice: as the masks stand, and with two boundary rings peeled off.
    # The score maps show PC1 lighting up the mask rim, which is what partial
    # -volume mixing of grain and well looks like rather than anything in the
    # grain. If that is what it is, eroding should remove a large share of the
    # voxel variance and push the rest towards being between-kernel.
    sampled = {}
    for erode in ERODE_PX:
        for mode in config.MODES:
            for hours in sorted(config.HOURS.values()):
                X, uids, dropped = sample_voxels(idx, patches, masks, mode, hours,
                                                 args.px, rng, erode)
                sampled[(mode, hours, erode)] = (X, uids)
                print(f"{mode:13s} {hours:g}h  erode {erode}px  {X.shape[0]:7d} "
                      f"voxels from {len(set(uids.tolist()))} kernels"
                      f"  ({dropped} non-finite dropped)", flush=True)

    results, rows = {}, []
    for deriv in DERIVS:
        for (mode, hours, erode), (X, uids) in sorted(sampled.items(), key=str):
            Z, _, _ = pca.prepare(X, deriv=deriv)
            fit = pca.fit(Z, N_PC)
            between = pca.eta_squared(fit["scores"], uids)
            results[(deriv, mode, hours, erode)] = {
                "fit": fit, "between": between, "total": fit["total_variance"]}
            for a in range(N_PC):
                rows.append({"deriv": deriv, "mode": mode, "hours": hours,
                             "erode_px": erode, "pc": a + 1,
                             "evr_pct": round(100 * fit["evr"][a], 3),
                             "cum_evr_pct": round(100 * fit["cum_evr"][a], 2),
                             "total_variance": round(fit["total_variance"], 4),
                             "between_kernel_pct": round(100 * between[a], 2),
                             "within_kernel_pct": round(100 * (1 - between[a]), 2)})
        for mode in config.MODES:
            fig_loadings(results, deriv, mode,
                         args.out / f"loadings_d{deriv}_{mode}.png")
    fig_scree(results, args.out / "scree.png")
    fig_split(results, args.out / "variance_split.png")

    # Score maps: a spread of germination outcomes, same kernels in every panel
    # so the maps can be read against each other rather than against nothing.
    for mode in config.MODES:
        for hours in sorted(config.HOURS.values()):
            sel = idx.take((idx["mode"] == mode) & (idx["hours"] == hours)
                           & (idx["side"] == config.SIDES[0]))
            picks, want = [], [1, 2, 3, 5, None]
            for target in want:
                for i in range(len(sel)):
                    uid = str(sel["kernel_uid"][i])
                    if labels.get(uid, germination.UNSCORED) == target and \
                            uid not in {u for u, _, _ in picks}:
                        picks.append((uid, int(sel["row"][i]),
                                      "never" if target is None else f"day {target}"))
                        break
            fig_maps(patches, masks, picks[:MAP_KERNELS],
                     results[(args.map_deriv, mode, hours, 0)]["fit"], args.map_deriv,
                     f"{mode} · {hours:g} h · voxel scores painted back onto the "
                     f"patch (derivative {args.map_deriv})",
                     args.out / f"score_maps_{mode}_{hours:g}h.png")

    with (args.out / "variance_split.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nshare of each component's voxel variance that is BETWEEN kernels")
    print(f"{'d':>2s} {'mode':13s} {'h':>3s} {'erode':>5s} {'totvar':>7s} | " +
          " ".join(f"{'PC' + str(a + 1):>7s}" for a in range(4)))
    for (deriv, mode, hours, erode), r in sorted(results.items(), key=str):
        print(f"{deriv:2d} {mode:13s} {hours:3.0f} {erode:5d} {r['total']:7.2f} | " +
              " ".join(f"{100 * r['between'][a]:7.1f}" for a in range(4)))
    write_summary(args.out, results, args.px)
    print(f"\nfigures + variance_split.csv + summary.md -> {args.out}  "
          f"({time.time() - t0:.0f}s)")


def write_summary(out, results, px):
    L = ["# PCA of individual voxels", "",
         f"{px} masked voxels sampled per view, both faces pooled, four matrices "
         "(two modes x 0 h / 8 h) of roughly 100k x 192. Each matrix is fitted "
         "twice: on the mask as drawn, and on the mask with two boundary rings "
         "eroded. Written by `python3 explore/pca_pixels.py`.", "",
         "## the number this was built for", "",
         "`spectra.npy` averages ~700 voxels into one row. This asks what that "
         "average destroys: of a component's variance over all voxels, how much "
         "is BETWEEN kernels — already carried by the cheap mean spectrum — and "
         "how much is WITHIN one, visible only to a per-pixel model.", "",
         "| d | mode | h | erode | total variance | PC1 between-kernel % | PC2 | PC3 |",
         "|---|---|---|---|---|---|---|---|"]
    for key in sorted(results, key=str):
        d, mode, hours, erode = key
        x = results[key]
        L.append(f"| {d} | {mode} | {hours:g} | {erode} px | {x['total']:.2f} | "
                 f"{100 * x['between'][0]:.1f} | {100 * x['between'][1]:.1f} | "
                 f"{100 * x['between'][2]:.1f} |")
    L += ["", "### how to read it", "",
          "* **On the mask as drawn, most voxel variance is within-kernel.** "
          "Reflectance at 0 h puts only 10.6% of PC1 between kernels; the other "
          "89% is variation across the face of a single grain.",
          "* **Most of that is the mask edge, not the grain.** Peeling two "
          "pixel rings off takes reflectance's total voxel variance from 13.07 "
          "to 5.53 at 0 h and 7.24 to 3.78 at 8 h — **48-58% of all voxel "
          "variance lived in the outermost two pixels of the mask** — and the "
          "between-kernel share of PC1 roughly doubles, 10.6% to 21.7% and "
          "33.2% to 53.3%. That is the signature of partial-volume mixing: the "
          "boundary voxels are part grain, part well.",
          "* `score_maps_*.png` show it directly. PC1 paints a ring around the "
          "mask rim and the two tips. PC3 is the one that looks anatomical — an "
          "end-to-end gradient, embryo against distal.",
          "* **Transmittance barely cares.** Its total variance falls only "
          "14-15% under the same erosion, because a transmission image has no "
          "hard rim shadow to mix into. It is the mode whose masks can be "
          "trusted at the edge.", "",
          "## what follows from it", "",
          f"* `config.MASK_ERODE_PX` is currently "
          f"{'off' if not config.MASK_ERODE_PX else str(config.MASK_ERODE_PX) + ' px'}"
          ". It was held back on the grounds that 2 px cost a median 23.5% of "
          "mask area for a contaminant nobody had measured. The contaminant is "
          "now measured, and in reflectance it is about half of all voxel "
          "variance. That trade is worth revisiting — for reflectance "
          "specifically.",
          "* The case for per-pixel modelling is weaker than the raw "
          "within-kernel share suggests. Once the boundary is removed, "
          "reflectance at 8 h and both transmittance matrices are majority "
          "between-kernel on PC1, meaning the mask-mean already carries most of "
          "what the leading components see.",
          "* Non-finite voxels are dropped, never imputed. Erosion removes most "
          "of them too (transmittance 8 h: 15,555 dropped at 0 px, 6,492 at "
          "2 px), which is a second, independent sign that the bad voxels are "
          "an edge phenomenon.", "",
          "## figures", "",
          "* `scree.png`, `loadings_d<k>_<mode>.png` — as drawn (no erosion).",
          "* `score_maps_<mode>_<h>h.png` — PC1-PC3 painted back onto five "
          "kernels spanning day 1 to never, cropped to the mask, with the grain "
          "itself on the top row.",
          "* `variance_split.png` — the between-kernel share, mask as drawn "
          "beside mask eroded.", "",
          "`variance_split.csv` has every component and both erosions.", ""]
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
