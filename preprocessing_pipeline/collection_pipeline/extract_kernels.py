"""Cut every kernel out of every capture into one array a DataLoader can index.

    python3 extract_kernels.py --dry-run        # counts and sizes, writes nothing
    python3 extract_kernels.py                  # ~21 GB, one pass over the cubes
    python3 extract_kernels.py --dish 0 --dish 1

Input is grid_view/ (so run grid_index.py first) plus the cubes it was fitted
against. Output:

    kernel_patches/patches.npy    (N, 128, 64, 194) float16, memory-mappable
                   manifest.csv   one row per patch: what it is, where it came
                                  from, and how trustworthy it is
                   bands.json     band indices and wavelengths

WHY A WARP AND NOT A CROP
Each cell arrives as four corners ordered in PLATE coordinates, so warping them
onto a fixed rectangle does four jobs at once: it removes the plate's rotation
(a few degrees, different every capture), it removes the dorsal/ventral mirror,
it puts reflectance and transmittance on a common grid, and it makes day 0 h and
day 24 h pixel-comparable. An axis-aligned crop does none of those.

THE 194 CHANNELS
0..191   the pixel's spectrum after SNV (standard normal variate: subtract the
         pixel's own mean, divide by its own standard deviation). This is the
         usual NIR scatter correction -- it removes path-length and illumination
         differences between kernels and between days, which is what lets an
         8 h patch be compared with a 0 h one at all.
192      that pixel's pre-SNV mean
193      that pixel's pre-SNV standard deviation
SNV throws the overall level away, and for a water-uptake experiment the level
may itself be signal, so the two moments are kept rather than discarded. Use
them or ignore them; the choice stays with the model instead of being baked in
here.

TIME
The folder names are NOT in time order -- day1 is 0 h, day9 is 8 h, day2 is 24 h.
Nothing downstream should sort on the folder name; the manifest carries `hours`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gridfit import plate, render   # noqa: E402

# The whole point of this constant: day1/day2/day9 are slot names from the
# capture rig, not timestamps, and they sort into the wrong order.
HOURS = {1: 0.0, 9: 8.0, 2: 24.0}

PATCH_H, PATCH_W = 128, 64        # rows follow the plate's row axis
CELL_INSET = 0.15                 # crop the well interior; keeps the bright wall out
BAND_LO, BAND_HI = 8, 200         # [lo, hi): valid in all 150 transmittance captures
WL_LO, WL_HI, N_BANDS_RAW = 900.0, 1700.0, 224
OPEN_BEAM_OK = (0.8, 1.25)        # transmittance sanity band, as the pipeline uses


def wavelengths():
    return np.linspace(WL_LO, WL_HI, N_BANDS_RAW)[BAND_LO:BAND_HI]


# ------------------------------------------------------------------- warping --
def _warp_channels(img, M, size):
    """warpPerspective over an (h, w, C) stack -- cv2 takes at most 4 channels."""
    out = np.empty((size[1], size[0], img.shape[2]), np.float32)
    for i in range(0, img.shape[2], 4):
        chunk = img[:, :, i:i + 4]
        out[:, :, i:i + 4] = cv2.warpPerspective(
            chunk, M, size, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan).reshape(
                size[1], size[0], chunk.shape[2])
    return out


def _inset(cell, frac):
    ctr = np.asarray(cell["center"], float)
    return ctr + (np.asarray(cell["corners"], float) - ctr) * (1.0 - 2.0 * frac)


def cut_cell(cube, sat, width, corners):
    """One cell -> (PATCH_H, PATCH_W, B) float32 with NaN where invalid.

    Only the cell's bounding box is pulled off the memmap and moved into the
    working frame, so a 0.6 GB cube costs about 12 MB per kernel rather than a
    full transpose.
    """
    q = np.asarray(corners, float)
    x0 = int(max(np.floor(q[:, 0].min()) - 2, 0))
    y0 = int(max(np.floor(q[:, 1].min()) - 2, 0))
    x1 = int(min(np.ceil(q[:, 0].max()) + 2, width))
    y1 = int(min(np.ceil(q[:, 1].max()) + 2, cube.shape[1]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None

    # working[r, c] = cube[width - 1 - x, y]; see render.to_working
    sub = np.asarray(cube[width - x1:width - x0, y0:y1, BAND_LO:BAND_HI], np.float32)
    work = sub[::-1].transpose(1, 0, 2)

    dst = np.float32([[0, 0], [PATCH_W, 0], [PATCH_W, PATCH_H], [0, PATCH_H]])
    M = cv2.getPerspectiveTransform(np.float32(q - [x0, y0]), dst)
    patch = _warp_channels(work, M, (PATCH_W, PATCH_H))

    sat_frac = 0.0
    if sat is not None:
        s = np.asarray(sat[width - x1:width - x0, y0:y1, BAND_LO:BAND_HI])
        sw = s[::-1].transpose(1, 0, 2).astype(np.float32)
        sm = _warp_channels(sw, M, (PATCH_W, PATCH_H)) > 0.5
        sat_frac = float(sm.mean())
        # A railed voxel carries no information about how bright it really was,
        # so it must not be allowed into the pixel's SNV statistics.
        patch[sm] = np.nan
    return patch, sat_frac


def snv(patch):
    """(H, W, B) -> (H, W, B+2): SNV spectrum, then the pre-SNV mean and std."""
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(patch, axis=2)
        sd = np.nanstd(patch, axis=2)
    good = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-6)
    z = np.zeros_like(patch)
    np.divide(patch - mu[:, :, None], np.where(sd > 1e-6, sd, 1.0)[:, :, None],
              out=z, where=good[:, :, None])
    z[~np.isfinite(z)] = 0.0
    return (np.concatenate([z, np.nan_to_num(mu)[:, :, None],
                            np.nan_to_num(sd)[:, :, None]], axis=2),
            float(1.0 - good.mean()))


# ---------------------------------------------------------------- the walker --
def captures(grid_view, dishes):
    """Every fitted capture, newest fit wins. -> list of records."""
    out = []
    for p in sorted(grid_view.glob("*_images/day*/dish*/*/cells.json")):
        rec = json.loads(p.read_text())
        c = rec["capture"]
        if rec["status"] == "fail" or not rec["cells"]:
            print(f"  skipping {p.parent} -- fit status {rec['status']}")
            continue
        if dishes and c["dish"] not in dishes:
            continue
        out.append(rec)
    return out


def radiometry(src):
    """Open-beam T for a transmittance capture, or None. Flags the bad ones."""
    m = json.loads((Path(src) / "capture_meta.json").read_text())
    ob = m.get("open_beam") or {}
    return ob.get("median")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-view", type=Path, default=ROOT / "grid_view")
    p.add_argument("--out", type=Path, default=ROOT / "kernel_patches")
    p.add_argument("--dish", action="append", type=int, help="repeatable; default all")
    p.add_argument("--dry-run", action="store_true", help="count and size, write nothing")
    args = p.parse_args()

    recs = captures(args.grid_view, set(args.dish or []))
    if not recs:
        raise SystemExit(f"no fitted captures under {args.grid_view} -- run grid_index.py")
    n_patches = sum(sum(1 for c in r["cells"] if c["kind"] == "kernel") for r in recs)
    n_ch = (BAND_HI - BAND_LO) + 2
    nbytes = n_patches * PATCH_H * PATCH_W * n_ch * 2

    print(f"{len(recs)} captures -> {n_patches} kernel patches "
          f"({PATCH_H}x{PATCH_W}x{n_ch} float16, {nbytes / 1e9:.1f} GB)")
    print(f"bands {BAND_LO}-{BAND_HI - 1} = {wavelengths()[0]:.0f}-{wavelengths()[-1]:.0f} nm")
    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bands.json").write_text(json.dumps({
        "band_index": list(range(BAND_LO, BAND_HI)),
        "wavelength_nm": [round(w, 2) for w in wavelengths()],
        "channels": {"0..191": "SNV spectrum", "192": "pre-SNV pixel mean",
                     "193": "pre-SNV pixel std"},
        "patch": {"h": PATCH_H, "w": PATCH_W, "cell_inset_frac": CELL_INSET,
                  "axes": "rows follow the plate row axis, cols the column axis"},
        "hours_by_day_folder": HOURS,
    }, indent=1))

    X = np.lib.format.open_memmap(args.out / "patches.npy", mode="w+", dtype=np.float16,
                                  shape=(n_patches, PATCH_H, PATCH_W, n_ch))
    fields = ["row", "kernel_id", "dish", "cell", "cell_index", "day_folder", "hours",
              "mode", "side", "status", "occupancy", "sat_frac", "invalid_frac",
              "open_beam_T", "radiometry_ok", "source"]
    fh = (args.out / "manifest.csv").open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()

    row = 0
    checked = False
    for n, rec in enumerate(recs, 1):
        c = rec["capture"]
        cube = np.load(Path(c["source"]) / "capture.npy", mmap_mode="r")
        width = cube.shape[0]
        sat = None
        if c["mode"] == "transmittance":
            sat = np.load(Path(c["source"]) / "capture_masks.npz")["saturated"]
        ob = radiometry(c["source"]) if c["mode"] == "transmittance" else None
        ok = ob is None or (OPEN_BEAM_OK[0] <= ob <= OPEN_BEAM_OK[1])

        if not checked:
            _verify_frame(cube, width)
            checked = True

        for cell in rec["cells"]:
            if cell["kind"] != "kernel":
                continue
            patch, sat_frac = cut_cell(cube, sat, width, _inset(cell, CELL_INSET))
            if patch is None:
                print(f"  {c['source']} {cell['name']}: cell falls outside the frame")
                continue
            stack, invalid = snv(patch)
            X[row] = stack.astype(np.float16)
            w.writerow({
                "row": row, "kernel_id": f"dish{c['dish']}_{cell['name']}",
                "dish": c["dish"], "cell": cell["name"], "cell_index": cell["index"],
                "day_folder": c["day"], "hours": HOURS.get(c["day"], ""),
                "mode": c["mode"], "side": c["side"], "status": rec["status"],
                "occupancy": None if cell.get("occupancy") is None
                else round(cell["occupancy"], 4),
                "sat_frac": round(sat_frac, 4), "invalid_frac": round(invalid, 4),
                "open_beam_T": None if ob is None else round(ob, 3),
                "radiometry_ok": int(ok), "source": c["source"],
            })
            row += 1
        del cube, sat
        print(f"[{n:3d}/{len(recs)}] {c['mode'][:5]} day{c['day']} dish{c['dish']:<2d} "
              f"{c['side']:7s} -> {row} patches"
              + ("" if ok else f"   RADIOMETRY off: open beam T={ob:.2f}"))

    fh.close()
    X.flush()
    print(f"\npatches : {args.out / 'patches.npy'} {X.shape}")
    print(f"manifest: {args.out / 'manifest.csv'} ({row} rows)")
    print("NOTE day1=0h, day9=8h, day2=24h -- join on `hours`, never on the folder name.")


def _verify_frame(cube, width):
    """The bbox-slice-and-transpose above must equal render.to_working exactly.

    It is three index tricks deep and a silent error would mirror every patch,
    which is precisely the failure this whole pipeline exists to prevent. So it
    is checked against the reference implementation on real data, once per run.
    """
    x0, x1, y0, y1 = 40, 90, 60, 120
    full = render.to_working(np.asarray(cube[:, :, BAND_LO], np.float32))
    sub = np.asarray(cube[width - x1:width - x0, y0:y1, BAND_LO], np.float32)
    mine = sub[::-1].T
    if not np.array_equal(mine, full[y0:y1, x0:x1]):
        raise SystemExit("working-frame slice disagrees with render.to_working -- "
                         "the patch orientation cannot be trusted, refusing to write")
    print("  working-frame slice verified against render.to_working")


if __name__ == "__main__":
    main()
