"""
Stress-test detect_grid.py's dish/lattice/cell detection against the kind of
day-to-day variation expected across a 5-day, ~20-scans/day series.

We don't have real multi-day captures yet, so this perturbs the one known-good
greyscale render (grain_ref_exp_2500, corrected) with synthetic variation that
approximates what will actually change between scans:
    - rotation      dish re-plated at a slightly different angle
    - translation   dish not perfectly re-centred on the stage
    - brightness/contrast   illumination / exposure drift
    - gaussian noise        sensor noise

The camera/belt is fixed; only the dish's own placement varies, so rotation and
translation are two independent degrees of freedom of the DISH, not of the
frame -- perturb() pivots the rotation on the baseline dish center (not the
image center) so a "rotation" trial doesn't also drag the dish sideways as a
side effect of a mismatched pivot.

and re-runs build_grid() + label_cells() on every variant. The invariant that
actually matters for tracking kernels across days is not "usable count" (a
cell can legitimately flip clipped<->usable near the rim) but:
    1. total cell count matches the baseline (no physical kernel goes missing)
    2. the exact set of marker-locked labels matches the baseline (a kernel's
       identity/label must not shift, or day N+1 data gets attached to the
       wrong kernel)

Run:
    python3 stress_test_grid.py                       # default corrected cube
    python3 stress_test_grid.py <cube.npy> --trials 30
    python3 stress_test_grid.py <cube.npy> --seed 1 --max-rot 3 --max-shift 15
"""

import argparse
import sys

import cv2
import numpy as np

from detect_grid import DEFAULT_CUBE, build_grid, label_cells, load_gray


def perturb(gray8, rng, max_rot, max_shift, max_contrast, max_bright, max_noise, center=None):
    """Apply one random rotation+shift+brightness/contrast+noise draw to gray8.

    center: pivot for the rotation, in (x, y) image coords -- pass the baseline
    dish center here, not the image center. The camera/belt is fixed; only the
    dish's own placement (its angle and position) varies day to day, so the
    rotation must pivot on the dish's own center. Pivoting on the image center
    instead (the dish sits ~49px off it on grain_ref_exp_2500) drags the whole
    dish sideways as a side effect of "rotation" alone -- e.g. +2.6px at the
    default 3 degree max -- conflating the two perturbations this function is
    meant to vary independently.
    """
    H, W = gray8.shape
    if center is None:
        center = (W / 2, H / 2)
    angle = rng.uniform(-max_rot, max_rot)
    dx = rng.uniform(-max_shift, max_shift)
    dy = rng.uniform(-max_shift, max_shift)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    warped = cv2.warpAffine(gray8, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

    contrast = rng.uniform(1 - max_contrast, 1 + max_contrast)
    bright = rng.uniform(-max_bright, max_bright)
    noisy = warped.astype(np.float32) * contrast + bright
    noisy += rng.normal(0, max_noise, size=warped.shape)
    out = np.clip(noisy, 0, 255).astype(np.uint8)
    return out, {"angle": angle, "dx": dx, "dy": dy, "contrast": contrast, "bright": bright}


def run_once(gray8):
    """build_grid + label_cells -> (all_labels, usable_labels, meta) or None on failure.

    Tracks every kernel-slot cell's label, clipped or not -- not just the usable
    subset (that was the bug: a cell flipping clipped<->usable near the rim is
    documented above as expected and harmless, but comparing only the usable set
    treated that exact flip as a failure). What actually has to hold:
      - no label present in the baseline's full lattice may vanish entirely
        (a physical kernel position lost, not just downgraded to clipped)
      - no label may appear as usable that wasn't even a valid baseline
        position (a mislabelled / lattice-scrambled cell)
    A cell moving between usable and clipped satisfies both and is not a failure.
    """
    try:
        cells, meta = build_grid(gray8)
    except SystemExit as e:
        return None, str(e)
    label_cells(cells, meta)
    all_labels = frozenset(c["label"] for c in cells if c["marker_id"] is None)
    usable_labels = frozenset(c["label"] for c in cells if not c["clipped"] and c["marker_id"] is None)
    return (all_labels, usable_labels, meta), None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?", default=DEFAULT_CUBE, help="Corrected *_cube.npy.")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rot", type=float, default=3.0, help="deg")
    p.add_argument("--max-shift", type=float, default=15.0, help="px")
    p.add_argument("--max-contrast", type=float, default=0.15, help="fraction")
    p.add_argument("--max-bright", type=float, default=20.0, help="DN, 0-255 scale")
    p.add_argument("--max-noise", type=float, default=6.0, help="gaussian sigma, DN")
    args = p.parse_args()

    gray8 = load_gray(args.src)
    print(f"Baseline: {args.src}  {gray8.shape[1]}x{gray8.shape[0]}")

    base_result, base_err = run_once(gray8)
    if base_result is None:
        sys.exit(f"baseline detection itself failed: {base_err}")
    base_all, base_usable, base_meta = base_result
    print(f"Baseline: {len(base_all)} lattice positions, {len(base_usable)} usable  "
          f"dish {base_meta['dish']}  rotation {base_meta['angle_deg']}°")

    rng = np.random.default_rng(args.seed)
    n_ok = n_lost = n_spurious = n_hard_fail = 0
    n_flicker = 0  # informational only: usable<->clipped flips, not a failure
    dish_centers, radii, rotations = [], [], []
    dish_center = tuple(base_meta["dish"][:2])

    for t in range(1, args.trials + 1):
        pert_gray, params = perturb(gray8, rng, args.max_rot, args.max_shift,
                                     args.max_contrast, args.max_bright, args.max_noise,
                                     center=dish_center)
        result, err = run_once(pert_gray)
        tag = (f"rot={params['angle']:+.2f} shift=({params['dx']:+.1f},{params['dy']:+.1f}) "
               f"contrast={params['contrast']:.2f} bright={params['bright']:+.1f}")
        if result is None:
            n_hard_fail += 1
            print(f"  [{t:2d}] HARD FAIL ({err})  {tag}")
            continue
        all_labels, usable_labels, meta = result
        cx, cy, r = meta["dish"]
        dish_centers.append((cx, cy))
        radii.append(r)
        rotations.append(meta["angle_deg"])

        lost = base_all - all_labels               # a physical kernel slot vanished -- real failure
        spurious = usable_labels - base_all         # usable label invalid in baseline -- real failure
        flicker = base_usable.symmetric_difference(usable_labels) - lost - spurious

        if lost or spurious:
            n_lost += bool(lost)
            n_spurious += bool(spurious)
            print(f"  [{t:2d}] LOST={sorted(lost)} SPURIOUS={sorted(spurious)}  {tag}")
        else:
            n_ok += 1
            if flicker:
                n_flicker += 1
                print(f"  [{t:2d}] ok, usable<->clipped flicker only: {sorted(flicker)}  {tag}")

    print()
    print(f"{n_ok}/{args.trials} trials had no lost/spurious kernel positions "
          f"(the invariant that actually matters).")
    if n_flicker:
        print(f"{n_flicker} of those were clean except for expected usable<->clipped "
              f"flicker near the rim.")
    if n_lost:
        print(f"{n_lost} trials LOST a baseline kernel position entirely.")
    if n_spurious:
        print(f"{n_spurious} trials produced a SPURIOUS usable label not in the baseline lattice.")
    if n_hard_fail:
        print(f"{n_hard_fail} trials failed detection outright (see above).")
    if dish_centers:
        cxs, cys = zip(*dish_centers)
        print(f"dish center std: ({np.std(cxs):.2f}, {np.std(cys):.2f}) px  "
              f"radius std: {np.std(radii):.2f} px  rotation std: {np.std(rotations):.3f}°")


if __name__ == "__main__":
    main()
