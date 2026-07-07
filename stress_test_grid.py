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


def perturb(gray8, rng, max_rot, max_shift, max_contrast, max_bright, max_noise):
    """Apply one random rotation+shift+brightness/contrast+noise draw to gray8."""
    H, W = gray8.shape
    angle = rng.uniform(-max_rot, max_rot)
    dx = rng.uniform(-max_shift, max_shift)
    dy = rng.uniform(-max_shift, max_shift)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
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
    """build_grid + label_cells -> (n_usable, usable_label_set, meta) or None on failure.

    "Usable" (not clipped, not marker-occupied) is the ground-truth invariant: a
    clipped cell isn't a real kernel position (it's a lattice position the rim
    or frame partially cuts), so it flickering in/out under perturbation is
    expected and harmless. Only usable-cell identity must be stable.
    """
    try:
        cells, meta = build_grid(gray8)
    except SystemExit as e:
        return None, str(e)
    label_cells(cells, meta)
    usable = frozenset(c["label"] for c in cells if not c["clipped"] and c["marker_id"] is None)
    return (len(usable), usable, meta), None


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
    base_n, base_labels, base_meta = base_result
    print(f"Baseline cells: {base_n}  dish {base_meta['dish']}  "
          f"rotation {base_meta['angle_deg']}°  labels {len(base_labels)}")

    rng = np.random.default_rng(args.seed)
    n_ok = n_count_mismatch = n_label_mismatch = n_hard_fail = 0
    dish_centers, radii, rotations = [], [], []

    for t in range(1, args.trials + 1):
        pert_gray, params = perturb(gray8, rng, args.max_rot, args.max_shift,
                                     args.max_contrast, args.max_bright, args.max_noise)
        result, err = run_once(pert_gray)
        tag = (f"rot={params['angle']:+.2f} shift=({params['dx']:+.1f},{params['dy']:+.1f}) "
               f"contrast={params['contrast']:.2f} bright={params['bright']:+.1f}")
        if result is None:
            n_hard_fail += 1
            print(f"  [{t:2d}] HARD FAIL ({err})  {tag}")
            continue
        n, labels, meta = result
        cx, cy, r = meta["dish"]
        dish_centers.append((cx, cy))
        radii.append(r)
        rotations.append(meta["angle_deg"])
        missing = base_labels - labels
        extra = labels - base_labels
        if n != base_n:
            n_count_mismatch += 1
            print(f"  [{t:2d}] CELL COUNT {n} != baseline {base_n}  {tag}")
        elif missing or extra:
            n_label_mismatch += 1
            print(f"  [{t:2d}] LABEL MISMATCH missing={sorted(missing)} "
                  f"extra={sorted(extra)}  {tag}")
        else:
            n_ok += 1

    print()
    print(f"{n_ok}/{args.trials} trials matched baseline exactly "
          f"(same count + same label set).")
    if n_count_mismatch:
        print(f"{n_count_mismatch} trials had a different total cell count.")
    if n_label_mismatch:
        print(f"{n_label_mismatch} trials had the same count but different labels.")
    if n_hard_fail:
        print(f"{n_hard_fail} trials failed detection outright (see above).")
    if dish_centers:
        cxs, cys = zip(*dish_centers)
        print(f"dish center std: ({np.std(cxs):.2f}, {np.std(cys):.2f}) px  "
              f"radius std: {np.std(radii):.2f} px  rotation std: {np.std(rotations):.3f}°")


if __name__ == "__main__":
    main()
