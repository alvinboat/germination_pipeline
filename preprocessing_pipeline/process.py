"""Process the capture currently sitting in raw_image/.

Normal use is zero-argument. Set the four variables in config.py, then:

    python3 process.py

It reads every .bin in raw_image/, corrects it for the mode named in
config.CAPTURE, and writes

    <mode>_images/day<DAY>/dish<DISH_NUMBER>/<side>/capture.npy      corrected cube
                                                    capture_meta.json
                                                    capture_masks.npz  (transmittance)
    raw_binaries/<mode>_images/day.../capture.bin                    raw archive
                                              capture.json
    preview/<mode>_day<DAY>_dish<DISH_NUMBER>_<side>.png             preview

and then empties raw_image/ so the next capture can be dropped straight in.

preview/ holds exactly one PNG per capture. The optional diagnostics
(capture_checkerboard_qc.png, capture_saturation.png -- both off in config.py)
are written beside the cube instead, never into preview/.
Re-imaging the same dish overwrites that dish's files in place.

raw_image/ is cleared LAST, after every output is on disk. A run that fails
anywhere leaves the raw capture untouched, so it can simply be run again.

Escape hatches, none needed for the normal flow:
    --image PATH        process something else -- a capture directory, a stitched
                        .npy, or an archived capture.bin. Skips archiving and
                        skips clearing raw_image/, so it is safe for reprocessing.
    --keep-raw-image    do everything, but leave raw_image/ full.
    --day/--capture/--dish/--side   override config.py for one run.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import common  # noqa: E402
from common import check_space, json_safe, line_files, stitch  # noqa: E402

RAW_IMAGE_DIR = ROOT / "raw_image"
PREVIEW_DIR = ROOT / "preview"

MODES = {
    "TRANSMITTANCE": {"folder": "transmittance_images", "stem": "transmittance",
                      "scripts": "transmittance_scripts"},
    "REFLECTANCE": {"folder": "reflectance_images", "stem": "reflectance",
                    "scripts": "reflectance_scripts"},
}
SIDES = ("DORSAL", "VENTRAL")


def load_mode(name, path):
    """Import a mode's correct.py under a unique module name.

    Both modes' modules are called correct.py, so they are loaded by path rather
    than by a sys.path search -- one run only ever needs one of them, and this
    makes it impossible for the wrong one to be found first.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_settings(cfg, args):
    """Validate the four per-capture variables. -> (day, capture, dish, side).

    Checked rather than trusted, because every one of them is a hand-edited value
    that names an output path. A typo in CAPTURE_SIDE silently writes a dorsal
    scan into the ventral folder, and nothing downstream could ever tell.
    """
    day = args.day if args.day is not None else cfg.DAY
    capture = (args.capture if args.capture is not None else cfg.CAPTURE)
    dish = args.dish if args.dish is not None else cfg.DISH_NUMBER
    side = (args.side if args.side is not None else cfg.CAPTURE_SIDE)

    try:
        day = int(day)
    except (TypeError, ValueError):
        raise SystemExit(f"DAY must be a whole number, got {day!r}.")
    if day < 1:
        raise SystemExit(f"DAY must be 1 or greater, got {day}.")

    capture = str(capture).strip().upper()
    if capture not in MODES:
        raise SystemExit(f"CAPTURE must be one of {', '.join(MODES)}, got {capture!r}.")

    try:
        dish = int(dish)
    except (TypeError, ValueError):
        raise SystemExit(f"DISH_NUMBER must be a whole number, got {dish!r}.")
    lo, hi = cfg.DISH_RANGE
    if not lo <= dish <= hi:
        raise SystemExit(f"DISH_NUMBER must be {lo}-{hi}, got {dish}.")

    side = str(side).strip().upper()
    if side not in SIDES:
        raise SystemExit(f"CAPTURE_SIDE must be one of {', '.join(SIDES)}, got {side!r}.")
    return day, capture, dish, side


def load_input(path, workers, archive_path):
    """Read the capture: a directory of .bin, an archived capture.bin, or an .npy.

    -> (cube, blank_line_indices, info)
    """
    path = Path(path)
    if path.is_dir():
        files = line_files(path)
        if not files:
            raise SystemExit(
                f"{path} holds no .bin files. Save the capture into {RAW_IMAGE_DIR.name}/ "
                f"first, or pass --image.")
        print(f"Stitching {len(files)} lines from {path}/ ...")
        t = time.time()
        cube, blank, info = stitch(path, workers=workers, archive_path=archive_path)
        print(f"  cube {cube.shape} uint16 in {time.time() - t:.1f}s"
              + (f", {len(blank)} blank line(s) at {common.runs(blank)}" if blank else ""))
        return cube, blank, info
    if path.suffix == ".bin":
        print(f"Reading archived capture {path} ...")
        cube = common.load_capture_bin(path)
        print(f"  cube {cube.shape} uint16")
        return cube, [], {"n_lines": int(cube.shape[1]),
                          "raw_bytes": path.stat().st_size, "n_blank": 0}
    if path.suffix == ".npy":
        print(f"Loading stitched cube {path} ...")
        cube = np.load(path)
        print(f"  cube {cube.shape} {cube.dtype}")
        return cube, [], {"n_lines": int(cube.shape[1]),
                          "raw_bytes": path.stat().st_size, "n_blank": 0}
    raise SystemExit(f"{path} is neither a capture directory, a capture.bin, nor an .npy.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=None,
                   help="process this instead of raw_image/ (a capture dir, an archived "
                        "capture.bin, or a stitched .npy). Skips archiving and skips "
                        "clearing raw_image/.")
    p.add_argument("--keep-raw-image", action="store_true",
                   help="leave raw_image/ full instead of clearing it at the end.")
    p.add_argument("--day", type=int, default=None, help="override config.DAY")
    p.add_argument("--capture", default=None, help="override config.CAPTURE")
    p.add_argument("--dish", type=int, default=None, help="override config.DISH_NUMBER")
    p.add_argument("--side", default=None, help="override config.CAPTURE_SIDE")
    args = p.parse_args()

    import config as cfg
    common.reset_warnings()
    t_start = time.time()

    day, capture, dish, side = resolve_settings(cfg, args)
    mode = MODES[capture]
    rel = Path(f"day{day}") / f"dish{dish}" / side.lower()
    stem = f"{mode['stem']}_day{day}_dish{dish}_{side.lower()}"

    out_dir = ROOT / mode["folder"] / rel
    raw_dir = Path(cfg.RAW_BINARIES_ROOT)
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    raw_dir = raw_dir / mode["folder"] / rel

    cube_path = out_dir / "capture.npy"
    masks_path = out_dir / "capture_masks.npz"
    meta_path = out_dir / "capture_meta.json"
    preview_path = PREVIEW_DIR / f"{stem}.png"
    # Diagnostics live with the capture they describe, not in preview/, which is
    # one PNG per capture and nothing else. Both are off by default.
    sat_path = out_dir / "capture_saturation.png"
    qc_path = out_dir / "capture_checkerboard_qc.png"

    print("=" * 78)
    print(f"{capture}  day {day}  dish {dish}  {side}")
    print("=" * 78)
    print(f"  cube    -> {cube_path}")
    print(f"  raw     -> {raw_dir / 'capture.bin'}")
    print(f"  preview -> {preview_path}")
    if cube_path.exists():
        print(f"  NOTE: this capture already exists and will be OVERWRITTEN "
              f"(re-image of day {day} dish {dish} {side.lower()}).")

    source = Path(args.image) if args.image else RAW_IMAGE_DIR
    archiving = args.image is None
    if source.is_dir():
        files = line_files(source)
        if not files:
            raise SystemExit(f"{source}/ is empty -- nothing to process. Save the capture "
                             f"into {RAW_IMAGE_DIR.name}/ first.")
        raw_bytes = sum((source / f).stat().st_size for f in files)
        check_space(len(files), raw_bytes if archiving else 0,
                    [out_dir, raw_dir] if archiving else [out_dir], cfg.MIN_FREE_GB)

    out_dir.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    cube, blank, info = load_input(source, cfg.STITCH_WORKERS,
                                   (raw_dir / "capture.bin") if archiving else None)
    t_stitch = time.time()

    mod = load_mode(f"{mode['stem']}_correct", ROOT / mode["scripts"] / "correct.py")
    corrected, masks, mode_meta = mod.process(cube, blank, day, cfg, source,
                                              preview_path, sat_path, qc_path)
    del cube
    t_correct = time.time()

    print("Writing:")
    np.save(cube_path, corrected)
    print(f"  cube: {cube_path} {corrected.shape} {corrected.dtype} "
          f"({corrected.nbytes / 1e6:.0f} MB)")
    if masks is not None:
        np.savez_compressed(masks_path, **masks)
        print(f"  masks: {masks_path} ({', '.join(sorted(masks))})")
    elif masks_path.exists():
        # A mode switch on the same dish would otherwise leave the other mode's
        # masks sitting next to a cube they do not describe.
        masks_path.unlink()

    meta = {"capture": {"day": day, "mode": capture, "dish": dish, "side": side,
                        "stem": stem},
            "source": str(source), "raw": info,
            "outputs": {"cube": str(cube_path), "meta": str(meta_path),
                        "masks": str(masks_path) if masks is not None else None,
                        "preview": str(preview_path),
                        "saturation_map": str(sat_path) if sat_path.exists() else None,
                        "checkerboard_qc": str(qc_path) if qc_path.exists() else None},
            "cube_shape": list(corrected.shape), "cube_dtype": str(corrected.dtype),
            "settings": {k: getattr(cfg, k) for k in dir(cfg)
                         if k.isupper() and not k.startswith("_")},
            "timing_s": {"stitch": round(t_stitch - t_start, 1),
                         "correct": round(t_correct - t_stitch, 1)},
            **mode_meta}
    meta["warnings"] = list(common.WARNINGS)
    meta_path.write_text(json.dumps(json_safe(meta), indent=2))
    print(f"  meta: {meta_path}")

    # Last, and only now: every output above is on disk, so clearing the capture
    # cannot lose data. A failure anywhere earlier leaves raw_image/ intact and
    # the run simply repeatable.
    if archiving and not args.keep_raw_image:
        removed = 0
        for f in line_files(RAW_IMAGE_DIR):
            (RAW_IMAGE_DIR / f).unlink()
            removed += 1
        print(f"  cleared {removed} .bin file(s) from {RAW_IMAGE_DIR.name}/ "
              f"-- ready for the next capture "
              f"(archived at {raw_dir / 'capture.bin'})")
    elif args.keep_raw_image:
        print(f"  {RAW_IMAGE_DIR.name}/ left as-is (--keep-raw-image)")

    n = len(common.WARNINGS)
    print("=" * 78)
    print(f"Done in {time.time() - t_start:.0f}s. {n} warning(s)."
          + ("" if n else " Nothing flagged."))
    if n:
        for w in common.WARNINGS:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
