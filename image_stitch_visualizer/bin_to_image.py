"""
Dispatcher: convert every (or one named) raw .bin capture directory under
raw_image_bin/ into a pseudo image PNG under pseudo_image/, routing each
capture to the reflectance or transmittance pipeline by name:

    'ref'  in the folder name  -> bin_to_image_ref.convert_ref   (checkerboard
                                   scan-scale correction applied)
    'tran' in the folder name  -> bin_to_image_trans.convert_trans (plain
                                   stitch + render for now)

A folder matching neither is skipped with a warning rather than guessed at.

Layout:
    image_stitch_visualizer/
        raw_image_bin/<capture_name>/*.bin   input -- one file per scanned line
        pseudo_image/<capture_name>.png      output
        hsi_save_load.py                     Mono12Packed codec (unchanged copy)
        stitching.py                         shared stitch/render (numpy only)
        bin_to_image_ref.py                  reflectance pipeline (needs OpenCV)
        bin_to_image_trans.py                transmittance pipeline (numpy only)
        bin_to_image.py                      this dispatcher

Run:
    python3 bin_to_image.py                     # converts every capture dir under raw_image_bin/
    python3 bin_to_image.py grain_ref_exp_2500  # converts just this one
"""

import argparse
import sys
from pathlib import Path

from bin_to_image_ref import convert_ref
from bin_to_image_trans import convert_trans
from stitching import CHANNELS, WIDTH

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw_image_bin"
OUT_DIR = HERE / "pseudo_image"


def convert(capture_dir, out_dir=OUT_DIR, w=WIDTH, c=CHANNELS):
    """Route capture_dir to the ref or trans pipeline by name; returns the PNG path or None."""
    name = Path(capture_dir).name.lower()
    if "ref" in name:
        print(f"Converting {Path(capture_dir).name} (reflectance) ...")
        return convert_ref(capture_dir, out_dir, w, c)
    if "tran" in name:
        print(f"Converting {Path(capture_dir).name} (transmittance) ...")
        return convert_trans(capture_dir, out_dir, w, c)
    print(f"Skipping {Path(capture_dir).name}: name contains neither 'ref' nor 'tran'")
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", nargs="?", default=None,
                        help="Name of a subfolder under raw_image_bin/ to convert "
                             "(default: convert every subfolder found there).")
    parser.add_argument("--width", type=int, default=WIDTH, help="Spatial pixels per line.")
    parser.add_argument("--channels", type=int, default=CHANNELS, help="Spectral bands per line.")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        sys.exit(f"{RAW_DIR} does not exist")

    if args.capture:
        targets = [RAW_DIR / args.capture]
    else:
        targets = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    if not targets:
        sys.exit(f"no capture directories found under {RAW_DIR}")

    for d in targets:
        if not d.is_dir():
            sys.exit(f"{d} is not a directory")
        convert(d, w=args.width, c=args.channels)


if __name__ == "__main__":
    main()
