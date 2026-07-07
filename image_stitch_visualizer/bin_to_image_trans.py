"""
Transmittance-mode bin -> pseudo image.

Currently identical to the plain stitch + mean-across-bands render (no
correction applied yet) -- transmittance-specific preprocessing (e.g. a
dedicated white-reference division, since white_ref_trans_exp_10000 exists
for transmittance unlike reflectance) is planned but not implemented here
yet. Kept as its own entry point so that preprocessing can be added later
without touching the reflectance path.

Run:
    python3 bin_to_image_trans.py <capture_dir> [out_dir]
"""

import sys
from pathlib import Path

from PIL import Image

from stitching import CHANNELS, WIDTH, pseudo_image, stitch

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "pseudo_image"


def convert_trans(capture_dir, out_dir=OUT_DIR, w=WIDTH, c=CHANNELS):
    """Stitch capture_dir and write <out_dir>/<capture_dir.name>.png. Returns the PNG path."""
    cube = stitch(capture_dir, w, c)
    img = pseudo_image(cube)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(capture_dir).name}.png"
    Image.fromarray(img, mode="L").save(out_path)
    print(f"  wrote {out_path} ({img.shape[1]}x{img.shape[0]})")
    return out_path


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 bin_to_image_trans.py <capture_dir> [out_dir]")
    capture_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR
    print(f"Converting {Path(capture_dir).name} (transmittance) ...")
    convert_trans(capture_dir, out_dir)


if __name__ == "__main__":
    main()
