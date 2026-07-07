# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Tooling and captured data for hyperspectral imaging (HSI) with a Specim push-broom
line-scan camera. There is no build system, test suite, or dependency manifest — it is
a small set of NumPy/OpenCV/matplotlib scripts plus raw capture directories.

- `loadstich/` — packing codec + `SpecimStitcher` class for unpacking/repacking Mono12Packed data and stitching per-line captures into a hypercube.
- `pipeline/` — script-based stitch → dark-subtract → white-correct → geometry-correct chain that turns a raw capture dir into the corrected cube `grid/` consumes.
- `grid/` — detects the petri-dish + per-kernel cutout grid on a corrected cube and renders the labelled overlay.
- `kernels/` — manual kernel-outline workflow: read off coordinates, record outlines, rasterize + plot per-kernel spectra. Includes an interactive lasso/polygon alternative to hand-picked coordinates.
- `white_exploration/` — R&D scripts behind `pipeline/white_correction.py`'s tape-blob extraction; kept for reference, not part of the run path.
- `07012026/` — raw capture data (one directory per acquisition; large binaries, not code; gitignored).

Third-party dependencies: `numpy`, `Pillow`, `opencv-python`, `matplotlib`.

## Architecture

### Radiometric/geometric pipeline (`pipeline/`)
Each stage is a standalone script; run in order, each consuming the previous stage's
output cube (or a raw capture dir directly):
1. `stitch_grain.py` — stitches a directory of per-line `.bin` captures into a `(width, n_lines, channels)` uint16 cube. Non-destructive (unlike `SpecimStitcher`, it does not delete source files). Reuses `loadstich/hsi_save_load.load_hsi` via a `sys.path` shim.
2. `correction.py` — dark-current subtraction: `clip(raw - mean(dark_lines), 0)`. Produces `*_darksub_cube.npy`.
3. `white_correction.py` — flat-field correction using two in-scene teflon-tape blobs (no dedicated white capture exists for reflectance mode): `darksub / median(tape_blob) * SATURATION` per band. The blob-extraction approach is documented in `white_exploration/`. Produces `*_whitecorr_cube.npy`.
4. `generate_viable_reflectance.py` — corrects the push-broom scan-axis stretch using a checkerboard target (anisotropic rescale only). Produces `*_corrected_cube.npy` — the default input `grid/detect_grid.py` expects.

`full_correction.py` chains all three correction stages (2-4) in one run, calling into
each module's functions directly rather than reimplementing them — no intermediate
`_darksub_cube.npy`/`_whitecorr_cube.npy` is written to disk. Verified bit-for-bit
identical (`np.allclose`, max abs diff 0.0) against running the three stages
separately. Use the individual scripts instead when you need to inspect or tune one
stage (e.g. `--tape-pct`, `--manual-scale`) without repeating the others.

### Grid detection (`grid/`)
- `detect_grid.py` — locates the dish + rim on a corrected cube, regenerates a full pitch/phase cell lattice (so cells clipped by the rim or hidden under a marker still get placed), flags each cell usable/clipped/marker-occupied, and writes `<name>_grid_overlay.png` (dish circle + labelled cells) and `<name>_grid_cells.json` (per-cell corners/center/flags).
- `stress_test_grid.py` — stress-tests `build_grid`'s robustness against `detect_grid.py`'s own detection functions.

### Manual kernel workflow (`kernels/`)
- `grid_overlay.py` — renders a cube with a labelled pixel-coordinate grid so kernel corners can be read off by eye (unrelated to `grid/detect_grid.py` despite the similar name/output).
- `kernels.py` — `KERNELS` list of hand-picked `(x, y)` outlines, built from `grid_overlay.py`'s coordinates.
- `analyze_kernels.py` — rasterizes outlines from a `.py`/`.json` file into masks, then plots mean ± std spectra per kernel. Used for transmittance QC (confirming light actually passes through a kernel).
- `segment_kernels.py` — shared `display_band` helper (used by the two scripts above) plus `KernelSegmenter`, an interactive lasso/polygon alternative that traces kernels by mouse instead of hand-coding coordinates.

### `loadstich/` — packing codec + class-based stitcher
The pipeline turns a directory of individual scan **lines** into a single stitched
hyperspectral cube.

### `loadstich/hsi_save_load.py` — the packing codec
Camera data is **Mono12Packed**: two 12-bit pixels packed into three 8-bit bytes.
- `load_hsi(uint8_arr) -> uint16` unpacks 3 bytes → 2 pixels. Input `channels` (last dim) must be a multiple of 3.
- `save_hsi(uint16_arr) -> uint8` repacks 2 pixels → 3 bytes. Input `channels` must be a multiple of 2.
- These are exact inverses and are the single source of truth for the bit layout — the byte-ordering math is subtle (note the corrected `1::2` line in `load_hsi`); do not reimplement the shifts inline elsewhere.

### `loadstich/specim_stitcher.py` — `SpecimStitcher`
Reads a capture directory of per-line files, orients each line, stacks them, then
writes one packed cube. `load_lines()` is the entry point and its side effects matter:
1. Lists the directory, sorts filenames lexically (see the open TODO about sort order / neighbour-pair swapping), and requires the files to be *exclusively* `.bin` or `.npy`.
2. For each file: `load_hsi` → `reshape([c, w])` → `swapaxes(0,1)` → reverse rows (`[::-1]`). A line that fails to parse is replaced by a zero frame and counted in `self.bad_lines`.
3. `stitch_lines()` stacks lines along axis 1 into `self.img` (shape ≈ width × n_lines × channels).
4. `save_hsi(self.img)` repacks and saves to `save_path/raw_hsi_img_mono12p.npy` (or `raw_hsi_dark_mono12p.npy` when `dark=True`).
5. **Deletes every source file** in the load directory. This is destructive and irreversible — treat capture dirs under `07012026/` as consumable inputs, and never point `load_path` at data you need to keep without backing it up first.

Default frame geometry is `w=640`, `c=224` (640 spatial px × 224 spectral channels per line).

### Import path gotcha
`specim_stitcher.py` imports the codec as `from jarvis_gui.utils.hsi_save_load import load_hsi, save_hsi`.
These modules are meant to live at `jarvis_gui/utils/` inside a larger `jarvis_gui`
project — that package is **not present in this checkout**, so the import will not
resolve as-is. When running here, either add `jarvis_gui` to `PYTHONPATH` or adjust the
import to the local `hsi_save_load`.

## Capture data (`07012026/`)

Each subdirectory is one acquisition; the name encodes `sample_mode_exp`:
- sample: `dark` / `white` reference or `grain` (the actual specimen)
- mode: `ref` (reflectance) / `trans` (transmittance)
- `exp<n>`: exposure time (e.g. `2500`, `10000`, `100k`)

Files are named `<index>_<timestamp>_w640_h224_pMono12Packed.bin`. Each `.bin` is a
single scan line of 215040 bytes = 224 × 640 pixels × 12 bits / 8. `dark_*` captures are
the ones loaded with `dark=True`.

## Running

`loadstich/specim_stitcher.py` has no CLI entry point; use the class directly, e.g.:

```python
from specim_stitcher import SpecimStitcher   # local import (see gotcha above)
SpecimStitcher(load_path="07012026/grain_ref_exp_2500",
               save_path="out", dark=False).load_lines()
```

Remember `load_lines()` deletes the source directory's contents on success — prefer
`pipeline/stitch_grain.py` (non-destructive) unless you specifically want the packed
`SpecimStitcher` output.

Everything under `pipeline/`, `grid/`, and `kernels/` is a standalone script with
`argparse --help`. Typical end-to-end run from the repo root:

```bash
python3 pipeline/stitch_grain.py            # 07012026/*/  -> 07012026/stitched/*_cube.npy
python3 pipeline/correction.py              # -> *_darksub_cube.npy
python3 pipeline/white_correction.py        # -> *_whitecorr_cube.npy
python3 pipeline/generate_viable_reflectance.py  # -> *_corrected_cube.npy
python3 grid/detect_grid.py                 # -> *_grid_overlay.png, *_grid_cells.json

python3 kernels/grid_overlay.py <cube>      # read off (x, y) kernel coordinates
# hand-edit kernels/kernels.py with those coordinates, then:
python3 kernels/analyze_kernels.py <cube> kernels/kernels.py --save-mask
```
