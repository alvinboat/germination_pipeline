# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Tooling and captured data for hyperspectral imaging (HSI) with a Specim push-broom
line-scan camera. There is no build system, test suite, or dependency manifest — it is
a small set of NumPy scripts plus raw capture directories.

- `loadstich/` — Python modules for unpacking/repacking Mono12Packed data and stitching per-line captures into a hypercube.
- `07012026/` — raw capture data (one directory per acquisition; large binaries, not code).

The only third-party dependency is `numpy`.

## Architecture

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

There is no CLI entry point. Use the class directly, e.g.:

```python
from specim_stitcher import SpecimStitcher   # local import (see gotcha above)
SpecimStitcher(load_path="07012026/grain_ref_exp_2500",
               save_path="out", dark=False).load_lines()
```

Remember `load_lines()` deletes the source directory's contents on success.
