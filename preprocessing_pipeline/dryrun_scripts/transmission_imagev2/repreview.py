"""Re-render the preview PNGs from an already-corrected cube."""
import sys
import numpy as np
sys.path.insert(0, ".")
from process import band_mean, preview_png, saturation_png

name = sys.argv[1] if len(sys.argv) > 1 else "day3_dish0_trans"
cube = np.load(f"corrected_file/{name}.npy", mmap_mode="r")
z = np.load(f"corrected_file/{name}_masks.npz")

mean_plane, clipped_frac = band_mean(cube, z["saturated"], z["band_kept"])
preview_png(mean_plane, clipped_frac, f"corrected_image/{name}.png",
            f"mean of {int(z['band_kept'].sum())} bands")
saturation_png(clipped_frac, np.isfinite(mean_plane),
               f"corrected_image/{name}_saturation.png")
