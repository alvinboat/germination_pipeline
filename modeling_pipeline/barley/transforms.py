"""Spectral preprocessing, applied at load time rather than baked into the file.

What is stored on disk is per-pixel pseudo-absorbance, `A = -log10(x)`. SNV is
applied here, by default, on every path -- so a model sees exactly -log10 -> SNV
as intended. Keeping it here rather than in build_dataset.py means trying a
Savitzky-Golay derivative, or changing how the mask is applied, is a flag rather
than an hour-long rebuild. That matters while the masks are still provisional.

SNV (standard normal variate) subtracts a spectrum's own mean and divides by its
own standard deviation. It is the usual NIR scatter correction: it removes
path-length and illumination differences between kernels and between days, which
is what makes an 8 h view comparable with an 0 h one at all.
"""
import numpy as np
from scipy.signal import savgol_filter


def snv(x, axis=-1):
    """Standard normal variate along `axis`. Zero-variance spectra come back 0."""
    x = np.asarray(x, np.float32)
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    return np.divide(x - mu, np.where(sd > 1e-6, sd, 1.0),
                     out=np.zeros_like(x), where=sd > 1e-6)


def savgol(x, deriv=1, window=11, poly=2, axis=-1):
    """Savitzky-Golay smoothing derivative -- standard NIR baseline removal.

    deriv=0 smooths, 1 removes an additive baseline, 2 removes a linear one.
    """
    if window % 2 == 0:
        raise ValueError("savgol window must be odd")
    return savgol_filter(np.asarray(x, np.float32), window, poly,
                         deriv=deriv, axis=axis).astype(np.float32)


def band_select(x, lo=None, hi=None, axis=-1):
    """Keep a wavelength sub-range, by index into the stored 192 bands."""
    sl = [slice(None)] * np.ndim(x)
    sl[axis] = slice(lo, hi)
    return x[tuple(sl)]


# ------------------------------------------------------------------ recipes --
def spectrum_pipeline(deriv=0, window=11, poly=2, bands=None):
    """-> f(X) for (N, B) kernel-mean spectra. This is the PLS input.

    SNV first, then the derivative: the scatter correction belongs on the raw
    absorbance, and differentiating afterwards removes whatever baseline SNV
    left. Chosen per run and recorded, not guessed once and forgotten.
    """
    def f(X):
        X = np.asarray(X, np.float32)
        if bands:
            X = band_select(X, *bands)
        X = snv(X)
        if deriv:
            X = savgol(X, deriv=deriv, window=window, poly=poly)
        return X
    f.spec = {"snv": True, "deriv": deriv, "window": window, "poly": poly,
              "bands": bands}
    return f


def patch_pipeline(deriv=0, window=11, poly=2, bands=None, zero_outside=True):
    """-> f(patch, mask) for (H, W, B) patches. This is the CNN input.

    Per-pixel SNV across bands, then everything outside the kernel is set to
    zero. Zeroing after SNV, not before, so the kernel's own pixels are
    normalised against themselves and the plate never enters the statistics.

    Returns (B, H, W) float32, channels-first, ready for torch.
    """
    def f(patch, mask):
        p = np.asarray(patch, np.float32)
        if bands:
            p = band_select(p, *bands)
        p = snv(p, axis=-1)
        if deriv:
            p = savgol(p, deriv=deriv, window=window, poly=poly, axis=-1)
        p[~np.isfinite(p)] = 0.0
        if zero_outside:
            p[~mask] = 0.0
        return np.ascontiguousarray(p.transpose(2, 0, 1))
    f.spec = {"snv": True, "deriv": deriv, "window": window, "poly": poly,
              "bands": bands, "zero_outside": zero_outside}
    return f
