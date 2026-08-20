"""
Complete-kernel segmentation via a two-endmember, hysteresis-grown Spectral
Angle Mapper. Adapted from the author's prior /home/alvin/pipeline/sam.py, which
solved the "we don't get the ENTIRE kernel" problem that a single-reference /
morphology-trimmed mask suffers from.

Why this recovers the WHOLE kernel where a single bright reference + morphology
does not:

  * A barley kernel is spectrally BIMODAL -- a starchy endosperm BODY plus a
    brighter, spectrally-distinct GERM/embryo at one end (redder; angle ~0.2-0.4
    to the body ref). One reference rounds the germ lobe off -> missing tip.
  * SAM is illumination-invariant (angle ignores magnitude), so brightness is
    used only to BOOTSTRAP references, never as the decision -- dim/matte kernel
    still classifies as kernel by spectral shape.
  * The body reference is RE-SEEDED from the whole tau-blob (endosperm, not just
    its specular shine), which shrinks the angle of matte tips so they're kept.
  * HYSTERESIS grow: a strict `seed_tau` core seeds the grow; membership is the
    looser `angle < tau`. Dim-but-contiguous kernel gets pulled in; isolated
    grid-wall / speckle blobs are dropped (no seed touches them). Morphological
    opening denoises the SEED ONLY -- the final mask is never eroded, so thin
    tips and the germ survive.
  * `tau` is kept below the grid-wall spectral angle so bright walls are rejected
    spectrally rather than by insetting the cell (which would clip edge kernel).

Entry point: kernel_mask(crop, cell_mask, **params) -> (mask, paper_ref,
body_ref, germ_ref).
"""
import cv2
import numpy as np


def spectral_angle(pixels, ref):
    """Spectral angle (rad) between each pixel spectrum and `ref`. Length-
    invariant (the defining SAM property). pixels: (..., B); ref: (B,)."""
    pixels = pixels.astype(np.float64)
    ref = ref.astype(np.float64)
    num = pixels @ ref
    den = np.linalg.norm(pixels, axis=-1) * np.linalg.norm(ref) + 1e-12
    return np.arccos(np.clip(num / den, -1.0, 1.0))


def endmembers(crop, cell_mask, dark_pct=50, bright_pct=80):
    """Bootstrap (paper_ref, body_ref) from brightness within a cell. body_ref =
    median of the brightest >= bright_pct% in-cell pixels (barley signature);
    paper_ref = median of the darkest dark_pct% (diagnostic + emptiness guard).
    Medians are robust to a few mixed/edge pixels."""
    b = crop.mean(-1)
    inb = b[cell_mask]
    paper = cell_mask & (b <= np.percentile(inb, dark_pct))
    kern = cell_mask & (b >= np.percentile(inb, bright_pct))
    paper_ref = np.median(crop[paper].astype(np.float64), axis=0)
    body_ref = np.median(crop[kern].astype(np.float64), axis=0)
    return paper_ref, body_ref


def _largest_blob(mask):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lbl == biggest

def _hysteresis(seed, grow):
    """Keep connected components of `grow` that contain >=1 `seed` pixel
    (Canny-style two-threshold region grow)."""
    n, lbl = cv2.connectedComponents(grow.astype(np.uint8), 8)
    keep = np.unique(lbl[seed])
    keep = keep[keep != 0]
    return np.isin(lbl, keep)


def _germ_ref(crop, body, a_body, germ_dilate, germ_bright_frac,
              germ_min_angle, min_germ_px):
    """Bootstrap the germ endmember, or (None, ...) if the kernel shows no
    exposed germ. Germ candidates are BRIGHT, body-ADJACENT (dilation ring),
    and spectrally DISTINCT from the body (angle >= germ_min_angle)."""
    bright = crop.mean(-1)
    core_bright = np.median(bright[body])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * germ_dilate + 1,) * 2)
    vic = cv2.dilate(body.astype(np.uint8), k).astype(bool)
    cand = (vic & ~body & (a_body >= germ_min_angle)
            & (bright >= germ_bright_frac * core_bright))
    if int(cand.sum()) < min_germ_px:
        return None, core_bright, vic
    return np.median(crop[cand].astype(np.float64), axis=0), core_bright, vic


def kernel_mask(crop, cell_mask, tau=0.25, seed_tau=0.15, dark_pct=50,
                bright_pct=80, open_px=1, keep_largest=True, min_frac=0.0,
                min_contrast=2.0, germ=True, germ_bright_frac=0.30,
                germ_min_angle=0.18, min_germ_px=10, germ_dilate=3,
                tau_germ=None):
    """Return (kernel_mask, paper_ref, body_ref, germ_ref) for one cell.

    See module docstring for the model. `crop` (H,W,B), `cell_mask` (H,W) bool.
    """
    paper_ref, body_ref = endmembers(crop, cell_mask, dark_pct, bright_pct)
    germ_ref = body_ref

    # emptiness guard: no kernel -> brightest core barely beats the dark floor
    if min_contrast and body_ref.mean() < min_contrast * (paper_ref.mean() + 1e-9):
        return np.zeros_like(cell_mask), paper_ref, body_ref, germ_ref

    # 1. body-representative ref: re-seed from the whole tau-blob (endosperm,
    #    not just its shine) so matte tips get small angles.
    body = (spectral_angle(crop, body_ref) < tau) & cell_mask
    if keep_largest:
        body = _largest_blob(body) & cell_mask
    if body.any():
        body_ref = np.median(crop[body].astype(np.float64), axis=0)
    a_body = spectral_angle(crop, body_ref)
    member = (a_body < tau) & cell_mask

    # 2. germ endmember -> a bright, body-connected spectral extension
    if germ and body.any():
        germ_cand, core_bright, vic = _germ_ref(
            crop, body, a_body, germ_dilate, germ_bright_frac,
            germ_min_angle, min_germ_px)
        if germ_cand is not None:
            germ_ref = germ_cand
            a_germ = spectral_angle(crop, germ_ref)
            tg = tau if tau_germ is None else tau_germ
            bright = crop.mean(-1)
            member = member | ((a_germ < tg)
                               & (bright >= germ_bright_frac * core_bright)
                               & vic & cell_mask)

    # 3. hysteresis grow with a denoised body SEED (final mask never eroded)
    seed = (a_body < seed_tau) & cell_mask
    if open_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1,) * 2)
        seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) & cell_mask
    km = _hysteresis(seed, member)

    if keep_largest:
        km = _largest_blob(km) & cell_mask
    if min_frac and km.sum() < min_frac * int(cell_mask.sum()):
        km = np.zeros_like(km)
    return km, paper_ref, body_ref, germ_ref



if __name__ == "__main__":
    pass