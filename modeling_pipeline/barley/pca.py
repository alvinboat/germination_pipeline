"""Unsupervised projection of the spectra: the shared machinery.

Two scripts use this -- `pca_spectra.py` on the mask-mean spectra and
`pca_pixels.py` on individual voxels -- and they must agree on every step, or
the two answers cannot be compared.

WHAT PCA NEEDS THAT THE MODELLING PATH DOES NOT
`transforms.spectrum_pipeline` is reused verbatim for SNV and the optional
derivative, so a PCA sees exactly what PLS sees. One thing is added here:
column mean-centring. SNV centres each *spectrum* against itself; PCA needs each
*band* centred across rows. They are different operations and doing the first
does not do the second -- an uncentred PCA spends PC1 on the grand mean and
every later component is wrong.

Bands are deliberately NOT scaled to unit variance. All 192 columns are the same
physical quantity in the same units, so autoscaling would only amplify the
noisiest low-signal bands into components that look real.

WHY THE COVARIANCE ROUTE
With 192 columns the 192x192 covariance eigendecomposition is exact, cheap and
identical for 438 rows or 400,000 -- so the per-pixel run and the mean-spectrum
run go through the same code rather than one of them needing a randomised SVD.
"""
import numpy as np

from . import transforms as T


def prepare(X, deriv=0, window=11, poly=2, bands=None):
    """-> (centred matrix, band means, the transform's own spec dict)."""
    f = T.spectrum_pipeline(deriv=deriv, window=window, poly=poly, bands=bands)
    Z = np.asarray(f(np.asarray(X, np.float32)), np.float32)
    mu = Z.mean(axis=0)
    return Z - mu, mu, f.spec


def fit(Z, n_components=None):
    """Eigendecomposition of the covariance. -> dict.

    loadings (n_pc, B), scores (N, n_pc), eigenvalues, explained-variance ratio.

    Component signs are pinned so that each loading's largest-magnitude element
    is positive. Sign is arbitrary in PCA, but leaving it arbitrary means two
    runs of the same data can produce mirror-image score plots, and a loading
    read as an absorption peak in one run reads as a trough in the next.
    """
    Z = np.asarray(Z, np.float64)
    n, b = Z.shape
    C = (Z.T @ Z) / max(n - 1, 1)
    w, V = np.linalg.eigh(C)                 # ascending
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    w = np.clip(w, 0.0, None)
    k = b if n_components is None else min(n_components, b)
    w, V = w[:k], V[:, :k]
    flip = np.sign(V[np.abs(V).argmax(axis=0), np.arange(k)])
    flip[flip == 0] = 1.0
    V = V * flip
    total = float(np.trace(C))
    return {"loadings": V.T.astype(np.float32),
            "scores": (Z @ V).astype(np.float32),
            "eigenvalues": w,
            "evr": w / total if total > 0 else np.zeros_like(w),
            "cum_evr": np.cumsum(w) / total if total > 0 else np.zeros_like(w),
            "n": n, "total_variance": total}


def project(Z, res):
    return np.asarray(Z, np.float64) @ res["loadings"].T.astype(np.float64)


def diagnostics(Z, res, n_pc):
    """-> (Hotelling T^2, Q residual) over the first `n_pc` components.

    The pair is the standard chemometric outlier map and it separates two
    different failures. A high T^2 is an extreme kernel that the model still
    describes -- unusually dark, unusually wet. A high Q is a kernel the model
    does NOT describe: its spectrum has structure no component accounts for,
    which is what a mask that caught the plastic clip, or a railed patch, looks
    like. Only the second casts doubt on the row itself.
    """
    n_pc = min(n_pc, res["loadings"].shape[0])
    Tsc = res["scores"][:, :n_pc].astype(np.float64)
    lam = np.maximum(res["eigenvalues"][:n_pc], 1e-12)
    t2 = (Tsc ** 2 / lam).sum(axis=1)
    P = res["loadings"][:n_pc].astype(np.float64)
    resid = np.asarray(Z, np.float64) - Tsc @ P
    return t2, (resid ** 2).sum(axis=1)


def center_within(scores, groups):
    """Subtract each group's own mean from its rows. -> same shape.

    The way to ask a question the confound would otherwise answer. Variety is
    perfectly confounded with dish and correlated with germination day, so a raw
    eta^2 for day cannot tell "day matters" from "the slow variety is a
    different variety". Centring within variety first removes every between-
    variety difference, and whatever eta^2 survives is within-variety.
    """
    scores = np.asarray(scores, np.float64).copy()
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = groups == g
        scores[m] -= scores[m].mean(axis=0)
    return scores


def eta_squared(scores, groups):
    """Share of a component's variance that sits BETWEEN groups. -> (n_pc,).

    The number that decides whether a score plot means anything here. Variety is
    perfectly confounded with dish, so a component that separates varieties
    beautifully and separates dishes just as well is reading the plate. Compare
    the two columns, never read either alone.
    """
    scores = np.asarray(scores, np.float64)
    groups = np.asarray(groups)
    out = np.zeros(scores.shape[1])
    for a in range(scores.shape[1]):
        x = scores[:, a]
        grand = x.mean()
        ss_tot = ((x - grand) ** 2).sum()
        if ss_tot <= 0:
            continue
        ss_between = sum(len(x[groups == g]) * (x[groups == g].mean() - grand) ** 2
                         for g in np.unique(groups))
        out[a] = ss_between / ss_tot
    return out
