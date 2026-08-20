# PCA of individual voxels

120 masked voxels sampled per view, both faces pooled, four matrices (two modes x 0 h / 8 h) of roughly 100k x 192. Each matrix is fitted twice: on the mask as drawn, and on the mask with two boundary rings eroded. Written by `python3 pca_pixels.py`.

## the number this was built for

`spectra.npy` averages ~700 voxels into one row. This asks what that average destroys: of a component's variance over all voxels, how much is BETWEEN kernels — already carried by the cheap mean spectrum — and how much is WITHIN one, visible only to a per-pixel model.

| d | mode | h | erode | total variance | PC1 between-kernel % | PC2 | PC3 |
|---|---|---|---|---|---|---|---|
| 0 | reflectance | 0 | 0 px | 13.07 | 10.6 | 6.9 | 10.9 |
| 0 | reflectance | 0 | 2 px | 5.53 | 21.7 | 16.4 | 17.7 |
| 0 | reflectance | 8 | 0 px | 7.24 | 33.2 | 29.3 | 9.5 |
| 0 | reflectance | 8 | 2 px | 3.78 | 53.3 | 24.6 | 22.0 |
| 0 | transmittance | 0 | 0 px | 6.64 | 48.7 | 17.9 | 4.6 |
| 0 | transmittance | 0 | 2 px | 5.69 | 64.5 | 37.3 | 29.7 |
| 0 | transmittance | 8 | 0 px | 4.80 | 51.5 | 13.1 | 6.9 |
| 0 | transmittance | 8 | 2 px | 4.10 | 63.5 | 27.5 | 63.7 |
| 1 | reflectance | 0 | 0 px | 0.04 | 5.2 | 12.9 | 15.8 |
| 1 | reflectance | 0 | 2 px | 0.02 | 13.6 | 23.1 | 25.9 |
| 1 | reflectance | 8 | 0 px | 0.02 | 57.7 | 7.6 | 23.1 |
| 1 | reflectance | 8 | 2 px | 0.01 | 62.4 | 20.2 | 25.6 |
| 1 | transmittance | 0 | 0 px | 0.04 | 51.5 | 14.8 | 6.3 |
| 1 | transmittance | 0 | 2 px | 0.03 | 69.9 | 28.3 | 15.6 |
| 1 | transmittance | 8 | 0 px | 0.03 | 56.5 | 10.6 | 9.0 |
| 1 | transmittance | 8 | 2 px | 0.03 | 67.3 | 22.6 | 47.8 |

### how to read it

* **On the mask as drawn, most voxel variance is within-kernel.** Reflectance at 0 h puts only 10.6% of PC1 between kernels; the other 89% is variation across the face of a single grain.
* **Most of that is the mask edge, not the grain.** Peeling two pixel rings off takes reflectance's total voxel variance from 13.07 to 5.53 at 0 h and 7.24 to 3.78 at 8 h — **48-58% of all voxel variance lived in the outermost two pixels of the mask** — and the between-kernel share of PC1 roughly doubles, 10.6% to 21.7% and 33.2% to 53.3%. That is the signature of partial-volume mixing: the boundary voxels are part grain, part well.
* `score_maps_*.png` show it directly. PC1 paints a ring around the mask rim and the two tips. PC3 is the one that looks anatomical — an end-to-end gradient, embryo against distal.
* **Transmittance barely cares.** Its total variance falls only 14-15% under the same erosion, because a transmission image has no hard rim shadow to mix into. It is the mode whose masks can be trusted at the edge.

## what follows from it

* `config.MASK_ERODE_PX` is currently off, on the grounds that 2 px cost a median 23.5% of mask area for a contaminant nobody had measured. The contaminant is now measured, and in reflectance it is about half of all voxel variance. That trade is worth revisiting — for reflectance specifically.
* The case for per-pixel modelling is weaker than the raw within-kernel share suggests. Once the boundary is removed, reflectance at 8 h and both transmittance matrices are majority between-kernel on PC1, meaning the mask-mean already carries most of what the leading components see.
* Non-finite voxels are dropped, never imputed. Erosion removes most of them too (transmittance 8 h: 15,555 dropped at 0 px, 6,492 at 2 px), which is a second, independent sign that the bad voxels are an edge phenomenon.

## figures

* `scree.png`, `loadings_d<k>_<mode>.png` — as drawn (no erosion).
* `score_maps_<mode>_<h>h.png` — PC1-PC3 painted back onto five kernels spanning day 1 to never, cropped to the mask, with the grain itself on the top row.
* `variance_split.png` — the between-kernel share, mask as drawn beside mask eroded.

`variance_split.csv` has every component and both erosions.
