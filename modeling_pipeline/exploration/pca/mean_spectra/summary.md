# PCA of the mask-mean spectra

438 kernels x 192 bands (929-1614 nm), side-averaged so every matrix is exactly one row per kernel. Six matrices: two modes x {0 h, 8 h, 8 h - 0 h}. Two preprocessing variants: SNV, and SNV then a first derivative. Written by `python3 pca_spectra.py`.

Two kernels are one-sided means, the annotator having missed the other face: dish13_R3C2, dish3_R1C2.

## the controls, which decide whether any score plot means anything

Each cell is the largest share, over PC1-PC3, of a component's variance that sits between the groups named. `dish|var` and `day|var` are computed after centring the scores within variety, so they are the parts that variety cannot explain.

| d | mode | view | PC1 % | PC1-3 % | dish | variety | germ | day | dish\|var | day\|var |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | reflectance | 0h | 81.4 | 95.3 | 0.575 | 0.347 | 0.044 | 0.065 | **0.369** | **0.022** |
| 0 | reflectance | 8h | 86.8 | 97.8 | 0.388 | 0.258 | 0.041 | 0.234 | **0.189** | **0.155** |
| 0 | reflectance | delta | 80.7 | 93.9 | 0.506 | 0.170 | 0.009 | 0.075 | **0.405** | **0.076** |
| 0 | transmittance | 0h | 85.0 | 98.1 | 0.173 | 0.115 | 0.068 | 0.034 | **0.125** | **0.036** |
| 0 | transmittance | 8h | 84.4 | 97.0 | 0.230 | 0.162 | 0.081 | 0.108 | **0.200** | **0.079** |
| 0 | transmittance | delta | 84.1 | 95.6 | 0.376 | 0.138 | 0.038 | 0.109 | **0.306** | **0.087** |
| 1 | reflectance | 0h | 34.4 | 80.4 | 0.426 | 0.296 | 0.050 | 0.045 | **0.187** | **0.022** |
| 1 | reflectance | 8h | 76.6 | 92.2 | 0.529 | 0.398 | 0.047 | 0.206 | **0.279** | **0.152** |
| 1 | reflectance | delta | 55.1 | 82.6 | 0.490 | 0.154 | 0.008 | 0.083 | **0.409** | **0.082** |
| 1 | transmittance | 0h | 83.9 | 96.4 | 0.165 | 0.149 | 0.058 | 0.034 | **0.107** | **0.032** |
| 1 | transmittance | 8h | 84.1 | 95.9 | 0.258 | 0.193 | 0.050 | 0.128 | **0.156** | **0.088** |
| 1 | transmittance | delta | 76.6 | 88.8 | 0.343 | 0.153 | 0.042 | 0.112 | **0.228** | **0.090** |

### how to read it

* **PC1 takes 76-87% of the variance** in every matrix but one. After SNV, that is one dominant axis and a long thin tail — three components carry 93-98%.
* **`dish` beats `variety` everywhere.** Variety is perfectly confounded with dish, so this is the decisive comparison: the leading components track the plate more closely than the cultivar. `dish|var` is the pure plate effect, and it reaches 0.41 on the reflectance delta — four tenths of a leading component's variance is which dish the kernel sat in, within a single variety.
* **Nothing separates germinated from never.** `germ` never exceeds 0.081. There is no unsupervised ever/never axis in these spectra, which is what the germination histograms would predict: 87-91% germinate, so the minority class is small and, on this evidence, not spectrally distinct.
* **The one real germination signal is reflectance at 8 h.** `day` 0.234 falls to `day|var` 0.155 once variety is removed — so about a third of it was variety — but 0.155 survives, and nothing else comes close. Eight hours of imbibition, seen in reflectance, is where to look.
* **Transmittance is the cleaner mode.** Its plate effect is roughly half reflectance's (`dish|var` 0.107-0.306 against 0.187-0.409). It also carries less germination signal, so the two modes are not redundant.
* **The delta is a disappointment.** 8 h - 0 h was the obvious place to look for imbibition, and it is the most dish-dominated matrix of the six. Differencing two captures cancels the kernel and keeps whatever drifted between sessions.
* **The derivative changes little.** It shifts variance between components — reflectance 0 h PC1 drops from 81% to 34% — without changing any conclusion above. Both are kept; read the deriv-0 figures unless you want the loadings resolved.

## figures

* `scree.png` — cumulative variance, all six matrices, both derivatives.
* `scores_d<k>_<mode>_<view>.png` — PC1/PC2 coloured by variety, by **dish** (the control, in the same coordinates), by germinated/never, and PC1/PC3 by germination day.
* `loadings_d<k>_<mode>.png` — PC1-PC4 against wavelength.
* `outliers_d<k>.png` — Hotelling T2 against Q residual. A high Q is a spectrum the components do not describe, which is what a mask that caught the retaining clip looks like; the worst kernel in each panel is named.

`controls.csv` has every component, not just the best of PC1-3.
