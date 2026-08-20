"""Chart palette and matplotlib style.

The categorical hues, their dark steps, the sequential blue ramp and the chrome
inks are a documented, validated palette -- the slot ORDER is the colourblind-
safety mechanism, not decoration, so hues are assigned in fixed order and never
cycled. Adjacent-pair separation was computed, not eyeballed: worst adjacent CVD
deltaE 9.1 light / 8.4 dark (OKLab x100, >=8 target), worst adjacent normal-vision
19.6 / 19.3 (>=15 floor).

Two rules that follow from that validation and are enforced by how figures.py
uses this module:

* The five categorical hues are used ONLY where position cannot separate the
  series -- i.e. overlapping line charts. On a bar or dot chart whose axis
  already names the category, colour would re-encode information the chart
  already shows, and comparing any two of five colours invokes the all-pairs
  gate, which only three of these slots clear.
* On the light surface three of the five hues sit below 3:1 contrast. That is a
  documented relief case conditional on visible direct labels or a table view --
  hence every line chart is direct-labelled and every figure has its numbers
  restated in summary.md.
"""
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in fixed order. Never cycle past the list.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
               "#008300", "#9085e9", "#e66767"]

# Sequential: one hue, light -> dark. Never a rainbow.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

CHROME = {
    "light": {"surface": "#fcfcfb", "plane": "#f9f9f7", "primary": "#0b0b0b",
              "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "plane": "#0d0d0d", "primary": "#ffffff",
             "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835"},
}

_MODE = "light"


def mode():
    return _MODE


def series(i=None):
    pal = SERIES_LIGHT if _MODE == "light" else SERIES_DARK
    if i is None:
        return pal
    if i >= len(pal):
        raise ValueError(
            f"no categorical slot {i}: the palette stops at {len(pal)}. A 9th "
            "series must fold into 'Other' or become small multiples, never a "
            "generated hue.")
    return pal[i]


def ink(role):
    return CHROME[_MODE][role]


def cmap_sequential():
    """Blue ramp as a matplotlib colormap, for continuous magnitude only."""
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def on_dark(hexcolor):
    """Pick readable ink for text drawn on top of a filled mark."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.5 else CHROME["light"]["primary"]


def use(mode_name="light"):
    """Install the style. Call once before building figures."""
    global _MODE
    if mode_name not in CHROME:
        raise ValueError(mode_name)
    _MODE = mode_name
    c = CHROME[mode_name]
    mpl.rcParams.update({
        "figure.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "axes.edgecolor": c["axis"],
        "axes.labelcolor": c["secondary"],
        "axes.titlecolor": c["primary"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        # Hairline, solid, one shade off the surface. Never dashed -- dashing
        # reads as "threshold" when it is only a grid.
        "grid.color": c["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelcolor": c["secondary"],
        "ytick.labelcolor": c["secondary"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": c["primary"],
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.titlepad": 10,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,          # 2px lines
        "lines.markersize": 6,           # >=8px marker area
        "lines.solid_capstyle": "round",
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def title(ax, text, subtitle=None):
    """Title in primary ink, optional subtitle in secondary. The title names the
    single series, so a one-series chart needs no legend box.

    The title pad has to make room for the subtitle explicitly -- matplotlib
    knows nothing about the annotation, so with the default pad the two render
    on top of each other.
    """
    ax.set_title(text, loc="left", color=ink("primary"), pad=24 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=8.5, color=ink("secondary"), va="bottom")


def reference_line(ax, y, label, color=None):
    """A recessive horizontal reference (chance level, pooled mean)."""
    color = color or ink("muted")
    ax.axhline(y, color=color, lw=1.0, zorder=1)
    ax.annotate(f"{label} {y:.3f}", xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=ink("secondary"), va="center", ha="left")


def hide_grid_x(ax):
    ax.grid(axis="x", visible=False)


def hide_grid_y(ax):
    ax.grid(axis="y", visible=False)


def nice_ylim(ax, values, floor=0.0, pad=0.08):
    lo, hi = float(np.min(values)), float(np.max(values))
    span = max(hi - lo, 0.05)
    ax.set_ylim(max(floor, lo - pad * span), min(1.0, hi + pad * span))
