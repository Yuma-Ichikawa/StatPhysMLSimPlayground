"""Journal plotting style shared by every predictive figure."""

from __future__ import annotations

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "1", "2", "3"]
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]
FIGSIZE = (6.4, 4.8)


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
            "axes.xmargin": 0.01,
            "axes.ymargin": 0.01,
            "mathtext.fontset": "stix",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        "figure.figsize": FIGSIZE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.bbox": None,
            "savefig.dpi": 300,
        }
    )
