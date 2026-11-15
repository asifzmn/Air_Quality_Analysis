"""Shared figure style for the PLOS ONE submission.
PLOS ONE: width 2.63–7.5 in, height ≤ 8.75 in, 300–600 dpi, TIFF (LZW) for submission; text 8–12 pt at print size."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# palette (agreed with author)
STEEL = "#3f78ad"      # main series
DEEP = "#246cb4"       # emphasis
LIGHT = "#9cc0d8"      # secondary series
GREYBLUE = "#b9c8d6"   # background / "other"
GREYLINE = "#7f97ad"
EDGE = "#2c4a66"       # outlines
INK = "#222222"
ROSE = "#b5506d"       # retired: user wants no red anywhere
SLATE = "#4f5d6b"      # dark slate, positive end of diverging heatmaps


FULL_W = 7.5           # in, full page width
COL_W = 5.2            # in, text-column width (PLOS template textwidth 5.25 in)

def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": "#e9e9e9", "grid.linewidth": 0.6, "savefig.dpi": 600,
        "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def ygrid(ax):
    ax.grid(axis="y", color="#ececec", lw=0.6); ax.set_axisbelow(True)

def save(fig, name):
    """Save PNG (Overleaf, 300 dpi) and TIFF (PLOS submission, 600 dpi, LZW)."""
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT, name + ".tif"), dpi=600, bbox_inches="tight", pad_inches=0.03,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", name)


# Pale US EPA AQI category bands (same colors as box_plot_avg); horizontal, for plots whose y-axis is PM2.5 in ug/m3
AQI_BANDS = [(0, 12, "Good", "#cfe8cf"), (12, 35.4, "Moderate", "#f6f0c4"), (35.4, 55.4, "Sensitive", "#f8dfc2"),
             (55.4, 150.4, "Unhealthy", "#f3cccc"), (150.4, 250.4, "Very unhealthy", "#e3d1e8"), (250.4, 500, "Hazardous", "#e8d3d6")]

def aqi_hbands(ax, ymax, labels=True):
    """Draw AQI bands behind the data up to ymax and label each visible band at the right edge."""
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(ax.transAxes, ax.transData)
    for lo, hi, name, col in AQI_BANDS:
        if lo >= ymax:
            break
        ax.axhspan(lo, min(hi, ymax), color=col, alpha=0.45, lw=0, zorder=0)
        if lo > 0:
            ax.axhline(lo, color="white", lw=1.0, zorder=0.5)
        if labels and (min(hi, ymax) - lo) > 0.035 * ymax:
            ax.text(1.008, (lo + min(hi, ymax)) / 2, name, transform=tr, ha="left", va="center", fontsize=6.5, color="#555",
                    clip_on=False)
    ax.set_ylim(0, ymax)
