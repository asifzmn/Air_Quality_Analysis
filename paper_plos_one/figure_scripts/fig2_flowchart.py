"""Fig 2 (redrawn 2026-11-03): data collection and preparation flowchart.
Replaces the hand-drawn diagram ("Scrapping", "Satellite Data -> Meteorological Data", "2017-2021 PM2.5").
Usage: python fig2_flowchart.py  -> out/data_collection_preparation.(png|tif)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plos_style as S
S.setup()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, YELLOW, PINK, GREY = "#d3e3f1", "#fdf1cc", "#ead6e3", "#dcdcdc"
EDGE = "#5b6b7a"
rows = [  # source, process, data, colour
    ("Meteoblue", "Download and\nprocessing", "2019 hourly weather\n(55 zones)", BLUE),
    ("Weather Underground", "Download and\nprocessing", "2019–2021 hourly weather\n(Dhaka airport)", BLUE),
    ("ERA5 reanalysis and\nNASA FIRMS", "API / archive download\nand processing", "2018–2021 boundary-layer\nheight and fire detections", BLUE),
    ("Berkeley Earth", "Automated retrieval\nand processing", "2018–2021 hourly PM$_{2.5}$\n(55 + 389 zones)", YELLOW),
    ("Bangladesh Bureau\nof Statistics", "PDF parsing and\nprocessing", "District demographic\nstatistics", PINK),
    ("Google Mobility\nReports", "Download and\nprocessing", "2020–2022 daily\nmobility", PINK),
]
analyses = {"met": (5.0, "Meteorology, boundary-layer\nand fire analysis"), "st": (3.5, "Spatio-temporal\nanalysis"),
            "fc": (2.5, "Forecasting"), "dem": (1.5, "Demographic\nanalysis"), "mob": (0.5, "Mobility\nanalysis")}
edges = {0: ["met"], 1: ["met", "fc"], 2: ["met"], 3: ["met", "st", "fc", "dem", "mob"], 4: ["dem"], 5: ["mob"]}

fig, ax = plt.subplots(figsize=(S.FULL_W, 4.6))
ax.set_xlim(0, 10.6); ax.set_ylim(-0.1, 5.6); ax.axis("off")

def box(x, y, w, h, text, fc, style="round,pad=0.02,rounding_size=0.12", fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec=EDGE, lw=0.7))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=S.INK, weight="bold" if bold else "normal")

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.7, shrinkA=0, shrinkB=0))

for x, title in [(1.0, "Source"), (3.3, "Collection"), (5.85, "Prepared data"), (9.2, "Analysis")]:
    ax.text(x, 5.5, title, ha="center", va="center", fontsize=8.2, weight="bold", color=S.INK)
ys = [5.0 - i * 0.9 for i in range(6)]
for i, (src, proc, data, fc) in enumerate(rows):
    y = ys[i]
    box(1.0, y, 1.9, 0.66, src, fc, style="round,pad=0.02,rounding_size=0.3")
    box(3.3, y, 1.9, 0.62, proc, fc)
    box(5.85, y, 2.3, 0.66, data, fc, style="square,pad=0.02")
    arrow(1.95, y, 2.35, y); arrow(4.25, y, 4.7, y)
for k, (y, name) in analyses.items():
    box(9.2, y, 2.4, 0.62, name, GREY)
for i, targets in edges.items():
    for k in targets:
        arrow(7.0, ys[i], 8.0, analyses[k][0])
fig.tight_layout(pad=0.2)
S.save(fig, "data_collection_preparation")
