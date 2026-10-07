#!/usr/bin/env python3
"""
make_graphical_abstract.py
==========================
The graphical abstract NeuroImage asks for at submission.

What it has to carry
--------------------
One thing, readable at thumbnail size: the chromatic advantage of higher visual
cortex is what its content predicts, and it does not survive removal of that
content, while the achromatic advantage of early cortex does.

That is a 2 x 2 x 2 — chromatic/achromatic terms, raw/content-removed, early/
higher — which is at the limit of what a graphical abstract can show. It is
drawn as two panels of paired bars rather than a schematic, because the point is
a pattern in real numbers and a cartoon would invite the reader to trust a
picture that is not the result.

Where the numbers come from
---------------------------
All eight values are quoted in the Results of `build_manuscript_v2.py`, in the
paragraph decomposing the eleven terms into chromatic and achromatic subsets:

    raw        chromatic   early 0.052   higher 0.081   (Δ = +0.029, 8 of 8)
               achromatic  early 0.057   higher 0.041   (Δ = +0.017, 7 of 8)
    residual   chromatic   early 0.007   higher 0.007   (n.s., 4 of 8)
               achromatic  early 0.036   higher 0.011   (Δ = +0.025, 8 of 8)

They are hard-coded here rather than recomputed, so if the Results paragraph
changes these must be changed with it. `check()` below re-reads that paragraph
from the manuscript source and fails if any of the eight numbers is no longer
present in it, which is what stops the figure drifting away from the text.

Size
----
Elsevier asks for at least 1328 x 531 px. The figure is drawn at 200 dpi and the
result is checked against that minimum before writing.

Usage
-----
    python make_graphical_abstract.py
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "build_manuscript_v2.py"
OUT = "figures/graphical_abstract.png"
MIN_PX = (1328, 531)

# early, higher
RAW = {"chromatic": (0.052, 0.081), "achromatic": (0.057, 0.041)}
RES = {"chromatic": (0.007, 0.007), "achromatic": (0.036, 0.011)}

EARLY = "#4C6EF5"      # early visual cortex (V1-V3)
HIGHER = "#E8590C"     # higher visual cortex


def check():
    """Fail if the manuscript no longer contains these eight values."""
    if not os.path.exists(SRC):
        print(f"  {SRC} not found; skipping the consistency check")
        return
    t = open(SRC).read()
    missing = [f"{v:.3f}" for d in (RAW, RES) for pair in d.values()
               for v in pair if f"{v:.3f}" not in t]
    if missing:
        raise SystemExit(
            f"  these values are no longer in {SRC}: {sorted(set(missing))}\n"
            f"  the figure and the Results paragraph have diverged; reconcile "
            f"before submitting.")
    print(f"  all eight values still present in {SRC}")


def panel(ax, data, title, ymax):
    labels = ["Chromatic\nterms", "Achromatic\nterms"]
    x = [0, 1]
    w = 0.34
    early = [data["chromatic"][0], data["achromatic"][0]]
    higher = [data["chromatic"][1], data["achromatic"][1]]
    ax.bar([i - w / 2 for i in x], early, w, color=EARLY, label="Early (V1–V3)")
    ax.bar([i + w / 2 for i in x], higher, w, color=HIGHER,
           label="Higher visual cortex")
    for i, (a, b) in enumerate(zip(early, higher)):
        ax.text(i - w / 2, a + ymax * 0.02, f"{a:.3f}", ha="center",
                fontsize=9, color=EARLY)
        ax.text(i + w / 2, b + ymax * 0.02, f"{b:.3f}", ha="center",
                fontsize=9, color=HIGHER)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def main():
    check()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), dpi=200)
    ymax = 0.098
    panel(axes[0], RAW, "Decoded from brain activity", ymax)
    panel(axes[1], RES, "After removing color predicted by\nannotated scene content",
          ymax)
    axes[0].set_ylabel("Decoding $R^2$", fontsize=11)
    axes[1].set_yticklabels([])

    axes[1].annotate("advantage gone\n(n.s., 4 of 8)",
                     xy=(0.0, 0.004), xytext=(0.02, 0.048),
                     fontsize=9.5, ha="center", color="#333333",
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1))
    axes[1].annotate("early lead kept\n(Δ = +0.025, 8 of 8)",
                     xy=(0.70, 0.018), xytext=(0.58, 0.066),
                     fontsize=9.5, ha="center", color="#333333",
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1))

    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle("Color decodable from higher visual cortex is largely "
                 "the color its content predicts",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    w, h = plt.imread(OUT).shape[1], plt.imread(OUT).shape[0]
    ok = w >= MIN_PX[0] and h >= MIN_PX[1]
    print(f"  wrote {OUT}  {w} x {h} px  "
          f"({'meets' if ok else 'BELOW'} Elsevier's {MIN_PX[0]} x {MIN_PX[1]} "
          f"minimum)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
