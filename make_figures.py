"""
make_figures.py
===============
Regenerate the manuscript figures from the canonical result files.

Why this exists: the figures had no generator. They were produced ad hoc, and
consequently drifted from the text — Figure 3 was still plotting the superseded
full-image fg/bg files (`roi_fgcolor_vw`) after the Results switched to the clean
subset (`roi_fgcolor_cleanvw`), and Figure 4's significance marks predated the
current statistics. Generating them from the same `.npy` files the text cites
makes that class of drift impossible.

Every figure reads from the `_v2` summaries (post shard-fix) and uses the
display names used in the paper ("higher visual", not the `concept` code key).

Run from the repository root:
    python make_figures.py                 # all figures -> figures/
    python make_figures.py --outdir /tmp   # elsewhere
"""

import argparse
import json
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Base font sizes. Raised from matplotlib defaults so figure text stays legible
# at journal column width (editor comment, 31 Aug 2026).
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
})

ROIS = ["early_v1v3", "v4_color", "concept"]
NICE = {"early_v1v3": "early\nV1–V3", "v4_color": "V4", "concept": "higher\nvisual"}
LEGEND = {"early_v1v3": "Early (V1–V3)", "v4_color": "V4", "concept": "Higher visual"}
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple",
           "pink", "brown", "black", "white", "gray"]
C_EARLY, C_V4, C_HIGH = "#4C72B0", "#937860", "#C44E52"


# ----------------------------------------------------------------- statistics
def paired(diff, n_boot=20000, seed=0):
    diff = np.asarray(diff, float)
    n = len(diff)
    obs = diff.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    p = float((np.abs((signs * diff).mean(1)) >= abs(obs) - 1e-12).mean())
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, p, float(lo), float(hi)


def bh(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ps[order[i]] * m / (i + 1))
        q[order[i]] = prev
    return q


def stars(q):
    return "**" if round(q, 3) < 0.01 else ("*" if round(q, 3) < 0.05 else "n.s.")


# ----------------------------------------------------------------- data
def load(stem):
    """Return dict of region -> (per-colour R² [subj x bins], overall R² [subj])."""
    d = np.load(f"{stem}_v2_8subj_summary.npy", allow_pickle=True).item()
    return {r: (np.asarray(d["agg"][r]["per"], float),
                np.asarray(d["agg"][r]["ov"], float)) for r in ROIS}


def sem(x):
    return x.std(0, ddof=1) / np.sqrt(x.shape[0])


# ----------------------------------------------------------------- figure 1
def fig_dissociation(out, interaction_panel=False):
    """Figure 1. With interaction_panel=True an extra panel B is drawn showing the
    tested colour-minus-luminance interaction (reviewer comment [26]); the default
    is unchanged so the v1 figure is not affected."""
    col = load("roi_color_vw")
    if interaction_panel:
        # Panel A spans the top row, B and C share the bottom. With all panels
        # in one row the figure was 3.3:1, so at the 6-inch page width it
        # rendered only ~1.8 inches tall and nothing in it was readable.
        fig = plt.figure(figsize=(13, 9.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[5.2, 4.4], hspace=0.32,
                              wspace=0.22)
        ax = fig.add_subplot(gs[0, :])
        axb = fig.add_subplot(gs[1, 0])
        axc = fig.add_subplot(gs[1, 1])
    else:
        fig, ax = plt.subplots(figsize=(13, 5.2))
        axb = axc = None
    x = np.arange(len(COLOURS))
    w = 0.27
    for k, (r, c) in enumerate(zip(ROIS, [C_EARLY, C_V4, C_HIGH])):
        per, ov = col[r]
        ax.bar(x + (k - 1) * w, per.mean(0), w, yerr=sem(per), color=c, capsize=2,
               label=f"{LEGEND[r]}   (overall R²={ov.mean():+.3f})")
    # higher-minus-early per colour, BH across the 11 colours
    diffs = col["concept"][0] - col["early_v1v3"][0]
    ps = [paired(diffs[:, j])[1] for j in range(len(COLOURS))]
    qs = bh(ps)
    top = max((col[r][0].mean(0) + sem(col[r][0])).max() for r in ROIS)
    for j, q in enumerate(qs):
        if round(q, 3) < 0.05:
            ax.text(x[j], top * 1.06, "*", ha="center", fontsize=15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(top=top * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(COLOURS, rotation=45, ha="right")
    # The BARS are per-colour R² (sklearn "raw_values"); only the OVERALL R² in
    # the legend is variance-weighted. Labelling the axis "variance-weighted"
    # conflated the two.
    ax.set_ylabel("test R² per color term  (mean ± SEM, n = 8)")
    # Legend ABOVE the axes: at loc="upper left" it covered the significance
    # asterisks over the first two colours, so a reader could not tell whether
    # red and orange were significant.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=3, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    if axb is not None:
        # Panel B: the double dissociation as a TESTED INTERACTION, not two
        # main effects. Interaction values are read from stats_table_v2.json so
        # the figure cannot drift from Table S3.
        lum = load("roi_luminance_vw")
        stats = {r["cells"][0]: r["cells"] for r in
                 json.load(open("stats_table_v2.json"))}
        xb = np.arange(len(ROIS)); wb = 0.36
        cvals = np.array([col[r][1].mean() for r in ROIS])
        lvals = np.array([lum[r][1].mean() for r in ROIS])
        csem  = np.array([col[r][1].std(ddof=1) / np.sqrt(len(col[r][1])) for r in ROIS])
        lsem  = np.array([lum[r][1].std(ddof=1) / np.sqrt(len(lum[r][1])) for r in ROIS])
        axb.bar(xb - wb/2, cvals, wb, yerr=csem, capsize=2, color="#5B8C5A",
                label="color")
        axb.bar(xb + wb/2, lvals, wb, yerr=lsem, capsize=2, color="#B0B0B0",
                label="luminance")
        top_b = (cvals + csem).max()
        axb.set_ylim(0, top_b * 1.18)
        axb.set_xticks(xb); axb.set_xticklabels([LEGEND[r] for r in ROIS], fontsize=11)
        axb.set_ylabel("overall R\u00b2  (mean \u00b1 SEM, n = 8)")
        axb.axhline(0, color="k", lw=0.8)
        axb.legend(frameon=False, fontsize=11, loc="upper left")
        axb.spines[["top", "right"]].set_visible(False)

        # Panel C: the tested quantity itself. B shows two main effects and
        # leaves the reader to subtract; C plots the colour-minus-luminance
        # difference per region with its 95% bootstrap interval, and brackets
        # the two contrasts that were actually tested (comment [26]).
        deltas = [paired(col[r][1] - lum[r][1]) for r in ROIS]
        dmean = np.array([d[0] for d in deltas])
        dlo   = np.array([d[2] for d in deltas])
        dhi   = np.array([d[3] for d in deltas])
        axc.errorbar(xb, dmean, yerr=[dmean - dlo, dhi - dmean], fmt="o",
                     ms=7, color="#4A4A4A", capsize=4, lw=1.6, zorder=3)
        axc.axhline(0, color="k", lw=0.8)
        axc.set_xticks(xb)
        axc.set_xticklabels([LEGEND[r] for r in ROIS], fontsize=11)
        axc.set_xlim(-0.5, len(ROIS) - 0.5)
        axc.set_ylabel("color \u2212 luminance R\u00b2\n(mean, 95% bootstrap CI)")
        axc.spines[["top", "right"]].set_visible(False)

        # brackets, drawn from the same stats_table_v2.json rows as Table S3
        span = dhi.max() - min(0.0, dlo.min())
        y0 = dhi.max() + span * 0.10
        for k, (key, i, j) in enumerate([
                ("Interaction (colour\u2212luminance): V4 \u2212 early", 0, 1),
                ("Interaction (colour\u2212luminance): higher \u2212 early", 0, 2)]):
            c = stats.get(key)
            if not c:
                continue
            y = y0 + k * span * 0.17
            tick = span * 0.035
            axc.plot([i, i, j, j], [y - tick, y, y, y - tick], color="k", lw=1.0,
                     clip_on=False)
            axc.text((i + j) / 2, y + span * 0.02,
                     f"{c[5]}  {c[1]}  95% CI {c[6]}  ({c[4]})",
                     ha="center", va="bottom", fontsize=10)
        axc.set_ylim(min(0.0, dlo.min()) - span * 0.06, y0 + span * 0.42)

        for lbl, a_ in (("A", ax), ("B", axb), ("C", axc)):
            a_.text(-0.08, 1.06, lbl, transform=a_.transAxes,
                    fontsize=15, fontweight="bold", va="top")

    if axb is None:
        fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    n_sig = int((np.round(qs, 3) < 0.05).sum())
    print(f"  fig1: overall " +
          " / ".join(f"{col[r][1].mean():.3f}" for r in ROIS) +
          f" | {n_sig}/11 colours significant")


# ----------------------------------------------------------------- figure 2
def fig_collapse(out):
    raw, res = load("roi_color_vw"), load("roi_colorresid_vw")
    praw, pres = load("roi_pcolor_vw"), load("roi_pcolorresid_vw")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (a, b), title in zip(axes, [(raw, res), (praw, pres)],
                                 ["physical color", "perceptual color (van de Weijer)"]):
        x = np.arange(3)
        w = 0.36
        rv = np.array([a[r][1].mean() for r in ROIS])
        sv = np.array([b[r][1].mean() for r in ROIS])
        re = np.array([a[r][1].std(ddof=1) / np.sqrt(8) for r in ROIS])
        se = np.array([b[r][1].std(ddof=1) / np.sqrt(8) for r in ROIS])
        ax.bar(x - w / 2, rv, w, yerr=re, capsize=3, color="#7C93C3", label="raw")
        ax.bar(x + w / 2, sv, w, yerr=se, capsize=3, color="#C4756E",
               label="residual\n(object identity removed)")
        ps = [paired(a[r][1] - b[r][1])[1] for r in ROIS]
        qs = bh(ps)
        for i, q in enumerate(qs):
            y = max(rv[i] + re[i], sv[i] + se[i]) * 1.06
            ax.plot([x[i] - w / 2, x[i] + w / 2], [y, y], color="k", lw=0.8)
            ax.text(x[i], y * 1.01, stars(q), ha="center", fontsize=12.5)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[r] for r in ROIS])
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("decoding R² (variance-weighted,\nmean ± SEM, n = 8)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("  fig2: raw " + " / ".join(f"{raw[r][1].mean():.3f}" for r in ROIS) +
          "  residual " + " / ".join(f"{res[r][1].mean():.3f}" for r in ROIS))


# ----------------------------------------------------------------- figure 3
def fig_fgbg(out):
    # NOTE: the CLEAN subset, matching Methods and Results. The full-image
    # variants (roi_fgcolor_vw / roi_bgcolor_vw) are superseded, an earlier
    # version of this figure used them, which is the drift this script prevents.
    fg, bg = load("roi_fgcolor_cleanvw"), load("roi_bgcolor_cleanvw")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(3)
    w = 0.36
    fv = np.array([fg[r][1].mean() for r in ROIS])
    bv = np.array([bg[r][1].mean() for r in ROIS])
    fe = np.array([fg[r][1].std(ddof=1) / np.sqrt(8) for r in ROIS])
    be = np.array([bg[r][1].std(ddof=1) / np.sqrt(8) for r in ROIS])
    ax.bar(x - w / 2, fv, w, yerr=fe, capsize=3, color="#DD8452",
           label="foreground (inside object box)")
    ax.bar(x + w / 2, bv, w, yerr=be, capsize=3, color="#4C93B0", label="background")
    ps = [paired(fg[r][1] - bg[r][1])[1] for r in ROIS]
    qs = bh(ps)
    for i, q in enumerate(qs):
        y = max(fv[i] + fe[i], bv[i] + be[i]) * 1.10
        s = stars(q)
        ax.text(x[i], y, s, ha="center", fontsize=12.5,
                color="k" if s != "n.s." else "grey")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[r] for r in ROIS])
    ax.set_ylim(0, max((fv + fe).max(), (bv + be).max()) * 1.30)
    ax.set_ylabel("decoding R² (variance-weighted,\nmean ± SEM, n = 8)")
    # Legend ABOVE the axes: with only three x positions there is no interior
    # space wide enough for it, and an inside legend overlapped the bars.
    ax.legend(frameon=False, fontsize=11, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("  fig3: fg " + " / ".join(f"{v:.3f}" for v in fv) +
          "  bg " + " / ".join(f"{v:.3f}" for v in bv) +
          "  (clean subset)")


# ----------------------------------------------------------------- figure 4
def fig_forest(out, statsfile="stats_table_v2.json"):
    """
    Forest plot of the group-level contrasts.

    Values, CIs and significance are READ FROM the generated statistics table,
    never recomputed here. Recomputing would apply FDR across only the contrasts
    drawn in this figure, which is a different family from the one Table S3
    corrects within — the same p would then earn a different number of stars in
    the figure than in the table.
    """
    import json
    stats = {r["contrast"]: r for r in json.load(open(statsfile))}

    blocks = [
        ("Dissociation", "#4C72B0", [
            ("higher − early  (color)", "Dissociation: higher − early (colour)"),
            ("higher − early  (interaction)", "Interaction (colour−luminance): higher − early"),
        ]),
        ("Object-identity collapse — physical target", "#C44E52", [
            ("early:  raw − residual", "Collapse (physical): early, raw − residual"),
            ("higher: raw − residual", "Collapse (physical): higher, raw − residual"),
            ("higher − early  (interaction)", "Interaction (physical): collapse, higher − early"),
        ]),
        ("Object-identity collapse — perceptual target", "#8C3B39", [
            ("early:  raw − residual", "Collapse (perceptual): early, raw − residual"),
            ("higher: raw − residual", "Collapse (perceptual): higher, raw − residual"),
            ("higher − early  (interaction)", "Interaction (perceptual): collapse, higher − early"),
        ]),
        ("Residual target: early − higher", "#55A868", [
            ("early − higher  (physical)", "Residual target (physical): early − higher"),
            ("early − higher  (perceptual)", "Residual target (perceptual): early − higher"),
        ]),
        ("Attention / retinotopy", "#8C7853", [
            ("FG − BG:  early", "Attention: FG − BG, early"),
            ("FG − BG:  higher", "Attention: FG − BG, higher visual cortex"),
            ("(BG−FG):  early − higher", "Retinotopy: (BG−FG) early − higher"),
        ]),
    ]
    rows = []
    for title, colr, items in blocks:
        rows.append(("header", title, colr, None, None, None, None))
        for lab, key in items:
            r = stats[key]
            rows.append(("row", lab, colr, r["mean"], r["ci"][0], r["ci"][1], r["sig"]))

    n_rows = len([r for r in rows if r[0] == "row"])
    fig, ax = plt.subplots(figsize=(8.6, 0.40 * n_rows + 0.46 * len(blocks) + 1.6))
    ypos, yticks, ylabels, headers = 0, [], [], []
    for kind, lab, colr, obs, lo, hi, sig in rows[::-1]:
        if kind == "header":
            # Kept OUT of the tick labels: block titles are much longer than the
            # contrast labels, and as ticks they set the left margin for the
            # whole figure, pushing the axes right and squeezing the data.
            headers.append((ypos, lab, colr))
            ypos += 1.0
            continue
        alpha = 1.0 if sig != "n.s." else 0.35
        ax.plot([lo, hi], [ypos, ypos], color=colr, lw=2.6, alpha=alpha,
                solid_capstyle="round")
        ax.plot([obs], [ypos], "o", mfc="white", mec=colr, mew=2, ms=8, alpha=alpha)
        ax.text(hi + 0.003, ypos, sig, va="center", fontsize=12,
                color="grey" if sig == "n.s." else "k")
        yticks.append(ypos); ylabels.append(lab)
        ypos += 1
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_ylim(-0.9, ypos)
    for ypos_h, lab, colr in headers:
        ax.text(0.0, ypos_h, lab, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=11.5, color=colr,
                fontweight="bold", clip_on=False)
    ax.set_xlabel("paired difference in decoding R²  (mean, 95% bootstrap CI)")
    ax.set_title("Group-level contrasts (n = 8)\n"
                 "“interaction” = a difference of differences\n"
                 "** q < 0.01   * q < 0.05   (faded = n.s.)\n"
                 "q as in Table S3 (FDR within family)",
                 fontsize=11, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  fig4: {n_rows} contrasts, q values read from {statsfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("Regenerating manuscript figures from the _v2 summaries")
    fig_dissociation(f"{args.outdir}/manu_fig1_dissociation.png")
    fig_dissociation(f"{args.outdir}/manu_fig1_dissociation_v2.png",
                     interaction_panel=True)
    fig_collapse(f"{args.outdir}/fig2b_ranking_combined.png")
    fig_fgbg(f"{args.outdir}/fig2_foreground_background.png")
    fig_forest(f"{args.outdir}/fig4_forest_stats.png")
    print(f"written to {args.outdir}/")


if __name__ == "__main__":
    main()
