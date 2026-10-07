"""
make_supp_figures.py
====================
Regenerate the SUPPLEMENTARY figures from the canonical result files.

Why this exists
---------------
`make_figures.py` closed this gap for the four main-text figures. The
supplementary figures still had no generator: they were produced ad hoc in
August and the code that made them is gone. Only `build_supplement.py` referred
to them, by filename, so nothing tied the pixels to the `.npy` files the text
cites — and they drifted. The split-half figure was still printing
r = 0.91 / 0.89 / 0.86 from a pre-shard-fix run while the text had been updated
to 0.90 / 0.88 / 0.84, and the per-colour figure still carries five panel titles
from the same superseded run.

Every figure below reads the same canonical files the manuscript cites, so that
class of drift cannot recur.

    S1  supp_fig_persubject.png          per-participant region profile
    S2  supp_fig_percolor.png            per-colour region profile
    S3  fig1_decode_vs_objectpred.png    decoding vs object-predictability
    S4  fig4_synthetic.png               NSD-synthetic null + positive control

The split-half figure (`reliability_v2_8subj.png`) is NOT regenerated here: it
is written directly by `brain2vision/reliability.py`, which already has a
generator and already emits the correct v2 version.

Inputs
------
S1, S2 and S4 read only committed `_summary.npy` files, so they regenerate from
a fresh clone. S3 additionally needs the per-image target arrays, which are
derived from the NSD/COCO stimuli and are therefore not redistributed; pass
their paths explicitly. Without them S3 is skipped and the others still build.

Run from the repository root:
    python make_supp_figures.py
    python make_supp_figures.py --outdir /tmp
    python make_supp_figures.py --color data/color_targets.npy \
        --semantic data/semantic_targets.npy      # to include S3
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROIS = ["early_v1v3", "v4_color", "concept"]
LEGEND = {"early_v1v3": "Early (V1–V3)", "v4_color": "V4", "concept": "Higher visual"}
NICE = {"early_v1v3": "early\nV1–V3", "v4_color": "V4", "concept": "higher\nvisual"}
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple",
           "pink", "brown", "black", "white", "gray"]
C_EARLY, C_V4, C_HIGH = "#4C72B0", "#937860", "#C44E52"

# Swatches for the per-colour panels. These are display colours for the 11 basic
# colour terms, not the HSV rules that define them (see color_targets.py).
SWATCH = {"red": "#E8000B", "orange": "#FF7C00", "yellow": "#F5D800",
          "green": "#1AC938", "blue": "#1F77B4", "purple": "#9F27B5",
          "pink": "#FF8CCF", "brown": "#8C564B", "black": "#000000",
          "white": "#FFFFFF", "gray": "#9A9A9A"}


def _load_per_ov(stem):
    """(per-colour R² [subj x 11], overall R² [subj]) per region, from a _v2 file."""
    d = np.load(f"{stem}_v2_8subj_summary.npy", allow_pickle=True).item()
    return {r: (np.asarray(d["agg"][r]["per"], float),
                np.asarray(d["agg"][r]["ov"], float)) for r in ROIS}, d


def _load_synth(stem):
    """(mean, sem) per region from a synthetic summary.

    The synthetic analyses are NOT affected by the held-out duplication fix and
    so have no _v2 rerun: `nsd_synthetic.load_synth_roi_betas` reads the
    NSD-synthetic betas directly with h5py and never calls the behav-alignment
    path that the fix corrected.
    """
    d = np.load(f"{stem}_8subj_summary.npy", allow_pickle=True).item()
    return {r: (float(d["summary"][r][0]), float(d["summary"][r][1])) for r in ROIS}


def sem(x):
    return x.std(0, ddof=1) / np.sqrt(x.shape[0])


# ------------------------------------------------------------------- S1
def fig_persubject(out):
    col, d = _load_per_ov("roi_color_vw")
    subs = d.get("subjects", list(range(1, 9)))
    fig, ax = plt.subplots(figsize=(8, 3.64))
    x = np.arange(len(subs))
    w = 0.27
    for k, (r, c) in enumerate(zip(ROIS, [C_EARLY, C_V4, C_HIGH])):
        ax.bar(x + (k - 1) * w, col[r][1], w, color=c, label=LEGEND[r])
    # Which participants run against the group ordering? Annotated rather than
    # hidden, because the split-half analysis exists to ask whether these two
    # are a reliable trait or noise.
    flagged = []
    # One height for every note, above the tallest bar in the chart, so a note
    # can never sit on a neighbouring participant's bar.
    _ymax = max(col[r][1].max() for r in ROIS)
    for i in range(len(subs)):
        e, c_ = col["early_v1v3"][1][i], col["concept"][1][i]
        if e >= c_:
            flagged.append(subs[i])
            ha = "right" if i == len(subs) - 1 else "left" if i == 0 else "center"
            ax.text(x[i], _ymax * 1.15, "early ≥ higher", ha=ha, fontsize=8,
                    color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subs])
    ax.set_xlabel("participant")
    ax.set_ylabel("color decoding R²  (variance-weighted)")
    ax.set_ylim(0, max(col[r][1].max() for r in ROIS) * 1.28)
    ax.legend(frameon=False, ncol=3, fontsize=9,
              loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.12), mode="expand",
              borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  S1 persubject: early ≥ higher visual in {flagged}")
    return flagged


# ------------------------------------------------------------------- S2
def fig_percolour(out):
    col, _ = _load_per_ov("roi_color_vw")
    delta = (col["concept"][0] - col["early_v1v3"][0]).mean(0)
    order = np.argsort(delta)                      # early-dominant -> concept-climbing
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.14), sharey=True)
    axes = axes.ravel()
    for ax, j in zip(axes, order):
        name = COLOURS[j]
        m = [col[r][0][:, j].mean() for r in ROIS]
        e = [sem(col[r][0])[j] for r in ROIS]
        # The line takes the swatch colour, except for white, which is invisible
        # against the axes; it gets a grey line and keeps a white marker face.
        line_c = "#C8C8C8" if name == "white" else SWATCH[name]
        ax.errorbar([0, 1, 2], m, yerr=e, color=line_c, lw=2,
                    marker="o", ms=9, mfc=SWATCH[name], mec="k", mew=0.8,
                    capsize=3, zorder=3)
        ax.axhline(0, color="0.8", lw=1)
        ax.set_title(f"{name}  (Δ={delta[j]:+.3f})", fontsize=11)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Early", "V4", "Higher visual"])
        ax.set_xlim(-0.35, 2.35)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    fig.supylabel("color decoding R²  (variance-weighted, mean ± SEM, n=8)",
                  fontsize=10)
    fig.suptitle("Per-color region profile (ordered early-dominant → higher-visual-climbing)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("  S2 percolour: Δ = " +
          ", ".join(f"{COLOURS[j]} {delta[j]:+.3f}" for j in order))
    return {COLOURS[j]: delta[j] for j in order}


# ------------------------------------------------------------------- S3
def _object_predictability(color_npy, semantic_npy, n_folds=5, seed=0):
    """Out-of-sample R²(colour ~ object presence), per colour term.

    Same estimator, folds and seed as `semantic_residual.residualize`, so the
    x-axis here is the identical quantity that defines the residual target used
    throughout the paper. Reimplemented rather than imported because
    `residualize` writes the residual target to disk as a side effect, and this
    figure must not overwrite it.
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from brain2vision.semantic_residual import _aligned

    y, S, _ = _aligned(color_npy, semantic_npy)
    y_sem = np.zeros_like(y)
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(S):
        y_sem[te] = RidgeCV(alphas=np.logspace(0, 5, 10)).fit(S[tr], y[tr]).predict(S[te])
    return r2_score(y, y_sem, multioutput="raw_values")


def fig_decode_vs_objectpred(out, color_npy, semantic_npy):
    col, _ = _load_per_ov("roi_color_vw")
    xs = _object_predictability(color_npy, semantic_npy)
    ys = col["concept"][0].mean(0)
    r = float(np.corrcoef(xs, ys)[0, 1])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    b, a = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
    ax.plot(xx, a + b * xx, "--", color="0.6", lw=1.2, zorder=1)
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    for j, name in enumerate(COLOURS):
        ax.scatter(xs[j], ys[j], s=190, color=SWATCH[name], edgecolor="k",
                   linewidth=1.0, zorder=3)
    ax.text(0.03, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=13, va="top")
    ax.set_xlabel("object-predictability of the color\nR²(color ~ object identity)")
    ax.set_ylabel("higher visual decoding R² for the color")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    # Labels go below the marker by default, falling back to above, then right,
    # then left, whenever that position would land on another marker or on a
    # label already placed. Positions are read after tight_layout, so the
    # coordinates the test runs against are the ones that get drawn.
    fig.canvas.draw()
    _pts = ax.transData.transform(np.column_stack([xs, ys]))
    _CAND = [(0, -20, "center", "top"), (0, 14, "center", "bottom"),
             (15, -4, "left", "center"), (-15, -4, "right", "center")]
    _placed, _mr = [], 9.0
    for j, name in enumerate(COLOURS):
        w, h = 6.4 * len(name), 11.0
        pick = _CAND[0]
        for dx, dy, ha, va in _CAND:
            cx, cy = _pts[j][0] + dx, _pts[j][1] + dy
            x0 = cx - w / 2 if ha == "center" else (cx if ha == "left" else cx - w)
            y0 = cy - h if va == "top" else (cy if va == "bottom" else cy - h / 2)
            box = (x0, y0, x0 + w, y0 + h)
            hits_marker = any(box[0] < q[0] + _mr and box[2] > q[0] - _mr and
                              box[1] < q[1] + _mr and box[3] > q[1] - _mr
                              for k, q in enumerate(_pts) if k != j)
            hits_label = any(box[0] < c[2] and box[2] > c[0] and
                             box[1] < c[3] and box[3] > c[1] for c in _placed)
            if not (hits_marker or hits_label):
                pick = (dx, dy, ha, va); _placed.append(box); break
        dx, dy, ha, va = pick
        ax.annotate(name, (xs[j], ys[j]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=10)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  S3 decode-vs-objectpred: r = {r:.3f}")
    return r, xs


# ------------------------------------------------------------------- S4
def fig_synthetic(out):
    hue = _load_synth("roi_synth_hue_avgfull")
    ident = _load_synth("roi_synth_chromID")
    chance = 1.0 / 64                       # identity among the 64 chromatic images
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.33))
    pal = ["#4C72B0", "#DD8452", "#55A868"]

    ax = axes[0]
    ax.bar([0, 1, 2], [hue[r][0] for r in ROIS],
           yerr=[hue[r][1] for r in ROIS], color=pal, capsize=0,
           error_kw=dict(lw=1.5))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("A  hue not decodable", loc="left", fontsize=10)
    ax.set_ylabel("hue R² (mean ± SEM, n = 8)")

    ax = axes[1]
    ax.bar([0, 1, 2], [ident[r][0] for r in ROIS],
           yerr=[ident[r][1] for r in ROIS], color=pal, capsize=0,
           error_kw=dict(lw=1.5))
    ax.axhline(chance, color="red", ls="--", lw=1.2, label="chance (1/64)")
    # Name the stimulus set. `roi_synth_chromID` is identity among the 64
    # CHROMATIC nsdsynthetic images, not the full 284-image set — and there is a
    # separate `roi_synth_identity` analysis over all 284 with different values
    # (0.158 / 0.088 / 0.046), so an unqualified "image identity" is ambiguous.
    ax.set_title("B  image identity IS decodable\n(positive control: the 64 chromatic images)",
                 loc="left", fontsize=10)
    ax.set_ylabel("image-identity top-1")
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    for ax in axes:
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([NICE[r] for r in ROIS])
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  S4 synthetic: hue " +
          " / ".join(f"{hue[r][0]:+.4f}" for r in ROIS) +
          "  identity " + " / ".join(f"{ident[r][0]:.4f}" for r in ROIS))
    return hue, ident


# ------------------------------------------------------------------- S5
def fig_ncsnr(out, ncsnr_npy="ncsnr_quality_v2_ncsnr.npy"):
    """Signal quality vs decoding.

    `ncsnr_quality.py` computes the ncsnr values and can draw a simple 3-panel
    scatter, but the figure the supplement actually uses is this two-panel
    version, which had no generator. It is rebuilt here from the two committed
    files — the stored per-participant ncsnr and the _v2 decode summary — so it
    needs neither a download nor nibabel.

    The ncsnr values themselves are unaffected by the held-out duplication fix:
    ncsnr is a per-voxel quantity NSD publishes, averaged within ROIs, with no
    trial split involved. Only the decoding R² it is plotted against moved, and
    that is why the old figure printed r = 0.22 where the text says 0.20.
    """
    nc = np.load(ncsnr_npy, allow_pickle=True).item()
    subs = nc["subjects"]
    n = {r: np.asarray(nc["ncsnr"][r], float) for r in ROIS}
    col, _ = _load_per_ov("roi_color_vw")
    R2 = {r: col[r][1] for r in ROIS}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.54))
    ax = axes[0]
    for r, c in zip(ROIS, [C_EARLY, C_V4, C_HIGH]):
        ax.scatter(n[r], R2[r], s=38, color=c, zorder=3, label=LEGEND[r])
        ax.scatter(n[r].mean(), R2[r].mean(), s=320, marker="X", color=c,
                   edgecolor="k", linewidth=1.4, zorder=4)
    # Arrow drawn between the group means, so it tracks the data rather than
    # being placed by hand.
    x0, y0 = n["early_v1v3"].mean(), R2["early_v1v3"].mean()
    x1, y1 = n["concept"].mean(), R2["concept"].mean()
    ax.annotate("", xy=(x1 - 0.005, y1 + 0.014), xytext=(x0 - 0.005, y0 + 0.006),
                arrowprops=dict(arrowstyle="->", color="0.25", lw=1.6))
    ax.text(x1 - 0.005, y1 + 0.020, "worse signal,\nbetter decoding",
            fontsize=10, color="0.25", ha="left")
    ax.set_xlabel("mean noise-ceiling SNR (ncsnr)  →  better data quality")
    ax.set_ylabel("color decoding R²  (variance-weighted)")
    ax.set_title("A  Higher visual decodes best despite\n"
                 "the worst signal quality",
                 fontsize=9.5, loc="left")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=True, framealpha=0.92, edgecolor="none",
              loc="lower left", fontsize=8,
              title="× = group mean", title_fontsize=8)

    ax = axes[1]
    dn = n["early_v1v3"] - n["concept"]
    dR = R2["early_v1v3"] - R2["concept"]
    r = float(np.corrcoef(dn, dR)[0, 1])
    ax.scatter(dn, dR, s=55, color="0.25", zorder=3)
    for i, s in enumerate(subs):
        ax.annotate(f"S{s}", (dn[i], dR[i]), textcoords="offset points",
                    xytext=(9, -11) if dR[i] > 0 else (9, 5),
                    fontsize=8, color="0.45")
    b, a = np.polyfit(dn, dR, 1)
    xx = np.linspace(dn.min() - 0.01, dn.max() + 0.015, 50)
    ax.plot(xx, a + b * xx, "--", color="0.7", lw=1.2, zorder=1)
    ax.axhline(0, color="0.8", lw=1)
    ax.set_xlabel("ncsnr difference  (early − higher visual)")
    ax.set_ylabel("decoding difference  (early − higher visual)")
    ax.set_title("B  Individual region profile only weakly\n"
                 f"tied to relative data quality  (r = {r:.2f}, n.s.)",
                 fontsize=9.5, loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  S5 ncsnr: mean ncsnr " +
          " / ".join(f"{n[r].mean():.2f}" for r in ROIS) +
          " | r(ncsnr, R²) " + " / ".join(f"{np.corrcoef(n[r], R2[r])[0,1]:+.2f}" for r in ROIS) +
          f" | difference r = {r:+.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--color", default="data/color_targets.npy")
    ap.add_argument("--semantic", default="data/semantic_targets.npy")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("Regenerating supplementary figures from the canonical summaries")
    fig_persubject(f"{args.outdir}/supp_fig_persubject.png")
    fig_percolour(f"{args.outdir}/supp_fig_percolor.png")
    fig_synthetic(f"{args.outdir}/fig4_synthetic.png")
    fig_ncsnr(f"{args.outdir}/ncsnr_quality_v2.png")
    if os.path.exists(args.color) and os.path.exists(args.semantic):
        fig_decode_vs_objectpred(f"{args.outdir}/fig1_decode_vs_objectpred.png",
                                 args.color, args.semantic)
    else:
        print(f"  S3 SKIPPED: needs {args.color} and {args.semantic}. These are "
              "derived from the NSD/COCO stimuli and are not redistributed; "
              "rebuild them with color_targets.py / semantic_targets.py.")
    print(f"written to {args.outdir}/")


if __name__ == "__main__":
    main()
