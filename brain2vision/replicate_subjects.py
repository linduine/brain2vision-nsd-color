"""
replicate_subjects.py
=====================
Replicate the voxel-count-matched ROI color comparison across multiple subjects
and summarise the result as mean +/- SEM over subjects.

This is the group-level test: a per-subject pattern (e.g. early visual decoding
"black"/luminance best, higher visual decoding chromatic/object colors best) is
only trustworthy if it holds across people. Each ROI is matched to the smallest
ROI's voxel count (V4's ~687) and decoded over several random voxel draws; the
per-subject mean is then averaged across subjects.

Usage
-----
    python -m brain2vision.replicate_subjects --subjects 1 2 5 7 \
        --color-targets data/color_targets.npy --n-draws 5 \
        --out roi_color_4subj.png

Downloads each subject's betas on first use (~1.5 GB each), then caches them.
Saves a figure and a .npy of all numbers for later write-up.
"""

import os
import argparse
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain2vision.roi import ROI_SETS, load_roi_masks
from brain2vision.color_decode import build_xy, train_eval
from brain2vision.color_targets import COLOR_NAMES


def _matched_draws(X, y, is_test, k, n_draws, labels):
    """Per-subject: subsample to k voxels, decode, average over draws."""
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    rng = np.random.default_rng(0)
    ov, t1, pc = [], [], []
    with contextlib.redirect_stdout(open(os.devnull, "w")):  # hush inner prints
        for _ in range(draws):
            cols = (np.arange(n_vox) if k >= n_vox
                    else rng.choice(n_vox, k, replace=False))
            r = train_eval(X[:, cols], y, is_test, model="ridge", labels=labels)
            ov.append(r["overall_r2"]); t1.append(r["top1"])
            pc.append([r["per_color_r2"][c] for c in labels])
    return float(np.mean(ov)), float(np.mean(t1)), np.mean(pc, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 5, 7])
    p.add_argument("--color-targets", "--target", dest="target", required=True,
                   help="Path to a target .npy (color, luminance, ...)")
    p.add_argument("--labels", default=None,
                   help="Comma-separated column names; default = 11 colors")
    p.add_argument("--n-draws", type=int, default=5)
    p.add_argument("--match-voxels", type=int, default=0)
    p.add_argument("--out", default="roi_color_4subj.png")
    args = p.parse_args()

    labels = args.labels.split(",") if args.labels else COLOR_NAMES

    sizes = {s: int(load_roi_masks(args.subjects[0], f).sum())
             for s, f in ROI_SETS.items()}
    k = args.match_voxels or min(sizes.values())
    print(f"sizes {sizes} -> match k = {k}")

    agg = {s: {"per": [], "ov": [], "t1": []} for s in ROI_SETS}
    for subj in args.subjects:
        print(f"\n=== subject {subj} ===")
        for s, f in ROI_SETS.items():
            X, y, te = build_xy(subj, args.target, rois=f)
            ov, t1, per = _matched_draws(X, y, te, k, args.n_draws, labels)
            del X
            agg[s]["per"].append(per); agg[s]["ov"].append(ov); agg[s]["t1"].append(t1)
            print(f"  {s:11s} R2={ov:+.3f} top1={t1:.3f}")

    print("\n=== across-subject summary (mean +/- SEM) ===")
    sem = lambda a: a.std(0, ddof=1) / np.sqrt(a.shape[0])
    summary = {}
    for s in ROI_SETS:
        ov = np.array(agg[s]["ov"]); t1 = np.array(agg[s]["t1"])
        per = np.array(agg[s]["per"])
        summary[s] = (per.mean(0), sem(per), ov.mean(), sem(ov), t1.mean(), sem(t1))
        print(f"{s:11s} R2={ov.mean():+.3f}+/-{sem(ov):.3f}  "
              f"top1={t1.mean():.3f}+/-{sem(t1):.3f}")

    print("\nper-target R2 (mean across subjects):")
    print("target     " + "  ".join(f"{s[:7]:>7s}" for s in ROI_SETS))
    for i, c in enumerate(labels):
        print(f"{c:9s} " + "  ".join(f"{summary[s][0][i]:+7.3f}" for s in ROI_SETS))

    # plot: per-target mean across subjects, error bars = SEM across subjects
    sets = list(ROI_SETS); x = np.arange(len(labels)); w = 0.8 / len(sets)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, s in enumerate(sets):
        ax.bar(x + i * w, summary[s][0], w, yerr=summary[s][1], capsize=2,
               label=f"{s} R2={summary[s][2]:+.3f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w * (len(sets) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"test R2 (mean +/- SEM, n={len(args.subjects)})")
    ax.set_title(f"Decoding by ROI, {len(args.subjects)} subjects, "
                 f"matched to {k} voxels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    np.save(args.out.replace(".png", "_summary.npy"),
            np.array({"agg": agg, "summary": summary,
                      "subjects": args.subjects, "labels": labels},
                     dtype=object), allow_pickle=True)
    print(f"\nSaved {args.out} and {args.out.replace('.png', '_summary.npy')}")


if __name__ == "__main__":
    main()
