"""
compare_rois.py
===============
Fair ROI comparison of color decoding, with VOXEL-COUNT MATCHING.

Bigger ROIs decode better for the trivial reason that they have more voxels
(more signal), which confounds any claim about a region being more
color-selective. To remove that confound we subsample every ROI down to the
same number of voxels (default: the smallest ROI's size, i.e. V4's ~687),
decode, and repeat over several random draws so the result doesn't hinge on
which voxels we happened to pick. Alpha is still tuned per fit (RidgeCV).

Usage
-----
    python -m brain2vision.compare_rois --subj 1 \
        --color-targets data/color_targets.npy --n-draws 10 \
        --out roi_color_matched.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain2vision.roi import ROI_SETS, load_roi_masks
from brain2vision.color_decode import build_xy, train_eval
from brain2vision.color_targets import COLOR_NAMES


def evaluate_matched(X, y, is_test, k, n_draws, seed=0):
    """
    Subsample columns of X to k voxels, decode, repeat n_draws times.
    If k >= X's voxel count (e.g. the smallest ROI), there is nothing to
    subsample, so a single deterministic run is used.
    Returns arrays: overall (n,), top1 (n,), per (n, 11).
    """
    rng = np.random.default_rng(seed)
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    overall, top1, per = [], [], []
    for d in range(draws):
        cols = (np.arange(n_vox) if k >= n_vox
                else rng.choice(n_vox, size=k, replace=False))
        res = train_eval(X[:, cols], y, is_test, model="ridge")
        overall.append(res["overall_r2"])
        top1.append(res["top1"])
        per.append([res["per_color_r2"][c] for c in COLOR_NAMES])
    return {"overall": np.array(overall), "top1": np.array(top1),
            "per": np.array(per)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subj", type=int, default=1)
    p.add_argument("--color-targets", required=True)
    p.add_argument("--n-draws", type=int, default=10)
    p.add_argument("--match-voxels", type=int, default=0,
                   help="Target voxel count; 0 = use the smallest ROI's size")
    p.add_argument("--out", default="roi_color_matched.png")
    args = p.parse_args()

    # ROI voxel counts are cheap to get from the mask file (no betas needed).
    sizes = {s: int(load_roi_masks(args.subj, frags).sum())
             for s, frags in ROI_SETS.items()}
    k = args.match_voxels or min(sizes.values())
    print(f"\nROI voxel counts: {sizes}")
    print(f"Matching every ROI to k={k} voxels, {args.n_draws} random draws each\n")

    # Load ONE ROI at a time, evaluate, then free its betas (keeps memory low).
    results = {}
    for set_name, frags in ROI_SETS.items():
        print(f"######## {set_name} ########")
        X, y, is_test = build_xy(args.subj, args.color_targets, rois=frags)
        results[set_name] = evaluate_matched(X, y, is_test, k, args.n_draws)
        del X
        r = results[set_name]
        print(f"{set_name}: overall R²={r['overall'].mean():+.3f}"
              f"±{r['overall'].std():.3f}  "
              f"top1={r['top1'].mean():.3f}±{r['top1'].std():.3f}\n")

    # ---- plot: per-color mean R² with error bars, grouped by ROI ----
    sets = list(results)
    x = np.arange(len(COLOR_NAMES))
    width = 0.8 / len(sets)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, s in enumerate(sets):
        mean = results[s]["per"].mean(0)
        std = results[s]["per"].std(0)
        ax.bar(x + i * width, mean, width, yerr=std, capsize=2,
               label=f"{s}: R²={results[s]['overall'].mean():+.3f}, "
                     f"top1={results[s]['top1'].mean():.3f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + width * (len(sets) - 1) / 2)
    ax.set_xticklabels(COLOR_NAMES, rotation=45, ha="right")
    ax.set_ylabel("test R² (mean ± sd over draws)")
    ax.set_title(f"subj{args.subj:02d}: color decoding by ROI, "
                 f"matched to {k} voxels ({args.n_draws} draws)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")
    np.save(args.out.replace(".png", "_metrics.npy"),
            np.array(results, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
