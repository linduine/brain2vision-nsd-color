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
        --color-targets data/color_targets.npy --n-draws 25 \
        --out roi_color_4subj.png

Downloads each subject's betas on first use (~1.5 GB each), then caches them.
Saves a figure and a .npy of all numbers for later write-up.
"""

import os
import gc
import argparse
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain2vision.roi import ROI_SETS, load_roi_masks
from brain2vision.color_decode import build_xy, train_eval
from brain2vision.color_targets import COLOR_NAMES


def _matched_draws(X, y, is_test, k, n_draws, labels, r2_weighting="uniform",
                   model="ridge"):
    """Per-subject: subsample to k voxels, decode, average over draws."""
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    rng = np.random.default_rng(0)
    ov, t1, pc = [], [], []
    with contextlib.redirect_stdout(open(os.devnull, "w")):  # hush inner prints
        for _ in range(draws):
            cols = (np.arange(n_vox) if k >= n_vox
                    else rng.choice(n_vox, k, replace=False))
            r = train_eval(X[:, cols], y, is_test, model=model, labels=labels,
                           r2_weighting=r2_weighting)
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
    p.add_argument("--n-draws", type=int, default=25,
                   help="Random voxel subsamples per ROI, averaged. The result "
                        "is insensitive to this (draw variance << subject SEM).")
    p.add_argument("--match-voxels", type=int, default=0)
    p.add_argument("--r2-weighting", choices=["uniform", "variance"], default="uniform",
                   help="how to pool per-colour R2 into overall R2. 'variance' "
                        "weights each colour by how much it varies, so near-absent "
                        "colours (e.g. purple) don't dominate -- recommended for "
                        "small/selected image subsets.")
    p.add_argument("--model",
                   choices=["ridge", "elasticnet", "kernel", "svr", "mlp"],
                   default="ridge",
                   help="decoder. ridge/elasticnet/svr = linear; kernel (RBF)/mlp "
                        "= nonlinear. svr = linear SVM (canonical MVPA decoder). "
                        "Use to check the dissociation is not a ridge artifact and "
                        "whether a nonlinear decoder changes any ROI. Nonlinear "
                        "decoders are much slower -- lower --n-draws (e.g. 3).")
    p.add_argument("--out", default="roi_color_4subj.png")
    args = p.parse_args()

    labels = args.labels.split(",") if args.labels else COLOR_NAMES

    # Match to the smallest ROI available in ANY included subject, so every
    # subject/ROI is subsampled to the same k (V4 sizes vary across subjects).
    per_subj_min = {}
    for subj in args.subjects:
        s_sizes = {s: int(load_roi_masks(subj, f).sum()) for s, f in ROI_SETS.items()}
        per_subj_min[subj] = min(s_sizes.values())
    k = args.match_voxels or min(per_subj_min.values())
    print(f"smallest ROI per subject {per_subj_min} -> match k = {k}")

    # ---- resumable per-subject checkpointing -------------------------------
    # Results are saved after EVERY subject, so a crash never loses more than
    # the subject in progress; re-running the same command resumes automatically.
    ckpt = args.out.replace(".png", "_summary.npy")
    cfg = dict(k=k, labels=labels, model=args.model, r2=args.r2_weighting,
               target=args.target)
    by_subj = {}
    if os.path.exists(ckpt):
        try:
            prev = np.load(ckpt, allow_pickle=True).item()
            if all(prev.get(kk) == vv for kk, vv in cfg.items()):
                by_subj = prev.get("by_subj", {})
                done = [s for s in args.subjects if s in by_subj]
                if done:
                    print(f"resuming from {ckpt}: subjects {done} already done")
            else:
                print(f"checkpoint {ckpt} has a different config -> starting fresh")
        except Exception as e:
            print(f"could not read checkpoint ({e}) -> starting fresh")

    def _save_ckpt(final_summary=None):
        order = [s for s in args.subjects if s in by_subj]
        agg = {s: {"per": [np.asarray(by_subj[j][s]["per"]) for j in order],
                   "ov": [by_subj[j][s]["ov"] for j in order],
                   "t1": [by_subj[j][s]["t1"] for j in order]} for s in ROI_SETS}
        blob = {"agg": agg, "by_subj": by_subj, "subjects": order, **cfg}
        if final_summary is not None:
            blob["summary"] = final_summary
        np.save(ckpt, np.array(blob, dtype=object), allow_pickle=True)

    for subj in args.subjects:
        if subj in by_subj:
            print(f"\n=== subject {subj} (from checkpoint) ===")
            for s in ROI_SETS:
                print(f"  {s:11s} R2={by_subj[subj][s]['ov']:+.3f} "
                      f"top1={by_subj[subj][s]['t1']:.3f}")
            continue
        print(f"\n=== subject {subj} ===")
        res = {}
        for s, f in ROI_SETS.items():
            X, y, te = build_xy(subj, args.target, rois=f)
            ov, t1, per = _matched_draws(X, y, te, k, args.n_draws, labels,
                                         r2_weighting=args.r2_weighting, model=args.model)
            del X, y, te; gc.collect()          # free the ROI matrix before next ROI
            res[s] = {"ov": ov, "t1": t1, "per": per}
            print(f"  {s:11s} R2={ov:+.3f} top1={t1:.3f}")
        by_subj[subj] = res
        _save_ckpt()
        print(f"  [checkpoint saved after subject {subj} -> {ckpt}]")

    agg = {s: {"per": [np.asarray(by_subj[j][s]["per"]) for j in args.subjects],
               "ov": [by_subj[j][s]["ov"] for j in args.subjects],
               "t1": [by_subj[j][s]["t1"] for j in args.subjects]} for s in ROI_SETS}

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
    fig.savefig(args.out, dpi=130); plt.close(fig)
    _save_ckpt(final_summary=summary)     # final write includes the summary
    print(f"\nSaved {args.out} and {ckpt}")


if __name__ == "__main__":
    main()
