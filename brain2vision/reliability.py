"""
reliability.py
==============
Split-half reliability of the per-participant region decoding.

Motivation
----------
Between-participant variation in the region ranking (e.g. a participant whose
early visual cortex decodes colour better than the concept region) is only worth
interpreting if it is RELIABLE rather than measurement noise. Agreement across
decoders (ridge / elastic-net / SVR) does NOT establish this, because those
decoders reuse the same trials; only a split of the DATA can.

What it does
------------
For each participant, all trials are split into two IMAGE-disjoint halves by
NSD-image-id parity, so every repeat of a given image stays in the same half
(no image leakage between halves). The standard voxel-count-matched decode is run
independently within each half, yielding two per-region R^2 estimates per
participant. It then reports the across-participant agreement between the halves:

  * Pearson r(A, B) of per-region R^2 across participants (reliability of level),
  * r(A, B) of the early-minus-concept difference, and how often its sign agrees
    across halves (reliability of the region ordering -- the S6/S8 question).

Requires the `build_xy(..., return_ids=True)` addition to color_decode.py.

Usage
-----
    python -m brain2vision.reliability --subjects 1 2 3 4 5 6 7 8 \
        --target data/color_targets.npy --r2-weighting variance \
        --model ridge --n-draws 5 --out reliability_8subj.png

Notes
-----
Each half uses half the test trials, so half-R^2 is noisier than the full
estimate -- expected for split-half. With n=8 participants the reliability r is
itself indicative, not definitive; read it together with the scatter plot.
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
from brain2vision.color_decode import build_xy, build_xy_multi, train_eval
from brain2vision.color_targets import COLOR_NAMES


def _decode(X, y, te, k, n_draws, labels, r2_weighting, model):
    """Voxel-count-matched decode; overall R^2 averaged over random draws."""
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    rng = np.random.default_rng(0)
    ov = []
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        for _ in range(draws):
            cols = (np.arange(n_vox) if k >= n_vox
                    else rng.choice(n_vox, k, replace=False))
            r = train_eval(X[:, cols], y, te, model=model, labels=labels,
                           r2_weighting=r2_weighting)
            ov.append(r["overall_r2"])
    return float(np.mean(ov))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--target", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--n-draws", type=int, default=5)
    p.add_argument("--match-voxels", type=int, default=0)
    p.add_argument("--r2-weighting", choices=["uniform", "variance"], default="variance")
    p.add_argument("--model", choices=["ridge", "elasticnet", "kernel", "svr", "mlp"],
                   default="ridge")
    p.add_argument("--out", default="reliability_8subj.png")
    p.add_argument("--load-once", action="store_true",
                   help="read the betas file once into the union of the ROI masks "
                        "instead of once per ROI. Faster; peak memory becomes the "
                        "union matrix plus one ROI matrix.")
    args = p.parse_args()
    labels = args.labels.split(",") if args.labels else COLOR_NAMES

    # match every region/participant to the smallest region's voxel count
    per_subj_min = {}
    for subj in args.subjects:
        sizes = {s: int(load_roi_masks(subj, f).sum()) for s, f in ROI_SETS.items()}
        per_subj_min[subj] = min(sizes.values())
    k = args.match_voxels or min(per_subj_min.values())
    print(f"match k = {k}")

    ckpt = args.out.replace(".png", "_summary.npy")
    cfg = dict(k=k, labels=labels, model=args.model, r2=args.r2_weighting,
               target=args.target)
    by_subj = {}
    if os.path.exists(ckpt):
        try:
            prev = np.load(ckpt, allow_pickle=True).item()
            if all(prev.get(kk) == vv for kk, vv in cfg.items()):
                by_subj = prev.get("by_subj", {})
                if by_subj:
                    print(f"resuming from {ckpt}: subjects {sorted(by_subj)} done")
        except Exception as e:
            print(f"could not read checkpoint ({e}) -> fresh")

    def _save():
        np.save(ckpt, np.array({"by_subj": by_subj, "subjects": list(args.subjects),
                                **cfg}, dtype=object), allow_pickle=True)

    for subj in args.subjects:
        if subj in by_subj:
            print(f"=== subject {subj} (from checkpoint) ===")
            continue
        print(f"\n=== subject {subj} ===")
        res = {}
        # Default: one ROI at a time (peak memory = one ROI matrix, as before).
        # The webdataset rescan is avoided either way, the alignment is memoised.
        data = (build_xy_multi(subj, args.target, ROI_SETS, return_ids=True)
                if args.load_once else None)
        for s, f in ROI_SETS.items():
            X, y, te, ids = (data.pop(s) if data is not None else
                             build_xy(subj, args.target, rois=f, return_ids=True))
            isA = (np.asarray(ids).astype(np.int64) % 2 == 0)   # image-disjoint halves
            res[s] = {}
            for tag, m in (("A", isA), ("B", ~isA)):
                res[s][tag] = _decode(X[m], y[m], te[m], k, args.n_draws,
                                      labels, args.r2_weighting, args.model)
            del X, y, te; gc.collect()
            print(f"  {s:11s} A={res[s]['A']:+.3f}  B={res[s]['B']:+.3f}")
        del data; gc.collect()
        by_subj[subj] = res
        _save()
        print(f"  [checkpoint saved -> {ckpt}]")

    # ---- across-participant reliability ------------------------------------
    rois = list(ROI_SETS)
    A = {s: np.array([by_subj[j][s]["A"] for j in args.subjects]) for s in rois}
    B = {s: np.array([by_subj[j][s]["B"] for j in args.subjects]) for s in rois}

    print("\n=== split-half reliability (across participants) ===")
    for s in rois:
        r = np.corrcoef(A[s], B[s])[0, 1]
        print(f"  {s:11s} r(A,B) = {r:+.3f}")
    dA = A["early_v1v3"] - A["concept"]
    dB = B["early_v1v3"] - B["concept"]
    rd = np.corrcoef(dA, dB)[0, 1]
    agree = int(np.sum(np.sign(dA) == np.sign(dB)))
    print(f"\n  early-minus-concept: r(A,B) = {rd:+.3f}; "
          f"sign agrees in {agree}/{len(dA)} participants")
    print("  (high r + sign agreement => the region profile is a reliable individual "
          "trait; low => it is measurement noise)")

    # scatter A vs B per region
    # Display names, not the internal ROI keys, and one shared axis range
    # across the panels so the three are directly comparable.
    NICE = {"early_v1v3": "Early (V1–V3)", "v4_color": "V4",
            "concept": "Higher visual"}
    lim = [min(min(A[s].min(), B[s].min()) for s in rois) - 0.005,
           max(max(A[s].max(), B[s].max()) for s in rois) + 0.005]
    fig, axes = plt.subplots(1, len(rois), figsize=(8, 3.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, rois):
        ax.scatter(A[s], B[s], zorder=3)
        ax.plot(lim, lim, "k--", lw=.7)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{NICE.get(s, s)}\nr = {np.corrcoef(A[s], B[s])[0, 1]:+.2f}",
                     fontsize=10)
        ax.set_xlabel("half A  R²", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("half B  R²", fontsize=9)
    fig.tight_layout()
    # Equal-aspect panels need the extra height, and the tight bbox keeps the
    # titles and axis labels from being cropped at the figure edge.
    fig.savefig(args.out, dpi=160, bbox_inches="tight"); plt.close(fig)
    _save()
    print(f"\nSaved {args.out} and {ckpt}")


if __name__ == "__main__":
    main()
