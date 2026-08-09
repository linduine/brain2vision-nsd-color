"""
synthetic_decode.py
===================
Voxel-matched hue decoding on NSD-synthetic, across ROIs and subjects -- the
colour-without-semantics control.

The test
--------
For each ROI (early V1-V3 / V4 / concept), restricted to the isoluminant
chromatic pink-noise images, can a linear decoder read out the stimulus HUE
(as a circular cos/sin target) from the betas? Because these images have no
objects, no figure, and constant luminance, a hue signal here is genuinely
chromatic -- it cannot be a semantic, attentional-foreground, or luminance
artifact. We therefore expect:

    * if the NSD-core "concept decodes colour best" result was structure-bound,
      the concept advantage should NOT reappear here, and early visual cortex
      should decode hue at least as well (consistent with NSD-synthetic's own
      finding that chromatic noise activates EVC > HVC);
    * if the concept pool has a genuine chromatic code, it should still decode
      hue competitively.

Method (matched to the NSD-core pipeline)
-----------------------------------------
Each ROI is subsampled to a common voxel count k (smallest ROI across the
included subjects), decoded with RidgeCV, and averaged over random voxel draws.
Evaluation is out-of-sample via GROUP k-fold, grouping by image so an image's
two task conditions (fixation / one-back) never straddle the train/test split.
Reported as R^2 on the 2-D (cos, sin) hue target, mean +/- SEM across subjects.

Usage
-----
    pip install scikit-learn numpy matplotlib nibabel h5py requests
    # 1) build hue targets from the colour stimuli (once):
    python -m brain2vision.synthetic_targets \
        --images nsdsynthetic_colorstimuli.hdf5 --out data/synth
    # 2) decode:
    python -m brain2vision.synthetic_decode --subjects 1 2 3 4 5 6 7 8 \
        --targets data/synth --out roi_synth_hue_8subj.png
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain2vision.nsd_synthetic import SYNTH_ROI_SETS, load_synth_roi_betas


def _cv_r2(X, y, groups, n_folds=5, seed=0, model="ridge"):
    """
    Out-of-sample R^2 via GroupKFold (leave-image-out). model: 'ridge' (linear),
    'kernel' (RBF KernelRidge, nonlinear), or 'mlp' (nonlinear). The kernel/mlp
    options test whether hue is decodable nonlinearly where linear ridge finds
    nothing.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    k = min(n_folds, len(np.unique(groups)))
    yhat = np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if model == "kernel":
            from sklearn.kernel_ridge import KernelRidge
            reg = KernelRidge(kernel="rbf", alpha=1.0, gamma=1.0 / Xtr.shape[1])
        elif model == "mlp":
            from sklearn.neural_network import MLPRegressor
            reg = MLPRegressor(hidden_layer_sizes=(64,), max_iter=800, random_state=0)
        else:
            from sklearn.linear_model import RidgeCV
            reg = RidgeCV(alphas=np.logspace(0, 5, 10))
        reg.fit(Xtr, y[tr]); yhat[te] = reg.predict(Xte)
    return float(r2_score(y, yhat))


def _matched_draws(X, y, groups, k, n_draws, seed=0, model="ridge"):
    """Subsample to k voxels, decode, average over draws."""
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        cols = (np.arange(n_vox) if k >= n_vox
                else rng.choice(n_vox, k, replace=False))
        out.append(_cv_r2(X[:, cols], y, groups, model=model))
    return float(np.mean(out))


def _identity_topk(X, groups, k, n_draws, seed=0):
    """
    POSITIVE CONTROL. Leave-one-trial-out nearest-prototype image classifier ->
    top-1 accuracy. Tests whether the ROI carries ANY decodable stimulus identity,
    using only the trial->image map (no hue, no colour-file offset). If this is
    well above chance (1/n_images) but hue R^2 is null, the beta/trial pipeline is
    sound and the problem is the hue labels; if identity also fails, the problem
    is upstream (beta<->trial alignment). Uses correlation between z-scored trial
    patterns and per-image mean prototypes, with a leave-one-out correction.
    """
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    rng = np.random.default_rng(seed)
    uimg, inv = np.unique(groups, return_inverse=True)
    n_img = len(uimg); counts = np.bincount(inv); ng = counts[inv]
    accs = []
    for _ in range(draws):
        cols = (np.arange(n_vox) if k >= n_vox
                else rng.choice(n_vox, k, replace=False))
        Z = X[:, cols]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)                 # z per voxel
        Zt = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-6)
        P = np.stack([Zt[inv == i].mean(0) for i in range(n_img)])  # prototypes
        S = Zt @ P.T / Zt.shape[1]                              # (trials, images)
        # leave-one-out correction on each trial's own-image column
        selfP = P[inv]
        loo = (ng[:, None] * selfP - Zt) / np.maximum(ng[:, None] - 1, 1)
        loo_sim = np.einsum("tk,tk->t", Zt, loo) / Zt.shape[1]
        S[np.arange(len(inv)), inv] = np.where(ng > 1, loo_sim, -np.inf)
        accs.append(float(np.mean(S.argmax(1) == inv)))
    return float(np.mean(accs)), 1.0 / n_img


def _cv_classacc(X, yc, groups, n_folds=5):
    """Grouped-CV multiclass accuracy (RidgeClassifier), leave-image-out."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeClassifierCV
    k = min(n_folds, len(np.unique(groups)))
    pred = np.zeros(len(yc), dtype=int)
    for tr, te in GroupKFold(n_splits=k).split(X, yc, groups):
        mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
        clf = RidgeClassifierCV(alphas=np.logspace(0, 5, 10))
        clf.fit((X[tr] - mu) / sd, yc[tr])
        pred[te] = clf.predict((X[te] - mu) / sd)
    return float(np.mean(pred == yc))


def _average_by_image(X, groups, y=None):
    """Collapse each image's repeat trials into one mean pattern (higher SNR)."""
    uimg = np.unique(groups)
    Xa = np.stack([X[groups == g].mean(0) for g in uimg])
    if y is None:
        return Xa, uimg
    ya = np.stack([y[groups == g][0] for g in uimg])   # label constant per image
    return Xa, ya, uimg


def _matched_class(X, yc, groups, k, n_draws, seed=0):
    """Voxel-matched multiclass accuracy, averaged over draws."""
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        cols = (np.arange(n_vox) if k >= n_vox
                else rng.choice(n_vox, k, replace=False))
        out.append(_cv_classacc(X[:, cols], yc, groups))
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 5, 7])
    p.add_argument("--targets", default="data/synth",
                   help="prefix from synthetic_targets.py (_hue/_chromatic/_ids)")
    p.add_argument("--n-draws", type=int, default=25)
    p.add_argument("--average", action="store_true",
                   help="hue/pattern modes: average each image's repeats into one "
                        "higher-SNR pattern before decoding (64 image-means instead "
                        "of ~146 single trials). Trades n for SNR.")
    p.add_argument("--model", choices=["ridge", "kernel", "mlp"], default="ridge",
                   help="hue-decode model: ridge (linear), kernel (RBF, nonlinear), "
                        "or mlp (nonlinear). Tests whether hue decodes nonlinearly.")
    p.add_argument("--match-voxels", type=int, default=0)
    p.add_argument("--no-match", action="store_true",
                   help="use ALL voxels in each ROI (skip voxel-count matching). "
                        "Most sensitive 'is the signal anywhere' config; not a fair "
                        "cross-ROI comparison.")
    p.add_argument("--cache", default="nsd_cache")
    p.add_argument("--betas-name", default="betas_nsdsynthetic.hdf5")
    p.add_argument("--design", default=None,
                   help="NSD-synthetic design .mat (trial->image ordering); "
                        "required for the single-trial betas layout")
    p.add_argument("--chromatic-only", action="store_true",
                   help="identity mode: restrict to the chromatic images "
                        "(global id >= chromatic-first-id). Tests whether the "
                        "chromatic responses carry ANY signal, using masterordering "
                        "alone (no colour->global mapping). Decisive check of "
                        "'weak-signal null' vs 'scrambled ordering'.")
    p.add_argument("--chromatic-first-id", type=int, default=221,
                   help="global image number of the FIRST image in the colour "
                        "stimuli file (nsdsynthetic's 64 chromatic images are the "
                        "last block, global 221..284). Local target id i maps to "
                        "global (first_id - 1 + i).")
    p.add_argument("--mode", choices=["hue", "identity", "pattern"], default="hue",
                   help="'hue' = decode stimulus hue (main experiment); "
                        "'identity' = control: can each ROI tell the 284 images "
                        "apart at all (top-1 vs chance)?; 'pattern' = control: "
                        "decode which of the 4 shared spatial noise patterns "
                        "(4-way, chance 0.25) through the same colour->global "
                        "alignment as hue -- tests the ordering assumption.")
    p.add_argument("--out", default="roi_synth_hue_4subj.png")
    args = p.parse_args()

    hue = np.load(f"{args.targets}_hue.npy")          # (n_images, 2)
    chrom = np.load(f"{args.targets}_chromatic.npy")  # (n_images,) bool
    ids = np.load(f"{args.targets}_ids.npy")          # local image numbers (1..n)
    # map each target row to its GLOBAL synthetic image number, keep only the
    # chromatic ones (the colour file already contains only chromatic images).
    off = args.chromatic_first_id - 1
    global_to_row = {int(off + i): r for r, i in enumerate(ids) if chrom[r]}
    print(f"targets: {hue.shape[0]} images, {int(chrom.sum())} chromatic "
          f"-> global ids {min(global_to_row)}..{max(global_to_row)}")

    # determine matched k across subjects/ROIs
    per_subj_min = {}
    roi_cache = {}
    for subj in args.subjects:
        sizes = {}
        for roi in SYNTH_ROI_SETS:
            X, image_of, task_of = load_synth_roi_betas(
                subj, roi, cache=args.cache, betas_name=args.betas_name,
                design_path=args.design)
            roi_cache[(subj, roi)] = (X, image_of, task_of)
            sizes[roi] = X.shape[1]
        per_subj_min[subj] = min(sizes.values())
    k = args.match_voxels or min(per_subj_min.values())
    if args.no_match:
        k = 10 ** 9   # >= any ROI size -> _matched_draws uses all voxels
        print("no voxel matching: using ALL voxels per ROI")
    else:
        print(f"smallest ROI per subject {per_subj_min} -> match k = {k}")
    ktxt = "all (no match)" if args.no_match else str(k)

    chance = None
    agg = {roi: [] for roi in SYNTH_ROI_SETS}
    for subj in args.subjects:
        print(f"\n=== subject {subj} ===")
        for roi in SYNTH_ROI_SETS:
            X, image_of, task_of = roi_cache[(subj, roi)]
            if args.mode == "identity":
                # depends only on the trial->image map (masterordering), not on
                # the colour->global mapping.
                io, Xi = image_of, X
                if args.chromatic_only:
                    m = io >= args.chromatic_first_id
                    io, Xi = io[m], X[m]
                acc, chance = _identity_topk(Xi, io, k, args.n_draws)
                agg[roi].append(acc)
                print(f"  {roi:11s} identity top1={acc:.3f} "
                      f"(chance={chance:.3f}, {len(np.unique(io))} images"
                      f"{', chromatic-only' if args.chromatic_only else ''})")
            elif args.mode == "pattern":
                # 4-way spatial pattern (row % 4), via the colour->global align.
                keep = np.array([int(i) in global_to_row for i in image_of])
                rows = np.array([global_to_row[int(i)] for i in image_of[keep]])
                Xp, patt, groups = X[keep], rows % 4, image_of[keep]
                if args.average:
                    Xp, patt, groups = _average_by_image(Xp, groups, patt)
                acc = _matched_class(Xp, patt, groups, k, args.n_draws)
                chance = 0.25
                agg[roi].append(acc)
                print(f"  {roi:11s} pattern acc={acc:.3f} "
                      f"(chance=0.25, {Xp.shape[0]} samples"
                      f"{', averaged' if args.average else ''})")
            else:
                keep = np.array([int(i) in global_to_row for i in image_of])
                rows = np.array([global_to_row[int(i)] for i in image_of[keep]])
                Xc, yc, groups = X[keep], hue[rows], image_of[keep]
                if args.average:
                    Xc, yc, groups = _average_by_image(Xc, groups, yc)
                r2 = _matched_draws(Xc, yc, groups, k, args.n_draws, model=args.model)
                agg[roi].append(r2)
                print(f"  {roi:11s} hue R2={r2:+.3f}  "
                      f"({Xc.shape[0]} samples{', averaged' if args.average else ''}, "
                      f"{len(np.unique(groups))} images)")

    metric = {"identity": "identity top1", "pattern": "pattern acc",
              "hue": "hue R2"}[args.mode]
    print("\n=== across-subject summary (mean +/- SEM) ===")
    if chance is not None:
        print(f"(chance top-1 = {chance:.3f})")
    sem = lambda a: a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    summary = {}
    for roi in SYNTH_ROI_SETS:
        a = np.array(agg[roi]); summary[roi] = (a.mean(), sem(a))
        print(f"{roi:11s} {metric}={a.mean():+.3f}+/-{sem(a):.3f}")

    sets = list(SYNTH_ROI_SETS)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(range(len(sets)), [summary[s][0] for s in sets],
           yerr=[summary[s][1] for s in sets], capsize=4,
           color=["#4C72B0", "#DD8452", "#55A868"])
    ax.axhline(0, color="k", lw=0.8)
    if chance is not None:
        ax.axhline(chance, color="r", ls="--", lw=1, label=f"chance={chance:.3f}")
        ax.legend()
    ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets)
    ax.set_ylabel(f"{metric} (mean +/- SEM, n={len(args.subjects)})")
    ax.set_title(f"NSD-synthetic {args.mode} by ROI ({ktxt} voxels)")
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    np.save(args.out.replace(".png", "_summary.npy"),
            np.array({"summary": summary, "agg": agg,
                      "subjects": args.subjects}, dtype=object), allow_pickle=True)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
