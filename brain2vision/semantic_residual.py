"""
semantic_residual.py
====================
Chromatic-vs-semantic variance partitioning, step 1: remove from the colour
target the part that is predictable from image semantics, leaving the *residual
colour*, the chromatic variation orthogonal to which objects are present.

Method
------
Cross-validated (k-fold over images) we predict the 11-way colour distribution
from a colour-free semantic feature (COCO category presence; semantic_targets.py)
using RidgeCV, giving out-of-sample predictions y_sem. The residual is
  y_resid = y_colour - y_sem
The out-of-sample R²(colour ~ semantics) quantifies the confound itself: how much
of image colour is explained just by what objects are present.

Then re-run the ROI comparison on `y_resid` (replicate_subjects --target ...):
  * if higher visual cortex still decodes residual colour best -> its colour
    advantage is genuinely chromatic;
  * if its advantage collapses toward early/V4 -> it was riding on semantics.

Usage
-----
    pip install scikit-learn numpy
    python -m brain2vision.semantic_residual \
        --color data/color_targets.npy \
        --semantic data/semantic_targets.npy \
        --out data/color_targets_residual.npy
    # then:
    python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 \
        --target data/color_targets_residual.npy --out roi_colorresid_8subj.png
"""

import argparse
import os

import numpy as np


def _aligned(color_npy, semantic_npy):
    y = np.load(color_npy); yid = np.load(color_npy.replace(".npy", "_ids.npy"))
    S = np.load(semantic_npy); sid = np.load(semantic_npy.replace(".npy", "_ids.npy"))
    smap = {int(i): k for k, i in enumerate(sid)}
    keep = np.array([int(i) in smap for i in yid])
    y, yid = y[keep], yid[keep]
    S = S[np.array([smap[int(i)] for i in yid])]
    return y.astype(np.float64), S.astype(np.float64), yid


# Mirrors color_targets.COLOR_NAMES. Duplicated deliberately: importing that
# module pulls in the whole package (and h5py), and a failed import must not
# silently relabel a colour target as c0..cN.
_BASIC_COLOUR_TERMS = ["red", "orange", "yellow", "green", "blue",
                       "purple", "pink", "brown", "black", "white", "gray"]


def _target_labels(target_npy, n_cols, labels=None):
    """Column names for the target, and a noun for the printed report.

    This module is not colour-specific: it residualises whatever target it is
    given, including the luminance (brightness-bin) target. Labelling brightness
    bins "red, orange, yellow …", which it used to do unconditionally, produces
    output that is numerically right and verbally wrong, so the names are now
    derived from the target rather than assumed.
    """
    if labels:
        names = [s.strip() for s in labels.split(",")]
        if len(names) != n_cols:
            raise ValueError(f"--labels has {len(names)} names, target has {n_cols} columns")
        return names, "target"
    stem = os.path.basename(target_npy).lower()
    if "lumin" in stem or "bright" in stem:
        return [f"L{i}" for i in range(n_cols)], "luminance"
    if "color" in stem or "colour" in stem:
        try:
            from brain2vision.color_targets import COLOR_NAMES
            names = list(COLOR_NAMES)
        except Exception:
            # Importing the package pulls in h5py; fall back to the same fixed
            # list rather than silently degrading to c0..cN on a colour target.
            names = _BASIC_COLOUR_TERMS
        if len(names) == n_cols:
            return names, "colour"
        print(f"  [warning: {n_cols} columns but {len(names)} colour names; "
              f"using generic labels. Pass --labels to name them.]")
    return [f"c{i}" for i in range(n_cols)], "target"


def residualize(color_npy, semantic_npy, out, n_folds=5, seed=0, labels=None):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    y, S, ids = _aligned(color_npy, semantic_npy)
    names, noun = _target_labels(color_npy, y.shape[1], labels)
    print(f"aligned: {noun} {y.shape}, predictors {S.shape}")

    y_sem = np.zeros_like(y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(S):
        reg = RidgeCV(alphas=np.logspace(0, 5, 10)).fit(S[tr], y[tr])
        y_sem[te] = reg.predict(S[te])
    resid = (y - y_sem).astype(np.float32)

    r2_overall = r2_score(y, y_sem)
    r2_per = r2_score(y, y_sem, multioutput="raw_values")

    np.save(out, resid)
    np.save(out.replace(".npy", "_ids.npy"), ids)

    # sklearn's default is a uniform average over columns; the manuscript reports
    # variance-weighted R², so both are printed to stop the two being conflated.
    r2_vw = r2_score(y, y_sem, multioutput="variance_weighted")
    print(f"\nOut-of-sample R²({noun} ~ predictors):")
    print(f"   uniform average   = {r2_overall:.3f}")
    print(f"   variance-weighted = {r2_vw:.3f}   <- comparable to the decoding R²")
    print(f"  (how much image {noun} is explained by the annotated content)")
    print(f"per-column R²({noun} ~ predictors):")
    for name, r in zip(names, r2_per):
        print(f"   {name:8s} {r:+.3f}")
    print(f"\nSaved residual {noun} target -> {out}")
    print("Next: decode it per ROI with replicate_subjects and compare to the "
          "original result. For a non-colour target, pass the matching --labels.")
    return resid, r2_overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--color", required=True, help="color_targets.npy")
    p.add_argument("--semantic", required=True, help="semantic_targets.npy")
    p.add_argument("--out", default="data/color_targets_residual.npy")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--labels", default=None,
                   help="comma-separated column names; inferred from the target "
                        "filename if omitted (colour names / L0..Ln for luminance)")
    args = p.parse_args()
    residualize(args.color, args.semantic, args.out, n_folds=args.n_folds,
                labels=args.labels)


if __name__ == "__main__":
    main()
