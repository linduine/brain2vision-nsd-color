"""
alignment_check.py
==================
Falsification test for the betas <-> image alignment.

The whole study rests on `read_behav_alignment` correctly pairing each betas row
with the image the subject saw (via the webdataset `behav` columns IMG_COL /
BETAS_COL). Those column indices are inferred from MindEye2's dataloader and
sanity-checked by range, but not independently proven. This script proves them
by falsification: if the pairing is real, breaking it should destroy decoding.

We decode once with the TRUE pairing, then `n_perm` times with the target labels
permuted (a standard label-permutation null). If the alignment carries real
signal, the true R^2 sits well above the shuffled null; if the pairing were
already broken, the true R^2 would be indistinguishable from the null.

It doubles as a permutation test of "is decoding above chance at all."

Usage
-----
    python -m brain2vision.alignment_check --subj 1 --rois V4 --n-perm 20 \
        --color-targets data/color_targets.npy
"""

import os
import argparse
import contextlib
import numpy as np

from brain2vision.color_decode import build_xy, train_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subj", type=int, default=1)
    p.add_argument("--color-targets", "--target", dest="target", required=True,
                   help="Target .npy (color or luminance)")
    p.add_argument("--rois", nargs="+", default=["V4"])
    p.add_argument("--n-perm", type=int, default=20)
    args = p.parse_args()

    X, y, is_test = build_xy(args.subj, args.target, rois=tuple(args.rois))

    # --- true pairing ---
    real = train_eval(X, y, is_test, model="ridge")
    real_r2, real_top1 = real["overall_r2"], real["top1"]

    # --- label-permutation null: each betas trial gets a RANDOM image's target ---
    rng = np.random.default_rng(0)
    null_r2, null_top1 = [], []
    for _ in range(args.n_perm):
        y_perm = y[rng.permutation(len(y))]
        with contextlib.redirect_stdout(open(os.devnull, "w")):  # hush inner prints
            r = train_eval(X, y_perm, is_test, model="ridge")
        null_r2.append(r["overall_r2"]); null_top1.append(r["top1"])
    null_r2 = np.asarray(null_r2); null_top1 = np.asarray(null_top1)

    # permutation p-value for R^2 (fraction of null >= observed)
    p_r2 = (1 + int((null_r2 >= real_r2).sum())) / (1 + args.n_perm)

    print(f"\n=== alignment falsification (subj {args.subj}, ROI {args.rois}) ===")
    print(f"TRUE pairing : R2 = {real_r2:+.4f}   top1 = {real_top1:.3f}")
    print(f"SHUFFLED null: R2 = {null_r2.mean():+.4f} +/- {null_r2.std():.4f}   "
          f"top1 = {null_top1.mean():.3f} +/- {null_top1.std():.3f}   (n={args.n_perm})")
    print(f"permutation p-value (R2): {p_r2:.3f}")
    print("(note: shuffled top1 reflects the modal-color base rate, not 1/11 — "
          "R2 is the clean falsification metric)")

    ok = real_r2 > null_r2.mean() + 3 * null_r2.std() and real_r2 > 0.005
    if ok:
        print("VERDICT: PASS — true decoding is well above the shuffled null, so "
              "the betas<->image alignment carries real signal.")
    else:
        print("VERDICT: CHECK — true decoding is NOT clearly above the null; "
              "re-examine IMG_COL / BETAS_COL in color_decode.py.")


if __name__ == "__main__":
    main()
