"""
test_elasticnet_speedup.py
==========================
A/B the elastic-net solver settings against results already computed under the
previous settings.

n_jobs and selection="random" are solver-level choices: they change how
coordinate descent reaches the minimum of the penalised least-squares problem,
not what that minimum is. So the fitted R2 should reproduce the stored values to
within the solver's own tolerance. This asserts that on real data instead of
taking it on faith.

The comparison is exact in the sense that matters: the same participant, the
same target, the same voxel-matching seed, the same alpha grid, only the solver
configuration differs.

Run (needs the betas for the chosen participant):
    python test_elasticnet_speedup.py --subj 1
    python test_elasticnet_speedup.py --subj 1 --tol 0.02
"""

import argparse
import time

import numpy as np

from brain2vision.roi import ROI_SETS, load_roi_masks
from brain2vision.color_decode import build_xy, train_eval
from brain2vision.color_targets import COLOR_NAMES

REF = "roi_color_elasticnet_v2_8subj_summary.npy"


def matched_draws(X, y, te, k, n_draws, labels):
    """Mirror of replicate_subjects._matched_draws (same rng, same seed)."""
    n_vox = X.shape[1]
    draws = n_draws if k < n_vox else 1
    rng = np.random.default_rng(0)
    ov = []
    for _ in range(draws):
        cols = (np.arange(n_vox) if k >= n_vox
                else rng.choice(n_vox, k, replace=False))
        r = train_eval(X[:, cols], y, te, model="elasticnet", labels=labels,
                       r2_weighting="variance")
        ov.append(r["overall_r2"])
    return float(np.mean(ov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--target", default="data/color_targets.npy")
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--tol", type=float, default=None,
                    help="only to report what a tolerance change would do; "
                         "not applied unless given")
    args = ap.parse_args()

    ref = np.load(REF, allow_pickle=True).item()
    by = ref["by_subj"]
    if args.subj not in by:
        raise SystemExit(f"subj{args.subj:02d} is not in {REF}; "
                         f"available: {sorted(by)}")
    k = ref["k"]
    print("=" * 74)
    print(f"ELASTIC-NET SOLVER A/B, subj{args.subj:02d}, k={k}, "
          f"{args.n_draws} draws")
    print(f"reference: {REF} (computed with n_jobs=1, cyclic selection)")
    print("=" * 74)

    ok = True
    for roi, frags in ROI_SETS.items():
        stored = by[args.subj][roi]["ov"]
        X, y, te = build_xy(args.subj, args.target, rois=frags)
        t0 = time.time()
        got = matched_draws(X, y, te, k, args.n_draws, COLOR_NAMES)
        dt = time.time() - t0
        d = got - stored
        good = abs(d) < 0.002
        ok &= good
        print(f"\n  {roi:11s} stored {stored:+.4f}   now {got:+.4f}   "
              f"Δ {d:+.4f}   {dt:6.1f}s   {'OK' if good else '**DIFFERS**'}")
        del X, y, te

    print("\n" + "=" * 74)
    print("PASS, solver change is numerically inert" if ok else
          "FAIL, the fitted solution moved; do not adopt these settings")
    print("Tolerance for 'inert' is |Δ| < 0.002, an order of magnitude below the")
    print("smallest effect the paper reports (+0.006).")
    print("=" * 74)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
