"""
test_build_xy_multi.py
======================
Verify that build_xy_multi returns exactly what per-ROI build_xy returned.

The refactor exists only to stop re-reading the same bytes three times per
participant. It must therefore be a pure performance change: same rows, same
columns, same order, same values. This asserts that element-wise, for one
participant, across all three ROIs — the same comparison the eventual results
depend on.

Run (needs the betas file for the chosen participant):
    python test_build_xy_multi.py --subj 3
"""

import argparse
import time

import numpy as np

from brain2vision.roi import ROI_SETS
from brain2vision.color_decode import build_xy, build_xy_multi

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=3)
    ap.add_argument("--target", default="data/color_targets.npy")
    args = ap.parse_args()

    print("=" * 70)
    print(f"build_xy_multi EQUIVALENCE CHECK (--load-once path) — subj{args.subj:02d}")
    print("=" * 70)

    print("\n-- old path: build_xy once per ROI --")
    t0 = time.time()
    old = {}
    for name, frags in ROI_SETS.items():
        old[name] = build_xy(args.subj, args.target, rois=frags)
    t_old = time.time() - t0
    print(f"   {t_old:.1f} s")

    print("\n-- new path: build_xy_multi once --")
    t0 = time.time()
    new = build_xy_multi(args.subj, args.target, ROI_SETS)
    t_new = time.time() - t0
    print(f"   {t_new:.1f} s")

    print("\n-- equivalence --")
    for name in ROI_SETS:
        Xo, yo, to = old[name]
        Xn, yn, tn = new[name]
        check(f"{name:11s} X shape", Xo.shape == Xn.shape, f"{Xo.shape} vs {Xn.shape}")
        check(f"{name:11s} X identical", np.array_equal(Xo, Xn),
              f"max |Δ| = {np.abs(Xo - Xn).max():.3e}" if Xo.shape == Xn.shape else "")
        check(f"{name:11s} y identical", np.array_equal(yo, yn))
        check(f"{name:11s} is_test identical", np.array_equal(to, tn),
              f"{int(to.sum())} vs {int(tn.sum())} held out")

    print("\n" + "=" * 70)
    n, k = len(results), sum(results)
    print(f"{k}/{n} checks passed")
    if t_new > 0:
        print(f"speedup: {t_old / t_new:.1f}x  ({t_old:.0f}s -> {t_new:.0f}s per participant)")
    print("Note: the second call benefits from the OS page cache and the memoised")
    print("alignment, so this understates the saving on a cold run.")
    print("=" * 70)
    raise SystemExit(0 if k == n else 1)


if __name__ == "__main__":
    main()
