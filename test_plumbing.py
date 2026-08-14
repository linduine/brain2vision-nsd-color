"""
test_plumbing.py
================
Verification of the DATA path, as opposed to the analysis logic.

`test_analysis.py` checks that the maths is right. This file checks that the
right numbers reached the maths: that targets correspond to the images they
claim to, that the residualised target really had object information removed,
that the in-box/out-of-box masking is arithmetically consistent, and that the
held-out split is what we say it is.

These are checks on the CODE PATH, not on construct validity. In particular, the
foreground/background masking check verifies the masking arithmetic; it does not
assert that a rectangular bounding box cleanly separates foreground from
background, which it does not (see the Methods caveat in the paper).

These checks run on the ACTUAL derived targets used in the paper, so they test
the real artefacts rather than a reconstruction of them.

Run:
    python test_plumbing.py

Requires the derived targets in data/ (no betas, no image file needed for most
checks). Checks whose inputs are missing are skipped and reported as such.
"""

import os
import sys

import numpy as np

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))


def skip(name, why):
    print(f"  [skip] {name}   ({why})")


def load(stem):
    p = f"data/{stem}.npy"
    return np.load(p) if os.path.exists(p) else None


def max_abs_corr(A, B, stride=4):
    """Largest |correlation| between any column of A and any column of B."""
    out = 0.0
    for j in range(0, A.shape[1], stride):
        if A[:, j].std() < 1e-9:
            continue
        for k in range(B.shape[1]):
            if B[:, k].std() < 1e-9:
                continue
            out = max(out, abs(np.corrcoef(A[:, j], B[:, k])[0, 1]))
    return out


def main():
    print("=" * 70)
    print("DATA-PLUMBING VERIFICATION  (does the right data reach the analysis?)")
    print("=" * 70)

    col, cid = load("color_targets"), load("color_targets_ids")
    if col is None:
        print("\n  data/color_targets.npy not found — nothing to check.")
        sys.exit(1)

    # ---------------------------------------------------------------- 1
    print("\n1. IMAGE-ID ALIGNMENT ACROSS TARGETS")
    print("   Every target must be indexed by the same image ids, or rows refer")
    print("   to different images than the analysis assumes.")
    for stem in ["color_targets_residual", "color_targets_perceptual",
                 "color_targets_perceptual_residual", "luminance_targets",
                 "semantic_targets", "fgbg_fg", "fgbg_bg"]:
        ids = load(stem + "_ids")
        if ids is None:
            skip(f"{stem} ids", "file absent")
        else:
            check(f"{stem:34s} ids identical to colour ids", np.array_equal(cid, ids))

    # ---------------------------------------------------------------- 2
    print("\n2. FOREGROUND/BACKGROUND MASKING (arithmetic consistency only)")
    print("   In-box and out-of-box pixels tile the image, so")
    print("   coverage*fg + (1-coverage)*bg must reconstruct the full-image histogram.")
    print("   NOTE: this checks the MASKING CODE, not whether a rectangular box is a")
    print("   good definition of 'foreground'. That construct is coarse by design.")
    fg, bg, cov = load("fgbg_fg"), load("fgbg_bg"), load("fgbg_coverage")
    if fg is None or bg is None or cov is None:
        skip("fg/bg partition", "fgbg targets absent")
    else:
        ok = ((~np.isnan(fg).any(1)) & (~np.isnan(bg).any(1))
              & (cov > 0.01) & (cov < 0.99)
              & (fg.sum(1) > 0.99) & (bg.sum(1) > 0.99))
        recon = cov[ok, None] * fg[ok] + (1 - cov[ok, None]) * bg[ok]
        err = np.abs(recon - col[ok]).sum(1)
        check("in-box/out-of-box masking reconstructs the full-image histogram",
              np.median(err) < 0.02,
              f"median |error| = {np.median(err):.5f} over {ok.sum():,} images")
        check("reconstruction holds for essentially every image",
              np.mean(err < 0.02) > 0.99, f"{100*np.mean(err < 0.02):.1f}% within 0.02")

    # ---------------------------------------------------------------- 3
    print("\n3. RESIDUALISATION APPLIED TO THE REAL TARGETS")
    print("   The residual target must retain much less object information than the raw one.")
    sem, res = load("semantic_targets"), load("color_targets_residual")
    if sem is None or res is None:
        skip("residualisation", "semantic or residual target absent")
    else:
        m_raw, m_res = max_abs_corr(sem, col), max_abs_corr(sem, res)
        check("object information is strongly reduced in the residual",
              m_res < m_raw / 5, f"max |r| {m_raw:.3f} (raw) -> {m_res:.3f} (residual)")
        check("residual is not simply a copy of the raw target", not np.allclose(res, col))
        check("residual retains substantial variance", res.std() > 0.5 * col.std(),
              f"sd {col.std():.4f} -> {res.std():.4f}")
        pres = load("color_targets_perceptual_residual")
        pcol = load("color_targets_perceptual")
        if pres is not None and pcol is not None:
            check("same holds for the perceptual target",
                  max_abs_corr(sem, pres) < max_abs_corr(sem, pcol) / 5)

    # ---------------------------------------------------------------- 4
    print("\n4. TARGET VALIDITY")
    for stem in ["color_targets", "color_targets_perceptual", "luminance_targets"]:
        a = load(stem)
        if a is None:
            continue
        s = a.sum(1)
        check(f"{stem:26s} rows are valid distributions",
              np.mean((s > 0.99) & (s < 1.01)) > 0.999 and (a >= -1e-9).all())
    if res is not None:
        check("residual sums to ~0 per row (mean removed)", abs(res.sum(1).mean()) < 1e-3,
              f"mean row-sum = {res.sum(1).mean():+.5f}")

    # ---------------------------------------------------------------- 5
    print("\n5. PERCEPTUAL vs PHYSICAL TARGETS ARE DISTINCT MEASURES")
    per = load("color_targets_perceptual")
    if per is None:
        skip("perceptual comparison", "absent")
    else:
        r = [np.corrcoef(col[:, k], per[:, k])[0, 1] for k in range(col.shape[1])
             if col[:, k].std() > 0 and per[:, k].std() > 0]
        check("related but not identical", 0.2 < np.median(r) < 0.99,
              f"median r = {np.median(r):+.2f} (min {min(r):+.2f}, max {max(r):+.2f})")

    # ---------------------------------------------------------------- 6
    print("\n6. HELD-OUT SPLIT")
    print("   shared1000.npy is a length-73,000 BOOLEAN MASK, not a list of ids.")
    print("   Reading it as ids gives {0, 1} and marks almost nothing as held out.")
    sh = load("shared1000")
    if sh is None:
        skip("shared-1000 split", "shared1000.npy absent")
    else:
        a = np.asarray(sh).ravel()
        is_mask = a.dtype == bool or (a.size == 73000
                                      and set(np.unique(a).tolist()) <= {0, 1})
        check("shared1000.npy is read in its documented format (73,000-long mask)",
              is_mask, f"shape {a.shape}, dtype {a.dtype}")
        shared = (set(np.flatnonzero(a).tolist()) if is_mask
                  else set(a.astype(int).tolist()))
        check("it resolves to exactly 1,000 images", len(shared) == 1000,
              f"{len(shared):,} images")
        n_test = sum(int(i) in shared for i in cid)
        check("the shared-1000 images are present among our targets",
              900 < n_test < 1100, f"{n_test} of {len(cid):,} images are shared-1000")
        # NOTE: the above concerns the TARGET ids only. Whether the decoding
        # pipeline actually holds those images out is a property of the split,
        # not of this file, and is checked separately by check_split.py
        # (train/test image and trial overlap) and check_shared.py (whether two
        # participants' held-out sets coincide, as the shared protocol requires).

    # ---------------------------------------------------------------- 7
    print("\n7. CLEAN FG/BG SUBSET")
    cfg, ccov = load("fgbg_clean_fg"), load("fgbg_clean_coverage")
    cfid = load("fgbg_clean_fg_ids")
    if cfg is None or ccov is None:
        skip("clean subset", "absent")
    else:
        check("clean subset size matches the reported count", cfg.shape[0] == 16808,
              f"{cfg.shape[0]:,} images")
        check("coverage bounds match the stated filter",
              0.09 < ccov.min() < 0.11 and 0.69 < ccov.max() < 0.71,
              f"[{ccov.min():.2f}, {ccov.max():.2f}]")
        if cfid is not None:
            check("clean ids are a subset of the full id set",
                  set(cfid.tolist()) <= set(cid.tolist()))

    print("\n" + "=" * 70)
    n, k = len(results), sum(results)
    print(f"{k}/{n} checks passed")
    print("Note: the betas-to-image correspondence is verified separately, by")
    print("falsification, in brain2vision/alignment_check.py.")
    print("=" * 70)
    sys.exit(0 if k == n else 1)


if __name__ == "__main__":
    main()
