"""
check_betas.py
==============
Empirical check of what the MindEye2 beta release actually contains, so the
Methods description of preprocessing is verified rather than assumed.

The MindEye2 appendix (A.2) states that the betas are single-trial estimates
from GLMsingle, restricted to `nsdgeneral`, and normalised by voxel-wise
z-scoring using training-split statistics. This script tests that description
against the files in data/:

  * matrix shape and dtype
  * whether voxels are in fact ~zero-mean / ~unit-variance (i.e. z-scored)
  * whether the normalisation looks GLOBAL (whole run) or per-session: NSD has
    750 trials per session, so if z-scoring were applied session-wise, each
    750-trial block would have mean ~0 and sd ~1 individually. If it were
    applied globally, block means will drift away from 0.

Run from the repository root:
    python check_betas.py

Requires h5py and the betas files in data/ (not redistributed; see DATA_TERMS.md).
"""

import glob
import os

import h5py
import numpy as np

TRIALS_PER_SESSION = 750  # NSD: 750 stimulus trials per scanning session


def first_dataset(f):
    for k in f.keys():
        if isinstance(f[k], h5py.Dataset):
            return k
    raise ValueError("no top-level dataset")


def report(path, n_trials=9000, voxel_stride=100):
    with h5py.File(path, "r") as f:
        d = f[first_dataset(f)]
        print(f"\n{os.path.basename(path)}")
        print(f"  shape {d.shape}  dtype {d.dtype}"
              f"   ({d.shape[0]:,} trials x {d.shape[1]:,} nsdgeneral voxels)")
        n = min(n_trials, d.shape[0])
        X = d[:n][:, ::voxel_stride].astype(np.float64)

    print(f"  sampled {X.shape[0]:,} trials x {X.shape[1]:,} voxels")
    print(f"  overall            mean {X.mean():+.4f}   sd {X.std():.4f}")
    print(f"  per-voxel mean     mean {X.mean(0).mean():+.4f}   sd {X.mean(0).std():.4f}")
    print(f"  per-voxel sd       mean {X.std(0).mean():+.4f}   sd {X.std(0).std():.4f}")

    zscored = abs(X.mean(0).mean()) < 0.1 and abs(X.std(0).mean() - 1) < 0.2
    print(f"  -> voxel-wise z-scored: {'YES' if zscored else 'NO / not exactly'}")

    print(f"  per-session blocks of {TRIALS_PER_SESSION} trials:")
    drift = []
    for s in range(0, X.shape[0] - TRIALS_PER_SESSION + 1, TRIALS_PER_SESSION):
        blk = X[s:s + TRIALS_PER_SESSION]
        drift.append(blk.mean())
        print(f"    trials {s:6,}-{s + TRIALS_PER_SESSION:6,}"
              f"   mean {blk.mean():+.4f}   sd {blk.std():.4f}")
    if drift:
        spread = float(np.std(drift))
        print(f"  -> session-mean spread = {spread:.4f}: "
              + ("consistent with SESSION-WISE normalisation"
                 if spread < 0.02 else
                 "session means drift -> normalisation looks GLOBAL, not session-wise"))


def check_trial_counts(subj, n_betas_rows):
    """
    Sanity-check the trial bookkeeping against the betas matrix itself.

    The betas matrix has one row per acquired trial, so it is the ground truth
    for how many trials exist. If the behav scan returns more entries than
    there are rows, some trials are being read more than once. (The NSD design
    ceiling of 40 x 750 = 30,000 is only an upper bound; participants who
    completed fewer sessions have fewer rows, so the row count is the correct
    reference, not the ceiling.)
    """
    from brain2vision.color_decode import read_behav_alignment

    rows, ids, is_test = read_behav_alignment(subj)
    n, n_test = len(rows), int(is_test.sum())
    uniq_pairs = len({(int(r), int(i)) for r, i in zip(rows, ids)})

    print(f"\nsubj{subj:02d} trial bookkeeping")
    if n_betas_rows is None:
        # Betas file absent: the highest row index still bounds the matrix.
        n_betas_rows = int(rows.max()) + 1
        print(f"  betas rows (inferred, file absent)  {n_betas_rows:,}")
    else:
        print(f"  betas rows (trials acquired)  {n_betas_rows:,}")
    sessions = n_betas_rows / TRIALS_PER_SESSION
    print(f"                                = {sessions:.0f} sessions x {TRIALS_PER_SESSION}")
    print(f"  behav entries read            {n:,}")
    print(f"  unique (betas row, image id)  {uniq_pairs:,}"
          f"   duplicate reads: {n - uniq_pairs:,}")
    print(f"  flagged held-out              {n_test:,}"
          f"   (expected ~{3000 * sessions / 40:,.0f} shared trials)")
    print(f"  unique images                 {len(set(ids.tolist())):,}")

    if uniq_pairs == n_betas_rows and n == n_betas_rows:
        print("  -> clean: every trial read exactly once")
    elif uniq_pairs == n_betas_rows:
        print(f"  -> {n - uniq_pairs:,} trials are read TWICE. Unique trials match the")
        print("     betas rows exactly, so this is re-reading, not extra data.")
        print("     Run check_split.py to see whether the extra copies land in the")
        print("     test shards (harmless) or straddle train and test (leakage).")
    else:
        print("  -> unique trials do NOT match the betas rows; alignment is wrong.")


ALL_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8)


def main():
    paths = sorted(glob.glob("data/betas_all_subj*_fp32_renorm.hdf5"))
    present = {int(os.path.basename(p).split("subj")[1][:2]): p for p in paths}
    missing = [s for s in ALL_SUBJECTS if s not in present]

    print("=" * 70)
    print("BETA RELEASE CHECK  (verifies the Methods description of preprocessing)")
    print("=" * 70)
    print(f"\nThe study used {len(ALL_SUBJECTS)} participants. Betas files present on disk: "
          f"{len(present)} ({', '.join(f'subj{s:02d}' for s in sorted(present)) or 'none'}).")
    if missing:
        print(f"NOT CHECKED HERE: {', '.join(f'subj{s:02d}' for s in missing)} — "
              "files absent (they are ~2 GB each and are re-downloaded on demand).")
        print("The normalisation check below therefore covers only part of the sample.")

    for s in sorted(present):
        report(present[s])
    print("\nNote: this describes the release as downloaded. Our own analysis")
    print("z-scores voxels again using TRAINING-trial statistics only, inside")
    print("the train/test split of this study (see color_decode.train_eval).")

    # The bookkeeping check reads only the small behav arrays, so it can and
    # should run for ALL participants regardless of which betas files are local.
    print("\n" + "=" * 70)
    print(f"TRIAL BOOKKEEPING CHECK  (all {len(ALL_SUBJECTS)} participants)")
    print("=" * 70)
    for subj in ALL_SUBJECTS:
        n_rows = None
        if subj in present:
            with h5py.File(present[subj], "r") as f:
                n_rows = f[first_dataset(f)].shape[0]
        try:
            check_trial_counts(subj, n_rows)
        except Exception as e:                      # network / cache missing
            print(f"\nsubj{subj:02d}: skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
