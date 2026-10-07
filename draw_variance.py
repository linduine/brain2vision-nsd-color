#!/usr/bin/env python3
"""
draw_variance.py
================
Is the decodable colour signal spread through higher visual cortex, or
concentrated in a small part of it?

Every region is subsampled to k = 397 voxels, 25 times, and the results averaged.
For V4 that is 397 of ~687 voxels (~58% of the region); for higher visual cortex
it is 397 of ~11,067 (~3.6%). If the signal in higher visual cortex lived in a
small sub-region, random 397-voxel draws would sometimes include it and sometimes
miss it, and performance would swing wildly from draw to draw.

So the across-draw spread is the diagnostic, and the cleanest version of it needs
no modelling assumptions at all:

    does the WORST of the 25 random draws still beat V4?

If yes, no small sub-region can account for the effect, because most draws
contain almost none of any given small sub-region.

Requires summaries written after the "ovd" (per-draw R2) field was added to
replicate_subjects.py. Older summaries will report the field as missing.

Usage:
    python draw_variance.py                                  # colour, _v2
    python draw_variance.py roi_colorresid_vw_v2_8subj_summary.npy
"""

from __future__ import annotations

import sys

import numpy as np

ROIS = ["early_v1v3", "v4_color", "concept"]
NICE = {"early_v1v3": "early", "v4_color": "V4", "concept": "higher"}


CANONICAL = "roi_color_vw_v2_8subj_summary.npy"


def _check_config(d: dict, path: str) -> None:
    """Warn loudly if this run's settings differ from the canonical analysis.

    The paper's runs are variance-weighted; `--r2-weighting` defaults to
    "uniform", so a hand-written command that omits the flag silently produces a
    different analysis with plausible-looking numbers. That has happened once.
    """
    import os
    cfg_keys = ("r2", "model", "k", "target")
    here = {k: d.get(k) for k in cfg_keys}
    print(f"config: " + "  ".join(f"{k}={here[k]!r}" for k in cfg_keys) + "\n")
    if not os.path.exists(CANONICAL) or os.path.abspath(path) == os.path.abspath(CANONICAL):
        return
    ref = np.load(CANONICAL, allow_pickle=True).item()
    diff = {k: (here[k], ref.get(k)) for k in cfg_keys if here[k] != ref.get(k)}
    if diff:
        print("!" * 70)
        print(f"WARNING: settings differ from {CANONICAL}:")
        for k, (a, b) in diff.items():
            print(f"    {k}: this run = {a!r}   canonical = {b!r}")
        print("  Results below are NOT comparable to the numbers in the manuscript.")
        print("!" * 70 + "\n")


def main(path: str) -> int:
    d = np.load(path, allow_pickle=True).item()
    agg = d["agg"]
    subs = d.get("subjects", list(range(1, 9)))
    _check_config(d, path)

    if "ovd" not in agg[ROIS[0]] or not any(len(x) for x in agg[ROIS[0]]["ovd"]):
        print(f"{path}: no per-draw values ('ovd') stored.\n"
              "Re-run the decode after the replicate_subjects.py change:\n"
              "    bash rerun_after_split_fix.sh main   # with FORCE=1 to overwrite")
        return 2

    print(f"Per-draw R² spread: {path}\n")
    print(f"{'region':8s}{'draws':>7s}{'mean':>9s}{'SD':>8s}{'min':>9s}{'max':>9s}"
          f"{'SD/mean':>9s}")

    per_region = {}
    for r in ROIS:
        draws = [np.asarray(v, float) for v in agg[r]["ovd"]]
        flat = np.concatenate(draws) if draws else np.array([])
        # within-subject SD across draws, then averaged over subjects: the
        # quantity we care about is how much ONE subject's estimate moves when
        # the voxels are redrawn, not how much subjects differ from each other.
        sds = [x.std(ddof=1) for x in draws if len(x) > 1]
        sd = float(np.mean(sds)) if sds else float("nan")
        mean = float(np.mean([x.mean() for x in draws]))
        per_region[r] = dict(draws=draws, mean=mean, sd=sd,
                             lo=float(min(x.min() for x in draws)),
                             hi=float(max(x.max() for x in draws)))
        n = len(draws[0])
        print(f"{NICE[r]:8s}{n:>7d}{mean:>9.4f}{sd:>8.4f}"
              f"{per_region[r]['lo']:>9.4f}{per_region[r]['hi']:>9.4f}"
              f"{sd/mean if mean else float('nan'):>9.2f}")

    # between-subject SEM, for scale
    print()
    for r in ROIS:
        ov = np.asarray(agg[r]["ov"], float)
        sem = ov.std(ddof=1) / np.sqrt(len(ov))
        print(f"  {NICE[r]:8s} between-subject SEM = {sem:.4f}   "
              f"within-subject across-draw SD = {per_region[r]['sd']:.4f}   "
              f"ratio = {per_region[r]['sd']/sem:.2f}")

    # ---- how concentrated can the signal be? --------------------------------
    # Graded version first: the binary "worst draw" test discards 24 of every 25
    # numbers and is hostage to a single unlucky sample. The fraction of draws
    # that beat V4 uses all of them and degrades gracefully.
    print("\n" + "=" * 68)
    print("Concentration test: what fraction of random higher-visual draws beat")
    print("that participant's near-complete V4 sample?")
    print("=" * 68)
    hi_draws = [np.asarray(v, float) for v in agg["concept"]["ovd"]]
    v4_mean = np.asarray(agg["v4_color"]["ov"], float)

    tot = wins = 0
    all_pass = 0
    for s, hd, v4 in zip(subs, hi_draws, v4_mean):
        n_win = int((hd > v4).sum())
        tot += len(hd); wins += n_win
        all_pass += (n_win == len(hd))
        print(f"  participant {s}: {n_win:2d}/{len(hd)} draws beat V4 "
              f"({100 * n_win / len(hd):5.1f}%)   mean margin {hd.mean() - v4:+.4f}"
              f"   worst {hd.min() - v4:+.4f}")
    frac = 100 * wins / tot
    print(f"\n  overall: {wins}/{tot} draws = {frac:.1f}%")
    print(f"  every draw beat V4 in {all_pass}/{len(subs)} participants")

    print("\n  Reading: a random draw covers ~3.6% of higher visual cortex but")
    print("  ~58% of V4. If the signal were confined to a small sub-region, most")
    print("  draws would miss it and this fraction would be far below 100%.")
    if frac >= 90 and all_pass < len(subs):
        print(f"\n  => {frac:.0f}% is high enough to rule out a small sub-region, but")
        print("     not unanimous. Report the fraction, not 'pervasively distributed'.")
    elif all_pass == len(subs):
        print("\n  => Unanimous across participants and draws.")
    else:
        print(f"\n  => Only {frac:.0f}%. Concentration is not ruled out; report cautiously.")
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "roi_color_vw_v2_8subj_summary.npy"
    raise SystemExit(main(p))
