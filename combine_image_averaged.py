#!/usr/bin/env python3
"""
combine_image_averaged.py
=========================
The contrasts that matter for reviewer comment [19], which a single run of
check_image_averaged_r2.py cannot produce.

Why this is separate
--------------------
check_image_averaged_r2.py scores ONE target and reports one between-region
contrast (higher visual minus early). That contrast is protected by a property we
measured: averaging within image lifts every ROI by nearly the same amount, so a
common lift cancels in a difference between regions.

That argument does not extend to contrasts BETWEEN TARGETS, and those are where
the manuscript's strongest claims live:

    residualisation collapse    raw minus residual, WITHIN an ROI
    collapse interaction        (collapse in higher) minus (collapse in early)
    double dissociation         (colour minus luminance) x (higher minus early)

Different targets have different variance and noise structure, so the lift need
not be the same for each, and a raw-minus-residual difference gets no automatic
cancellation. This script computes those contrasts from the saved per-participant
numbers under both aggregations. No refitting: it is arithmetic on results that
already exist.

Inputs are the three JSON files written by check_image_averaged_r2.py, each
holding, per participant and per ROI, a triple:

    (R2 across trials, R2 averaged within image, R2 image-averaged at fixed
     pooling weights)

Anchors from STATS_AUDIT.md, for the trial-level column to be checked against:

    collapse (physical), higher            +0.057   8/8
    collapse interaction, higher - early   +0.021   8/8
    double dissociation, higher - early    +0.023   8/8

Usage
-----
    python combine_image_averaged.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROIS = ["early_v1v3", "v4_color", "concept"]
HI, EARLY = "concept", "early_v1v3"
AGG = {0: "per trial", 1: "per image", 2: "image, fixed w"}

# Anchors are DERIVED from the published per-participant summaries, not typed
# in. An earlier version hardcoded STATS_AUDIT's +0.057 for the collapse and
# reported a mismatch — but that figure is the 80-category residual
# (color_targets_residual.npy) while these runs use the 171-category one
# (color_targets_bothresid.npy), whose published value is +0.0616. The run was
# right and the constant was wrong. Reading the anchor from the file that
# matches the target removes the chance of comparing against the wrong one.
PUBLISHED = {"raw": "roi_color_vw_v2_8subj_summary.npy",
             "lum": "roi_luminance_vw_v2_8subj_summary.npy",
             "resid": "roi_colorbothresid_vw_8subj_summary.npy",
             # The 80-category (things-only) residual. Kept because it is the
             # figure STATS_AUDIT.md quotes (+0.057), and someone reading these
             # files in six months will otherwise have no way to tell which
             # residual a number came from.
             "resid80": "roi_colorresid_vw_v2_8subj_summary.npy"}


def published(path):
    """{subject: {roi: R2}} from a published summary, or None if absent."""
    if not os.path.exists(path):
        return None
    p = np.load(path, allow_pickle=True).item()
    if "by_subj" not in p:
        return None
    bs = p["by_subj"]
    try:
        return {int(s): {r: float(np.mean(bs[s][r]["ov"])) for r in ROIS}
                for s in p["subjects"]}
    except (KeyError, TypeError):
        return None


def load(path):
    d = json.load(open(path))
    ps = d["per_subject"]
    return ({int(s): {r: tuple(v) for r, v in row.items()}
             for s, row in ps.items()},
            d.get("k_matched"), d.get("target"))


def sign_flip_p(x):
    n = len(x)
    obs = abs(x.mean())
    return sum(1 for m in range(2 ** n)
               if abs(np.array([(1 if (m >> i) & 1 else -1)
                                for i in range(n)]).dot(x) / n)
               >= obs - 1e-12) / 2 ** n


def report(name, per_subject_diff, subs, anchor=None):
    print(f"\n  {name}")
    print(f"    {'aggregation':16s}{'mean':>9s}{'agree':>8s}{'p':>9s}"
          f"{'vs trial':>10s}")
    base = None
    for j in (0, 1, 2):
        x = np.array([per_subject_diff[j][s] for s in subs])
        if base is None:
            base = x.mean()
        print(f"    {AGG[j]:16s}{x.mean():+9.4f}"
              f"{f'{(x > 0).sum()}/{len(x)}':>8s}{sign_flip_p(x):9.4f}"
              f"{x.mean() - base:+10.4f}")
    if anchor is not None:
        x0 = np.array([per_subject_diff[0][s] for s in subs])
        ok = abs(x0.mean() - anchor) < 0.003
        print(f"    published anchor {anchor:+.4f} -> "
              f"{'matches' if ok else 'DOES NOT MATCH — do not quote'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="image_averaged_r2.json")
    ap.add_argument("--lum", default="image_averaged_lum.json")
    ap.add_argument("--resid", default="image_averaged_resid.json",
                    help="171-category (thing+stuff) residual")
    ap.add_argument("--resid80", default="image_averaged_resid80.json",
                    help="80-category (things-only) residual; optional, "
                         "reported for comparison if present")
    args = ap.parse_args()

    for f in (args.raw, args.lum, args.resid):
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
    raw, k1, t1 = load(args.raw)
    lum, k2, t2 = load(args.lum)
    res, k3, t3 = load(args.resid)

    if not (k1 == k2 == k3):
        raise SystemExit(f"the three runs used different k ({k1}, {k2}, {k3}); "
                         f"they are not comparable")
    subs = sorted(set(raw) & set(lum) & set(res))
    print(f"  k = {k1}, participants {subs}")
    for lab, t in (("raw", t1), ("luminance", t2), ("residual", t3)):
        print(f"    {lab:10s} {t}")

    # Anchors computed from the published summaries that match these targets.
    P = {k: published(v) for k, v in PUBLISHED.items()}
    for k_, v in PUBLISHED.items():
        print(f"    anchor[{k_}]: {v} -> "
              f"{'loaded' if P[k_] else 'UNAVAILABLE, that check will be skipped'}")

    def anch(fn):
        try:
            return float(np.mean([fn(s) for s in subs])) if all(P.values()) \
                else None
        except (KeyError, TypeError):
            return None

    # ---- 1. residualisation collapse, within each ROI --------------------
    for roi in ROIS:
        diff = {j: {s: raw[s][roi][j] - res[s][roi][j] for s in subs}
                for j in (0, 1, 2)}
        report(f"collapse (raw - residual) in {roi}", diff, subs,
               anchor=anch(lambda s: P["raw"][s][roi] - P["resid"][s][roi])
               if (P["raw"] and P["resid"]) else None)

    # ---- 2. collapse interaction ----------------------------------------
    diff = {j: {s: (raw[s][HI][j] - res[s][HI][j])
                   - (raw[s][EARLY][j] - res[s][EARLY][j]) for s in subs}
            for j in (0, 1, 2)}
    report(f"collapse interaction: ({HI} - {EARLY})", diff, subs,
           anchor=anch(lambda s: (P["raw"][s][HI] - P["resid"][s][HI])
                       - (P["raw"][s][EARLY] - P["resid"][s][EARLY]))
           if (P["raw"] and P["resid"]) else None)

    # ---- 3. double dissociation ------------------------------------------
    diff = {j: {s: (raw[s][HI][j] - lum[s][HI][j])
                   - (raw[s][EARLY][j] - lum[s][EARLY][j]) for s in subs}
            for j in (0, 1, 2)}
    report(f"double dissociation: (colour - luminance) x ({HI} - {EARLY})",
           diff, subs,
           anchor=anch(lambda s: (P["raw"][s][HI] - P["lum"][s][HI])
                       - (P["raw"][s][EARLY] - P["lum"][s][EARLY]))
           if (P["raw"] and P["lum"]) else None)

    # ---- 4. does the lift differ BETWEEN targets? ------------------------
    #
    # This is the mechanism the whole question turns on. If image-averaging
    # lifts the raw and residual targets by different amounts, a within-ROI
    # difference between them does not get the cancellation that protects the
    # between-region contrasts.
    print(f"\n  === lift from image-averaging, by target and ROI ===")
    print(f"    {'roi':12s}{'raw':>10s}{'luminance':>11s}{'residual':>10s}"
          f"{'max gap':>10s}")
    for roi in ROIS:
        lifts = [np.mean([d[s][roi][1] - d[s][roi][0] for s in subs])
                 for d in (raw, lum, res)]
        print(f"    {roi:12s}" + "".join(f"{v:+10.4f}" for v in lifts)
              + f"{max(lifts) - min(lifts):10.4f}")
    # ---- 5. the 80-category residual, for comparison ---------------------
    if os.path.exists(args.resid80):
        r80, k80, t80 = load(args.resid80)
        if k80 != k1:
            print(f"\n  {args.resid80} used k = {k80}, not {k1} — SKIPPED")
        else:
            print(f"\n  === 80-category residual ({t80}) ===")
            for roi in ROIS:
                diff = {j: {s: raw[s][roi][j] - r80[s][roi][j] for s in subs}
                        for j in (0, 1, 2)}
                report(f"collapse (raw - residual80) in {roi}", diff, subs,
                       anchor=anch(lambda s: P["raw"][s][roi]
                                   - P["resid80"][s][roi])
                       if (P["raw"] and P["resid80"]) else None)
            diff = {j: {s: (raw[s][HI][j] - r80[s][HI][j])
                           - (raw[s][EARLY][j] - r80[s][EARLY][j])
                        for s in subs} for j in (0, 1, 2)}
            report(f"collapse80 interaction: ({HI} - {EARLY})", diff, subs,
                   anchor=anch(lambda s: (P["raw"][s][HI] - P["resid80"][s][HI])
                               - (P["raw"][s][EARLY] - P["resid80"][s][EARLY]))
                   if (P["raw"] and P["resid80"]) else None)
            print(f"\n    80 vs 171 categories, collapse at {HI}, per trial: "
                  f"{np.mean([raw[s][HI][0]-r80[s][HI][0] for s in subs]):+.4f}"
                  f" vs "
                  f"{np.mean([raw[s][HI][0]-res[s][HI][0] for s in subs]):+.4f}")
    else:
        print(f"\n  ({args.resid80} not present — 80-category comparison "
              f"skipped)")

    print(f"\n    A common lift cancels in any difference. Where the gap "
          f"between targets is\n    large relative to the contrast, it does "
          f"not, and that contrast needs\n    reporting under both "
          f"aggregations rather than argued away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
