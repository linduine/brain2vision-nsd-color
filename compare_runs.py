"""
compare_runs.py
===============
Compare the pre-fix results against the re-run (_v2) results, so "materially
unchanged" is a measurement rather than an expectation.

Pairs each old summary with its _v2 counterpart and reports, per region, the
group-mean R² before and after, the difference, and whether any qualitative
conclusion in the paper would change: the ordering of the three regions, the
sign of the key contrasts, and the significance of the interactions.

Run after rerun_after_split_fix.sh:
    python compare_runs.py
"""

import os

import numpy as np

ROIS = ["early_v1v3", "v4_color", "concept"]
SHORT = {"early_v1v3": "early", "v4_color": "V4", "concept": "concept"}

PAIRS = [
    ("colour (ridge)", "roi_color_vw_8subj_summary.npy",
     "roi_color_vw_v2_8subj_summary.npy"),
    ("luminance", "roi_luminance_vw_8subj_summary.npy",
     "roi_luminance_vw_v2_8subj_summary.npy"),
    ("colour residualised", "roi_colorresid_vw_8subj_summary.npy",
     "roi_colorresid_vw_v2_8subj_summary.npy"),
    ("perceptual colour", "roi_pcolor_vw_8subj_summary.npy",
     "roi_pcolor_vw_v2_8subj_summary.npy"),
    ("perceptual residualised", "roi_pcolorresid_vw_8subj_summary.npy",
     "roi_pcolorresid_vw_v2_8subj_summary.npy"),
    ("foreground", "roi_fgcolor_cleanvw_8subj_summary.npy",
     "roi_fgcolor_cleanvw_v2_8subj_summary.npy"),
    ("background", "roi_bgcolor_cleanvw_8subj_summary.npy",
     "roi_bgcolor_cleanvw_v2_8subj_summary.npy"),
    ("elastic-net", "roi_color_elasticnet_8subj_summary.npy",
     "roi_color_elasticnet_v2_8subj_summary.npy"),
    ("linear SVR", "roi_color_svr_8subj_summary.npy",
     "roi_color_svr_v2_8subj_summary.npy"),
    ("RBF kernel", "roi_color_kernel_8subj_summary.npy",
     "roi_color_kernel_v2_8subj_summary.npy"),
    ("MLP", "roi_color_mlp2_8subj_summary.npy",
     "roi_color_mlp2_v2_8subj_summary.npy"),
]


def means(path):
    """Group-mean R² per region, and the per-participant values."""
    d = np.load(path, allow_pickle=True).item()
    out = {}
    for r in ROIS:
        ov = np.asarray(d["agg"][r]["ov"], dtype=float)
        out[r] = ov
    return out


def main():
    print("=" * 78)
    print("PRE-FIX vs RE-RUN  (group-mean R² per region, n = 8)")
    print("=" * 78)
    print(f"{'analysis':24s} {'region':8s} {'before':>8s} {'after':>8s} "
          f"{'diff':>8s} {'max|Δ| per subj':>16s}")
    print("-" * 78)

    keep = {}
    for name, old, new in PAIRS:
        if not (os.path.exists(old) and os.path.exists(new)):
            print(f"{name:24s} -- skipped ("
                  f"{'old' if not os.path.exists(old) else 'new'} file absent)")
            continue
        o, n = means(old), means(new)
        keep[name] = (o, n)
        for r in ROIS:
            ob, nb = o[r].mean(), n[r].mean()
            per = np.abs(n[r] - o[r]).max() if o[r].shape == n[r].shape else np.nan
            print(f"{name if r == ROIS[0] else '':24s} {SHORT[r]:8s} "
                  f"{ob:+8.4f} {nb:+8.4f} {nb-ob:+8.4f} {per:16.4f}")
        print("-" * 78)

    # ------------------------------------------------------------ verdicts
    print("\nQUALITATIVE CHECKS (these are what the paper actually claims)\n")

    def ordering(m):
        return tuple(sorted(ROIS, key=lambda r: -m[r].mean()))

    for name, (o, n) in keep.items():
        same = ordering(o) == ordering(n)
        print(f"  region ordering, {name:24s} "
              f"{'unchanged' if same else '**CHANGED**'}   "
              f"{' > '.join(SHORT[r] for r in ordering(n))}")

    if "colour (ridge)" in keep and "luminance" in keep:
        c, l = keep["colour (ridge)"][1], keep["luminance"][1]
        co, lo = keep["colour (ridge)"][0], keep["luminance"][0]
        inter_new = ((c["concept"] - c["early_v1v3"])
                     - (l["concept"] - l["early_v1v3"])).mean()
        inter_old = ((co["concept"] - co["early_v1v3"])
                     - (lo["concept"] - lo["early_v1v3"])).mean()
        print(f"\n  colour x region vs luminance x region interaction:"
              f"  {inter_old:+.4f} -> {inter_new:+.4f}")

    if "colour (ridge)" in keep and "colour residualised" in keep:
        raw, res = keep["colour (ridge)"][1], keep["colour residualised"][1]
        rawo, reso = keep["colour (ridge)"][0], keep["colour residualised"][0]
        col_new = ((raw["concept"] - res["concept"])
                   - (raw["early_v1v3"] - res["early_v1v3"])).mean()
        col_old = ((rawo["concept"] - reso["concept"])
                   - (rawo["early_v1v3"] - reso["early_v1v3"])).mean()
        print(f"  collapse interaction (concept minus early):"
              f"        {col_old:+.4f} -> {col_new:+.4f}")

    print("\n  Re-run brain2vision.stats on the _v2 summaries for exact p-values;")
    print("  the permutation floor at n = 8 is 2/2^8 = 0.0078.")


if __name__ == "__main__":
    main()
