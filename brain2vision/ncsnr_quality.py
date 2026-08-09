"""
ncsnr_quality.py
================
Per-participant SIGNAL-QUALITY control for the region comparison.

Question
--------
Is the per-participant region decoding profile (e.g. the early-dominant S6/S8)
explained by DATA QUALITY rather than perception? The reliability analysis
(reliability.py) shows the profile is a stable trait; this asks whether that
stable trait is just "cleaner data in one region", using NSD's per-voxel
noise-ceiling SNR (ncsnr).

Why ncsnr (and not tSNR/motion)
-------------------------------
The single-trial betas are already motion-corrected, and being betas (not a
timeseries) they do not support a temporal SNR. NSD's ncsnr is the appropriate
per-voxel signal-quality metric: it is the ratio of signal SD to noise SD
estimated from repeated presentations of the same image.

What it does
------------
For each participant it downloads the raw-NSD ncsnr volume and the ROI atlases,
averages ncsnr within each region (early = V1-V3, V4 = hV4, concept = higher
visual streams), and -- if a decoding summary is supplied -- correlates, ACROSS
participants, per-region ncsnr with per-region decoding R^2, and the
early-minus-concept ncsnr difference with the early-minus-concept R^2 difference.

  * strong positive correlation  => the region profile tracks signal quality
                                     (a "clean-scan" effect);
  * near-zero correlation         => the region effect is NOT merely quality.

Caveat
------
ncsnr ROIs are drawn from the raw-NSD prf-visualrois / streams atlases and
APPROXIMATE (do not exactly reproduce) the MindEye2 nsdgeneral region masks used
for decoding. Treat the region correspondence as approximate.

Data: raw NSD, public S3 bucket "natural-scenes-dataset" (needs nibabel; you must
agree to the NSD Terms of Use). ncsnr.nii.gz and the ROI volumes are small.

Usage
-----
    python -m brain2vision.ncsnr_quality --subjects 1 2 3 4 5 6 7 8 \
        --decode-summary roi_color_vw_8subj_summary.npy \
        --out ncsnr_quality.png
"""

import os
import argparse
import numpy as np
import nibabel as nib

from brain2vision.raw_nsd import S3_BASE, _download, build_roi_mask

EARLY = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]     # prf-visualrois
V4 = ["hV4"]                                            # prf-visualrois
CONCEPT_DEFAULT = ["midventral", "midlateral", "midparietal",
                   "ventral", "lateral", "parietal"]    # streams (non-early)
ROIS = ["early_v1v3", "v4_color", "concept"]


def _ncsnr_url(subj):
    return (f"{S3_BASE}/nsddata_betas/ppdata/subj{subj:02d}/func1pt8mm/"
            f"betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz")


def per_subject_ncsnr(subj, concept_streams, cache="nsd_cache"):
    """Mean ncsnr in early / V4 / concept for one participant."""
    ncpath = _download(_ncsnr_url(subj),
                       os.path.join(cache, f"subj{subj:02d}", "ncsnr.nii.gz"))
    nc = nib.load(ncpath).get_fdata()
    masks = {
        "early_v1v3": build_roi_mask(subj, "prf-visualrois", EARLY, cache=cache),
        "v4_color":   build_roi_mask(subj, "prf-visualrois", V4, cache=cache),
        "concept":    build_roi_mask(subj, "streams", concept_streams, cache=cache),
    }
    out = {}
    for name, m in masks.items():
        if m.shape != nc.shape:
            raise ValueError(f"ncsnr {nc.shape} vs {name} mask {m.shape} mismatch "
                             "(orientation?) -- inspect before trusting.")
        out[name] = float(nc[m].mean())
    print(f"subj {subj}: " + "  ".join(f"{k}={v:.3f}" for k, v in out.items()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--concept-streams", nargs="+", default=CONCEPT_DEFAULT,
                   help="streams sub-regions to treat as 'concept' (approx. higher_vis)")
    p.add_argument("--decode-summary", default=None,
                   help="roi_color_vw_8subj_summary.npy to correlate ncsnr against")
    p.add_argument("--cache", default="nsd_cache")
    p.add_argument("--out", default="ncsnr_quality.png")
    args = p.parse_args()

    nc = {r: [] for r in ROIS}
    for s in args.subjects:
        d = per_subject_ncsnr(s, args.concept_streams, cache=args.cache)
        for r in ROIS:
            nc[r].append(d[r])
    nc = {r: np.array(v) for r, v in nc.items()}
    np.save(args.out.replace(".png", "_ncsnr.npy"),
            np.array({"subjects": list(args.subjects), "ncsnr": nc,
                      "concept_streams": args.concept_streams}, dtype=object),
            allow_pickle=True)

    sem = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    print("\n=== mean ncsnr by region (across participants) ===")
    for r in ROIS:
        print(f"  {r:11s} {nc[r].mean():.3f} +/- {sem(nc[r]):.3f}")

    if not args.decode_summary:
        print("\n(no --decode-summary given; skipping the quality-vs-decoding correlation)")
        return

    D = np.load(args.decode_summary, allow_pickle=True).item()
    R2 = {r: np.array(D["agg"][r]["ov"]) for r in ROIS}
    print("\n=== does ncsnr explain decoding? (across-participant correlations) ===")
    for r in ROIS:
        rr = np.corrcoef(nc[r], R2[r])[0, 1]
        print(f"  {r:11s} r(ncsnr, R2) = {rr:+.3f}")
    dnc = nc["early_v1v3"] - nc["concept"]
    dR2 = R2["early_v1v3"] - R2["concept"]
    rr = np.corrcoef(dnc, dR2)[0, 1]
    print(f"\n  early-minus-concept: r(ncsnr-diff, R2-diff) = {rr:+.3f}")
    print("  strong positive => region profile tracks signal quality (clean-scan effect);")
    print("  near zero        => the region effect is NOT merely a quality artefact.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, r in zip(axes, ROIS):
        ax.scatter(nc[r], R2[r], zorder=3)
        ax.set_xlabel(f"{r}  mean ncsnr")
        ax.set_ylabel(f"{r}  decoding R²")
        ax.set_title(f"{r}\nr = {np.corrcoef(nc[r], R2[r])[0, 1]:+.2f}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"\nSaved {args.out} and {args.out.replace('.png', '_ncsnr.npy')}")


if __name__ == "__main__":
    main()
