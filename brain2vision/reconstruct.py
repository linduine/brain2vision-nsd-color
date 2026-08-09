"""
reconstruct.py
==============
Retinotopic reconstruction: recover the coarse spatial layout of a seen image
from early-visual betas, using NSD's population-receptive-field (pRF) solutions.

Why this works where hue decoding didn't
----------------------------------------
Retinotopy is a coarse, super-voxel map: each voxel has a pRF -- a location and
size in the visual field, already fit by the NSD team. Spatial layout survives
1.8 mm voxel averaging (unlike fine-scale colour tuning), which is exactly why
identity/spatial decoding was strong on these stimuli. So a simple linear
back-projection gives a recognisable reconstruction.

Method (back-projection)
------------------------
Each voxel v is a Gaussian bump G_v centred at its pRF location (x_v, y_v) with
sigma = pRF size. For an image's response pattern r (one z-scored beta per
voxel), the reconstruction over the visual field is

    R(gx, gy) = sum_v  r_v * G_v(gx, gy)

i.e. voxels that responded place energy at the part of the visual field they
"look at". This is the standard linear read-out of a retinotopic code; a fitted
encoding-model inverse is the fancier version, but the back-projection is the
honest first result.

Data (raw NSD, S3 bucket "natural-scenes-dataset")
--------------------------------------------------
    pRF params:  nsddata/ppdata/subjXX/func1pt8mm/  ->  prf angle / eccentricity
                 / size volumes (VERIFY exact filenames; pass via --prf-*).
    ROI atlas:   prf-visualrois.nii.gz   (reused from raw_nsd)
    betas:       nsdsyntheticbetas... (via nsd_synthetic loader)
    stimuli:     nsdsynthetic_stimuli.hdf5 (achromatic set)

Conventions to VERIFY on first run
----------------------------------
pRF angle/eccentricity units and the angle zero/direction set the reconstruction
orientation. Defaults assume angle in DEGREES counter-clockwise from the positive
x-axis and eccentricity in DEGREES. If a reconstruction of a known stimulus comes
out rotated/mirrored, flip with --angle-offset / --flip-x / --flip-y. Validate
against an image with obvious structure (a spiral or an oriented grating).

Dependencies: pip install nibabel h5py numpy matplotlib scikit-learn requests
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import nibabel as nib

from brain2vision.raw_nsd import S3_BASE, _download, build_roi_mask, _align_betas_to_mask
from brain2vision.nsd_synthetic import SYNTH_ROI_SETS, load_synth_roi_betas


def _prf_url(subj, name):
    return (f"{S3_BASE}/nsddata/ppdata/subj{subj:02d}/func1pt8mm/{name}")


def load_prf_roi(subj, roi, prf_angle, prf_ecc, prf_size, prf_r2, cache="nsd_cache"):
    """
    Load per-voxel pRF params restricted to `roi`, in the SAME voxel order as
    load_synth_roi_betas (both index the ROI atlas mask). Returns dict with
    angle, ecc, size (deg) and r2 (pRF goodness-of-fit).
    """
    atlas, regions = SYNTH_ROI_SETS[roi]
    mask3d = build_roi_mask(subj, atlas, regions, cache=cache)

    def load_vol(name):
        p = _download(_prf_url(subj, name),
                      os.path.join(cache, f"subj{subj:02d}", name))
        return nib.load(p).get_fdata()

    return {"angle": load_vol(prf_angle)[mask3d],
            "ecc": load_vol(prf_ecc)[mask3d],
            "size": load_vol(prf_size)[mask3d],
            "r2": load_vol(prf_r2)[mask3d]}


def prf_to_xy(prf, angle_offset=0.0, flip_x=False, flip_y=False, radians=False):
    """(angle, eccentricity) -> Cartesian (x, y) in visual-field degrees."""
    a = prf["angle"].astype(float)
    a = a if radians else np.deg2rad(a)
    a = a + np.deg2rad(angle_offset)
    ecc = prf["ecc"].astype(float)
    x = ecc * np.cos(a) * (-1 if flip_x else 1)
    y = ecc * np.sin(a) * (-1 if flip_y else 1)
    sigma = np.maximum(prf["size"].astype(float), 0.1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (ecc > 0)
    return x, y, sigma, valid


def backproject(resp, x, y, sigma, extent, n=128):
    """
    Response-weighted reconstruction over [-extent, extent]^2:

        R(gx,gy) = sum_v resp_v * G_v(gx,gy) / (sum_v G_v(gx,gy) + eps)

    The normalization (dividing by the summed Gaussians) turns the raw
    back-projection into a response-weighted *average* at each location, so
    regions that merely have more/overlapping voxels don't dominate. Returns an
    (n,n) image. `resp` should be a mean-removed / z-scored response per voxel.
    """
    g = np.linspace(-extent, extent, n)
    gx, gy = np.meshgrid(g, g)
    num = np.zeros((n, n)); den = np.zeros((n, n))
    for r, cx, cy, s in zip(resp, x, y, sigma):
        G = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s))
        num += r * G; den += G
    return num / (den + 1e-6)


def forward_matrix(x, y, sigma, extent, n):
    """
    Build the forward encoding matrix W (n_vox x n*n): row v is voxel v's pRF
    Gaussian over the image grid, normalized to unit sum (so W @ s = the
    pRF-weighted average of image s = the predicted voxel response).
    """
    g = np.linspace(-extent, extent, n)
    gx, gy = np.meshgrid(g, g)
    W = np.empty((len(x), n * n))
    for i, (cx, cy, s) in enumerate(zip(x, y, sigma)):
        G = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s)).ravel()
        W[i] = G / (G.sum() + 1e-12)
    return W


def inverse_operator(x, y, sigma, extent, n, lam=0.0):
    """
    Regularized inverse of the forward model: M = W^T (W W^T + lam I)^-1, so
    reconstruction = M @ response. Unlike back-projection (which is ~ W^T r), this
    deconvolves the overlapping pRFs and recovers detail finer than one pRF
    (super-resolution-like), up to the fMRI floor. lam<=0 -> auto (10% of the
    mean diagonal of W W^T).
    """
    W = forward_matrix(x, y, sigma, extent, n)
    A = W @ W.T
    l = lam if lam > 0 else 0.1 * np.trace(A) / A.shape[0]
    A[np.diag_indices_from(A)] += l
    return W.T @ np.linalg.inv(A)                     # (n_pix, n_vox)


def _subject_recons(subj, roi, prf_files, image_ids, extent, n, cache, design,
                    angle_offset, flip_x, flip_y, min_r2_pct,
                    method="backproject", lam=0.0):
    """One subject: return {gid: reconstruction}, the extent used, median pRF size."""
    X, image_of, _ = load_synth_roi_betas(subj, roi, cache=cache, design_path=design)
    prf = load_prf_roi(subj, roi, *prf_files, cache=cache)
    x, y, sigma, valid = prf_to_xy(prf, angle_offset, flip_x, flip_y)
    thr = np.nanpercentile(prf["r2"][valid], min_r2_pct)
    keep = valid & (prf["r2"] >= thr)
    x, y, sigma = x[keep], y[keep], sigma[keep]
    if extent is None:
        extent = float(np.nanpercentile(prf["ecc"][keep], 95))
    med_size = float(np.median(sigma))
    print(f"  subj{subj:02d} {roi}: {int(keep.sum())}/{int(valid.sum())} voxels "
          f"kept (pRF R2 >= {thr:.1f}), median pRF size={med_size:.2f} deg [{method}]")
    if method == "inverse":
        M = inverse_operator(x, y, sigma, extent, n, lam)   # built once per subject
        recs = {gid: (M @ X[image_of == gid].mean(0)[keep]).reshape(n, n)
                for gid in image_ids}
    else:
        recs = {gid: backproject(X[image_of == gid].mean(0)[keep], x, y, sigma, extent, n)
                for gid in image_ids}
    return recs, extent, med_size


def reconstruct(subjects, roi, prf_files, images_h5, image_ids, out,
                cache="nsd_cache", design=None, extent=None, n=128,
                angle_offset=0.0, flip_x=False, flip_y=False, min_r2_pct=50.0,
                method="backproject", lam=0.0):
    """
    Reconstruct each image and save a stimulus-vs-reconstruction figure. With
    multiple subjects, reconstructions are averaged in common visual-field
    coordinates (cancels per-subject noise; won't beat the pRF resolution floor).
    method: "backproject" (simple, blunt) or "inverse" (forward-model deconvolution).
    """
    subjects = list(subjects)
    acc = {gid: np.zeros((n, n)) for gid in image_ids}
    for subj in subjects:                                # first subject fixes extent
        recs, extent, _ = _subject_recons(
            subj, roi, prf_files, image_ids, extent, n, cache, design,
            angle_offset, flip_x, flip_y, min_r2_pct, method, lam)
        for gid in image_ids:
            acc[gid] += recs[gid]
    for gid in image_ids:
        acc[gid] /= len(subjects)

    tag = (f"subj{subjects[0]:02d}" if len(subjects) == 1
           else f"mean of {len(subjects)} subjects")
    with h5py.File(images_h5, "r") as f:
        key = next(k for k in f if isinstance(f[k], h5py.Dataset))
        stim = f[key]; n_img = stim.shape[0]
        fig, axes = plt.subplots(2, len(image_ids), figsize=(3 * len(image_ids), 6))
        axes = np.atleast_2d(axes)
        for j, gid in enumerate(image_ids):
            si = gid - 1
            im = np.asarray(stim[si]) if si < n_img else np.zeros((n, n))
            if im.ndim == 3 and im.shape[0] in (1, 3):
                im = im.transpose(1, 2, 0)
            axes[0, j].imshow(im, cmap="gray")
            axes[0, j].set_title(f"stimulus {gid}", fontsize=9)
            axes[1, j].imshow(acc[gid], cmap="gray", origin="lower",
                              extent=[-extent, extent, -extent, extent])
            axes[1, j].set_title("reconstruction", fontsize=9)
            for a in (axes[0, j], axes[1, j]):
                a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{tag} {roi}: retinotopic reconstruction (pRF back-projection)")
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


def reconstruct_ladder(subjects, rois, prf_files, images_h5, image_ids, out,
                       cache="nsd_cache", design=None, extent=None, n=128,
                       angle_offset=0.0, flip_x=False, flip_y=False, min_r2_pct=50.0,
                       method="backproject", lam=0.0):
    """
    Reconstruct each image from every ROI in `rois` (a hierarchy ladder) and lay
    them out as a grid: rows = images, columns = [stimulus, ROI_1, ROI_2, ...].
    Each ROI column is titled with its median pRF size, so you can watch the
    reconstruction coarsen as pRF size grows up the hierarchy. Multiple subjects
    are averaged in common visual-field coordinates. A single, common `extent`
    (from the first ROI) is used so all panels share the same coordinate scale.
    """
    subjects = list(subjects)
    acc = {(roi, gid): np.zeros((n, n)) for roi in rois for gid in image_ids}
    med_size = {}
    for roi in rois:
        sizes = []
        for subj in subjects:
            recs, extent, ms = _subject_recons(
                subj, roi, prf_files, image_ids, extent, n, cache, design,
                angle_offset, flip_x, flip_y, min_r2_pct, method, lam)
            sizes.append(ms)
            for gid in image_ids:
                acc[(roi, gid)] += recs[gid]
        for gid in image_ids:
            acc[(roi, gid)] /= len(subjects)
        med_size[roi] = float(np.mean(sizes))

    with h5py.File(images_h5, "r") as f:
        key = next(k for k in f if isinstance(f[k], h5py.Dataset))
        stim = f[key]; n_img = stim.shape[0]
        ncol = 1 + len(rois); nrow = len(image_ids)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.4 * nrow),
                                 squeeze=False)
        for i, gid in enumerate(image_ids):
            si = gid - 1
            im = np.asarray(stim[si]) if si < n_img else np.zeros((n, n))
            if im.ndim == 3 and im.shape[0] in (1, 3):
                im = im.transpose(1, 2, 0)
            axes[i, 0].imshow(im, cmap="gray")
            axes[i, 0].set_ylabel(f"image {gid}", fontsize=9)
            if i == 0:
                axes[i, 0].set_title("stimulus", fontsize=10)
            for c, roi in enumerate(rois):
                axes[i, c + 1].imshow(acc[(roi, gid)], cmap="gray", origin="lower",
                                      extent=[-extent, extent, -extent, extent])
                if i == 0:
                    axes[i, c + 1].set_title(f"{roi}\npRF~{med_size[roi]:.1f}°",
                                             fontsize=9)
            for a in axes[i]:
                a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Retinotopic reconstruction up the visual hierarchy "
                 f"(mean of {len(subjects)} subject{'s' if len(subjects)>1 else ''})")
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")
    print("median pRF size by ROI (deg):",
          {r: round(med_size[r], 2) for r in rois})


def main():
    p = argparse.ArgumentParser(description="Retinotopic reconstruction from NSD betas.")
    p.add_argument("--subjects", nargs="+", type=int, default=[1],
                   help="one or more subjects; multiple are averaged in "
                        "visual-field coordinates (cleaner, not sharper)")
    p.add_argument("--roi", default="early_v1v3", choices=list(SYNTH_ROI_SETS))
    p.add_argument("--rois", nargs="+", choices=list(SYNTH_ROI_SETS),
                   help="LADDER mode: reconstruct each image from every listed ROI "
                        "side by side (e.g. V1 V2 V3 V4 ventral lateral).")
    p.add_argument("--prf-angle", default="prf_angle.nii.gz")
    p.add_argument("--prf-ecc", default="prf_eccentricity.nii.gz")
    p.add_argument("--prf-size", default="prf_size.nii.gz")
    p.add_argument("--prf-r2", default="prf_R2.nii.gz")
    p.add_argument("--min-r2-pct", type=float, default=50.0,
                   help="keep voxels above this percentile of pRF R2 "
                        "(50 = best-fit half; raise to 75 for cleaner, sparser)")
    p.add_argument("--images", required=True, help="nsdsynthetic_stimuli.hdf5 (achromatic)")
    p.add_argument("--image-ids", nargs="+", type=int, required=True,
                   help="global image numbers to reconstruct (e.g. spirals/gratings)")
    p.add_argument("--design", default=None, help="nsdsynthetic_expdesign.mat")
    p.add_argument("--extent", type=float, default=None, help="visual-field deg (default: 95th pct ecc)")
    p.add_argument("--n", type=int, default=128, help="reconstruction grid size")
    p.add_argument("--method", choices=["backproject", "inverse"], default="backproject",
                   help="'backproject' = simple pRF-weighted sum (blunt, blurry); "
                        "'inverse' = regularized forward-model deconvolution (sharper, "
                        "exploits overlapping pRFs)")
    p.add_argument("--lam", type=float, default=0.0,
                   help="inverse regularization (0 = auto; raise for smoother/less "
                        "noisy, lower for sharper/riskier)")
    p.add_argument("--angle-offset", type=float, default=0.0)
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--cache", default="nsd_cache")
    p.add_argument("--out", default="reconstruction.png")
    args = p.parse_args()
    prf_files = (args.prf_angle, args.prf_ecc, args.prf_size, args.prf_r2)
    common = dict(cache=args.cache, design=args.design, extent=args.extent,
                  n=args.n, angle_offset=args.angle_offset, flip_x=args.flip_x,
                  flip_y=args.flip_y, min_r2_pct=args.min_r2_pct,
                  method=args.method, lam=args.lam)
    if args.rois:
        reconstruct_ladder(args.subjects, args.rois, prf_files,
                           args.images, args.image_ids, args.out, **common)
    else:
        reconstruct(args.subjects, args.roi, prf_files,
                    args.images, args.image_ids, args.out, **common)


if __name__ == "__main__":
    main()
