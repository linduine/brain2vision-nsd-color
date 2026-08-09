"""
nsd_synthetic.py
================
Load NSD-SYNTHETIC single-trial betas restricted to our three visual pools
(early V1-V3, V4, higher "concept" cortex), for the colour-without-semantics
control experiment.

Why this experiment
-------------------
On NSD-core (natural images) the concept pool decoded image colour best, but
residualising colour against object presence collapsed that advantage across
*all* ROIs -- leaving it ambiguous whether the concept area has a genuine
chromatic code or was reading object/scene identity (and attended-foreground
colour). NSD-synthetic breaks the tie: it contains 64 *isoluminant chromatic
pink-noise* images spanning 16 hues (Hue01-Hue16), with no objects, no figure,
and no luminance confound. If the concept pool still decodes hue here, its
colour code is genuinely chromatic; if early visual cortex wins, the natural-
image advantage was structure/semantics-bound.

(The NSD-synthetic data paper, Nat. Commun. 2026, reports that chromatic noise
activates early visual cortex more than higher visual cortex -- our decoding
test is the stricter, voxel-matched complement to that activation result.)

Data source (RAW NSD, public S3 bucket "natural-scenes-dataset")
----------------------------------------------------------------
  * ROI label volumes:  nsddata/ppdata/subjXX/func1pt8mm/roi/<atlas>.nii.gz
  * synthetic betas:    nsddata_betas/ppdata/subjXX/func1pt8mm/
                        nsdsyntheticbetas_fithrf_GLMdenoise_RR/
                        betas_nsdsynthetic.hdf5              (single file/subject)
Agree to the NSD Terms of Use first (https://naturalscenesdataset.org/) and do
NOT redistribute downloaded betas/masks.

VERIFY-BEFORE-RUN note
----------------------
The exact betas *filename* inside nsdsyntheticbetas_fithrf_GLMdenoise_RR and the
condition ordering are set by the NSD manual. This loader prints the loaded
shape and lets you override the filename (--betas-name) and the condition->image
mapping (see condition_to_image). Sanity-check n_conditions against the manual
on first run (expected: 284 images, each in 2 tasks -> 568 conditions, unless
the release stores single trials).

Dependencies
------------
    pip install nibabel h5py numpy requests
"""

import os
import argparse
import numpy as np
import h5py

from brain2vision.raw_nsd import (
    S3_BASE, N_SESSIONS, BETA_SCALE, _download, build_roi_mask,
    _align_betas_to_mask,
)

# ---------------------------------------------------------------------------
# Our three pools, expressed as (atlas, [sub-region names]) in raw-NSD atlases.
# early/V4 come from the retinotopic atlas; "concept" from the streams atlas
# (all high-level streams, i.e. everything past "early").
# ---------------------------------------------------------------------------
SYNTH_ROI_SETS = {
    "early_v1v3": ("prf-visualrois", ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]),
    "v4_color":   ("prf-visualrois", ["hV4"]),
    "concept":    ("streams", ["midventral", "midlateral", "midparietal",
                               "ventral", "lateral", "parietal"]),
    # individual rungs of the visual hierarchy, for the reconstruction "ladder"
    # (retinotopic V1->V4, then higher ventral/lateral/parietal streams). pRF
    # size grows along this progression, so reconstructions coarsen.
    "V1":       ("prf-visualrois", ["V1v", "V1d"]),
    "V2":       ("prf-visualrois", ["V2v", "V2d"]),
    "V3":       ("prf-visualrois", ["V3v", "V3d"]),
    "V4":       ("prf-visualrois", ["hV4"]),
    "ventral":  ("streams", ["ventral"]),
    "lateral":  ("streams", ["lateral"]),
    "parietal": ("streams", ["parietal"]),
}

# Number of distinct synthetic images shown to every subject.
N_SYNTH_IMAGES = 284


def _synth_betas_url(subj, betas_name="betas_nsdsynthetic.hdf5"):
    return (f"{S3_BASE}/nsddata_betas/ppdata/subj{subj:02d}/func1pt8mm/"
            f"nsdsyntheticbetas_fithrf_GLMdenoise_RR/{betas_name}")


def _pick_ordering(arrays, n_rows, n_images=N_SYNTH_IMAGES):
    """
    From a dict {name: ndarray} (e.g. a loaded .mat), pick the trial->image
    ordering: the 1-D integer array of length `n_rows` whose values all fall in
    [1, n_images]. Returns (name, ordering_int_array) or raises with a summary.

    Kept pure (takes plain arrays) so it can be unit-tested without scipy/h5py.
    """
    best = None
    for name, a in arrays.items():
        if name.startswith("__"):
            continue
        v = np.asarray(a).squeeze()
        if v.ndim != 1 or v.size != n_rows:
            continue
        if not np.issubdtype(v.dtype, np.number):
            continue
        vi = np.rint(v).astype(np.int64)
        if np.all(np.abs(v - vi) < 1e-6) and vi.min() >= 1 and vi.max() <= n_images:
            # prefer a field literally named like an ordering
            score = (2 if "order" in name.lower() else
                     1 if "stim" in name.lower() or "cond" in name.lower() else 0)
            if best is None or score > best[0]:
                best = (score, name, vi)
    if best is None:
        summary = ", ".join(f"{k}{np.asarray(v).shape}" for k, v in arrays.items()
                            if not k.startswith("__"))
        raise ValueError(
            f"No trial->image ordering (len {n_rows}, values 1..{n_images}) found "
            f"in design file. Available fields: {summary}")
    return best[1], best[2]


def _load_ordering(design_path, n_rows, n_images=N_SYNTH_IMAGES):
    """Load a .mat design file (v7 via scipy, v7.3 via h5py) and pick the ordering."""
    try:
        from scipy.io import loadmat
        arrays = loadmat(design_path)
    except NotImplementedError:               # MATLAB v7.3 == HDF5
        arrays = {}
        with h5py.File(design_path, "r") as f:
            f.visititems(lambda n, o: arrays.__setitem__(
                n, o[()]) if isinstance(o, h5py.Dataset) else None)
    name, order = _pick_ordering(arrays, n_rows, n_images)
    print(f"  design: using field '{name}' as trial->image ordering "
          f"({order.min()}..{order.max()}, {len(order)} trials)")
    return order


def condition_to_image(n_conditions, n_images=N_SYNTH_IMAGES, ordering=None):
    """
    Map each beta row -> 1-based synthetic image number.

    * ordering given  -> use it directly (the single-trial case: the NSD-synthetic
      betas hold one beta per stimulus trial, 744/subject, in presentation order,
      so the trial->image map must come from the design file's ordering).
    * n_conditions == n_images      -> identity (condition-averaged file).
    * n_conditions == 2 * n_images  -> two stacked task blocks (fixation/one-back).

    Returns (image_of_row, task_of_row), int arrays of length n_conditions.
    """
    if ordering is not None:
        ordering = np.asarray(ordering).astype(int)
        if len(ordering) != n_conditions:
            raise ValueError(
                f"ordering length {len(ordering)} != n betas rows {n_conditions}")
        return ordering, np.zeros(n_conditions, dtype=int)
    if n_conditions == n_images:
        return np.arange(1, n_images + 1), np.zeros(n_conditions, dtype=int)
    if n_conditions == 2 * n_images:
        img = np.concatenate([np.arange(1, n_images + 1)] * 2)
        task = np.concatenate([np.zeros(n_images, int), np.ones(n_images, int)])
        return img, task
    raise ValueError(
        f"n_conditions={n_conditions} is not {n_images} or {2*n_images}; this is "
        f"the single-trial layout -> pass the design file via design_path/--design.")


def load_synth_roi_betas(subj, roi, cache="nsd_cache", zscore=True,
                         betas_name="betas_nsdsynthetic.hdf5", design_path=None):
    """
    Download the NSD-synthetic betas for `subj`, mask to one pool, and return.

    roi         : a key of SYNTH_ROI_SETS ("early_v1v3" | "v4_color" | "concept").
    design_path : path to the NSD-synthetic design .mat (needed for the
                  single-trial betas layout, 744 rows/subject, to map each trial
                  to its image). Not needed if the file is condition-averaged.

    Returns
    -------
    X          : (n_rows, n_roi_voxels) float32, optionally z-scored.
    image_of   : (n_rows,) 1-based synthetic image number per row.
    task_of    : (n_rows,) 0/1 task label per row.
    """
    if roi not in SYNTH_ROI_SETS:
        raise KeyError(f"roi '{roi}' not in {list(SYNTH_ROI_SETS)}")
    atlas, regions = SYNTH_ROI_SETS[roi]
    mask3d = build_roi_mask(subj, atlas, regions, cache=cache)

    bpath = _download(_synth_betas_url(subj, betas_name),
                      os.path.join(cache, f"subj{subj:02d}", betas_name))
    with h5py.File(bpath, "r") as f:
        betas = f["betas"][()]                      # int16, (n_rows, *spatial) or reversed
    betas = _align_betas_to_mask(betas, mask3d)
    X = betas[:, mask3d].astype(np.float32) / BETA_SCALE
    del betas
    if zscore:
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    ordering = _load_ordering(design_path, X.shape[0]) if design_path else None
    image_of, task_of = condition_to_image(X.shape[0], ordering=ordering)
    print(f"  subj{subj:02d} {roi}: X={X.shape}, "
          f"{X.shape[0]} rows -> images 1..{int(image_of.max())}")
    return X, image_of, task_of


def main():
    p = argparse.ArgumentParser(
        description="Load NSD-synthetic ROI betas (colour-without-semantics control).")
    p.add_argument("--subj", type=int, default=1)
    p.add_argument("--roi", default="v4_color", choices=list(SYNTH_ROI_SETS))
    p.add_argument("--betas-name", default="betas_nsdsynthetic.hdf5")
    p.add_argument("--design", default=None,
                   help="NSD-synthetic design .mat (trial->image ordering)")
    p.add_argument("--no-zscore", action="store_true")
    p.add_argument("--cache", default="nsd_cache")
    args = p.parse_args()
    X, img, task = load_synth_roi_betas(
        args.subj, args.roi, cache=args.cache, design_path=args.design,
        zscore=not args.no_zscore, betas_name=args.betas_name)
    print("X:", X.shape, "| unique images:", len(np.unique(img)),
          "| tasks:", np.unique(task))


if __name__ == "__main__":
    main()
