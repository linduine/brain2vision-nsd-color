"""
raw_nsd_roi_loader.py
=====================
FUTURE / ADVANCED path. Not needed for the higher_vis-based first experiment.

Extract NSD single-trial betas for FINE-GRAINED ROIs that the preprocessed
MindEye2 release does not expose -- e.g. individual category-selective regions
(FFA, PPA, EBA, VWFA, ...) or individual retinotopic areas (V1v, V1d, hV4, ...).

It works directly on the RAW NSD data:
  * ROI label volumes:  nsddata/ppdata/subjXX/func1pt8mm/roi/<atlas>.nii.gz
  * volumetric betas:   nsddata_betas/ppdata/subjXX/func1pt8mm/
                        betas_fithrf_GLMdenoise_RR/betas_sessionNN.hdf5

Data source
-----------
Public S3 bucket "natural-scenes-dataset" (AWS Open Data). No credentials
needed for read, but you must agree to the NSD Terms of Use:
    https://naturalscenesdataset.org/  (Manual + terms)
Do NOT redistribute the downloaded betas/masks in your GitHub repo.

Heads-up on size
----------------
NSD offers no server-side voxel slicing, so to get an ROI you download the FULL
volumetric betas for a session (~1 GB / session, 30-40 sessions per subject),
mask them, and discard the rest. This loader streams one session at a time so
peak disk stays ~1 session, but total transfer is large. This is the cost of
arbitrary ROIs; the preprocessed higher_vis path avoids it.

Dependencies
------------
    pip install nibabel h5py numpy requests
"""

import os
import argparse
import numpy as np
import h5py
import nibabel as nib
import requests

# ---------------------------------------------------------------------------
# S3 (HTTPS) layout
# ---------------------------------------------------------------------------
S3_BASE = "https://natural-scenes-dataset.s3.amazonaws.com"

# Sessions completed per subject (subj 1,2,5,7 finished all 40).
N_SESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}

# ---------------------------------------------------------------------------
# Integer label -> sub-region name, per ROI atlas volume.
# Source: NSD data manual. VERIFY against the matching .ctab files in
# nsddata/freesurfer/subjXX/label/ if you need to be certain for a subject.
# ---------------------------------------------------------------------------
REGION_LABELS = {
    "prf-visualrois": {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d",
                       5: "V3v", 6: "V3d", 7: "hV4"},
    "floc-faces":     {1: "OFA", 2: "FFA-1", 3: "FFA-2",
                       4: "mTL-faces", 5: "aTL-faces"},
    "floc-places":    {1: "OPA", 2: "PPA", 3: "RSC"},
    "floc-bodies":    {1: "EBA", 2: "FBA-1", 3: "FBA-2", 4: "mTL-bodies"},
    "floc-words":     {1: "OWFA", 2: "VWFA-1", 3: "VWFA-2",
                       4: "mfs-words", 5: "mTL-words"},
    "streams":        {1: "early", 2: "midventral", 3: "midlateral",
                       4: "midparietal", 5: "ventral", 6: "lateral",
                       7: "parietal"},
}

# NSD betas are stored as int16; divide by this to get % signal change.
BETA_SCALE = 300.0


def _roi_url(subj, atlas):
    return (f"{S3_BASE}/nsddata/ppdata/subj{subj:02d}/func1pt8mm/"
            f"roi/{atlas}.nii.gz")


def _betas_url(subj, session):
    return (f"{S3_BASE}/nsddata_betas/ppdata/subj{subj:02d}/func1pt8mm/"
            f"betas_fithrf_GLMdenoise_RR/betas_session{session:02d}.hdf5")


def _download(url, dest):
    """Stream a URL to a local file (skips if already present)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        os.replace(tmp, dest)
    return dest


def build_roi_mask(subj, atlas, regions, cache="nsd_cache"):
    """
    Return a boolean 3D volume selecting the requested sub-regions of an atlas.

    atlas   : one of REGION_LABELS keys, e.g. "floc-faces".
    regions : list of sub-region NAMES (e.g. ["FFA-1","FFA-2"]) or ["*"] for all
              labeled voxels in the atlas.
    """
    if atlas not in REGION_LABELS:
        raise KeyError(f"Unknown atlas '{atlas}'. Options: {list(REGION_LABELS)}")
    label_map = REGION_LABELS[atlas]
    name_to_id = {v.lower(): k for k, v in label_map.items()}

    if regions == ["*"]:
        wanted_ids = set(label_map.keys())
    else:
        wanted_ids = set()
        for r in regions:
            key = r.lower()
            if key not in name_to_id:
                raise ValueError(
                    f"'{r}' not in {atlas}. Valid: {list(label_map.values())}")
            wanted_ids.add(name_to_id[key])

    roi_path = _download(_roi_url(subj, atlas),
                         os.path.join(cache, f"subj{subj:02d}", f"{atlas}.nii.gz"))
    vol = nib.load(roi_path).get_fdata()
    mask3d = np.isin(np.rint(vol).astype(int), list(wanted_ids))
    print(f"  atlas={atlas} regions={regions} -> {int(mask3d.sum())} voxels")
    return mask3d


def _align_betas_to_mask(betas, mask3d):
    """
    NSD betas hdf5 store space with axes in reverse order relative to the NIfTI.
    Reorder the betas' spatial axes so they match mask3d.shape, keeping the
    trial axis first. Returns betas as (n_trials, *mask3d.shape).
    """
    n_trials = betas.shape[0]
    spatial = betas.shape[1:]
    target = mask3d.shape
    if spatial == target:
        return betas
    if tuple(reversed(spatial)) == target:
        return betas[:, ::, ...].transpose(0, 3, 2, 1)
    # fall back: find a permutation of the 3 spatial axes matching the mask
    from itertools import permutations
    for perm in permutations((1, 2, 3)):
        if tuple(betas.shape[a] for a in perm) == target:
            return betas.transpose((0,) + perm)
    raise ValueError(
        f"Cannot align betas spatial dims {spatial} to mask {target}. "
        f"Inspect orientation manually.")


def load_raw_roi_betas(subj, atlas, regions, sessions=None,
                       cache="nsd_cache", zscore=True):
    """
    Download volumetric betas session-by-session, apply the ROI mask, and
    concatenate across sessions.

    Returns betas_roi of shape (total_trials, n_roi_voxels), float32.
    Trials are in NSD session/trial order; pair them with images via
    nsd_expdesign.mat (masterordering / subjectim) separately.
    """
    mask3d = build_roi_mask(subj, atlas, regions, cache=cache)
    if sessions is None:
        sessions = range(1, N_SESSIONS[subj] + 1)

    chunks = []
    for s in sessions:
        bpath = _download(_betas_url(subj, s),
                          os.path.join(cache, f"subj{subj:02d}",
                                       f"betas_session{s:02d}.hdf5"))
        with h5py.File(bpath, "r") as f:
            betas = f["betas"][()]  # int16, (n_trials, *spatial)
        betas = _align_betas_to_mask(betas, mask3d)
        roi = betas[:, mask3d].astype(np.float32) / BETA_SCALE  # % signal change
        if zscore:
            roi = (roi - roi.mean(0, keepdims=True)) / (roi.std(0, keepdims=True) + 1e-6)
        chunks.append(roi)
        print(f"  session {s:02d}: {roi.shape}")
        del betas

    out = np.concatenate(chunks, axis=0)
    print(f"Loaded raw ROI betas: {out.shape}")
    return out


def main():
    p = argparse.ArgumentParser(
        description="Extract raw-NSD betas for specific fine-grained ROIs.")
    p.add_argument("--subj", type=int, default=1)
    p.add_argument("--atlas", required=True, choices=list(REGION_LABELS),
                   help="ROI atlas volume")
    p.add_argument("--regions", nargs="+", required=True,
                   help='Sub-region names, or "*" for all, e.g. FFA-1 FFA-2')
    p.add_argument("--sessions", nargs="+", type=int,
                   help="Session numbers to load (default: all for subject)")
    p.add_argument("--no-zscore", action="store_true")
    p.add_argument("--cache", default="nsd_cache")
    args = p.parse_args()

    betas = load_raw_roi_betas(
        args.subj, args.atlas, args.regions,
        sessions=args.sessions, cache=args.cache, zscore=not args.no_zscore)
    print("Final shape:", betas.shape)


if __name__ == "__main__":
    main()
