"""
nsd_roi_loader.py
-----------------
Load NSD single-trial betas for ONE subject, restricted to specific ROI(s),
from the MindEye2 Hugging Face release (pscotti/mindeyev2).

Why this design
---------------
The preprocessed betas are stored as one flattened "nsdgeneral" array per
subject: betas_all_subjXX_fp32_renorm.hdf5, shape (n_trials, n_nsdgeneral_vox).
brain_region_masks.hdf5 gives boolean masks over that same voxel axis for
sub-regions (V1, V2, V3, V4, higher visual, streams, etc.). To use a specific
ROI you take the True columns of the chosen mask:  betas[:, roi_mask].

IMPORTANT download tradeoff
---------------------------
This release only ships the FULL nsdgeneral betas per subject (~1.2-1.9 GB
each). You must download that one file, then subselect the ROI in memory. You
do NOT download all subjects, and you only keep/train on the ROI voxels -- but
the on-disk file is nsdgeneral-sized.

If you want to download ONLY an ROI's voxels (never the full nsdgeneral blob),
that is not possible from this preprocessed repo; you would need the raw NSD
volumetric betas (AWS S3 bucket "natural-scenes-dataset" / OpenNeuro ds004496)
plus the ROI NIfTI masks in nsddata/ppdata/subjXX/func1pt8mm/roi/, and mask the
volume yourself. Heavier download, full flexibility. See notes in README.

Usage
-----
    python nsd_roi_loader.py --subj 1 --rois V1 V2 --preview

Run inspect_roi_masks.py first to see the exact region names available.
"""

import argparse
import numpy as np
import h5py
from huggingface_hub import hf_hub_download

REPO_ID = "pscotti/mindeyev2"
MASK_FILE = "brain_region_masks.hdf5"

# ---------------------------------------------------------------------------
# Named ROI sets. Three functional groupings, loaded independently.
#
# These are the EXACT region names in brain_region_masks.hdf5 (verified via
# inspect_roi_masks.py). Per subject the file stores a clean partition:
#     V1, V2, V3, V4          individual early/intermediate areas
#     early_vis               = union of V1..V4
#     higher_vis              = everything in nsd_general beyond early_vis
#     nsd_general             = early_vis + higher_vis (the full ROI)
#
# NOTE: this file does NOT contain separate floc-faces/places/bodies/words
# category ROIs. The only concept-level mask available here is `higher_vis`.
# To split faces vs places vs bodies vs words you must use the raw-NSD
# floc-*.nii.gz masks (see README).
# ---------------------------------------------------------------------------
ROI_SETS = {
    # Early visual cortex: low-level features, retinotopic.  (~3970 vox, subj01)
    "early_v1v3": ["V1", "V2", "V3"],
    # V4: color / intermediate form.                          (~687 vox, subj01)
    "v4_color": ["V4"],
    # Concept-level: all higher visual cortex beyond V1-V4.  (~11067 vox, subj01)
    "concept": ["higher_vis"],
}


def _betas_filename(subj: int) -> str:
    return f"betas_all_subj{subj:02d}_fp32_renorm.hdf5"


def _first_dataset_key(h5file) -> str:
    """Return the name of the first top-level Dataset in an open h5py.File."""
    for k in h5file.keys():
        if isinstance(h5file[k], h5py.Dataset):
            return k
    raise ValueError("No top-level dataset found in betas file.")


def load_roi_masks(subj: int, rois, mask_path=None, match="exact"):
    """
    Return a boolean array (length = n_nsdgeneral_voxels) that is the UNION of
    the requested ROI masks for the given subject.

    `rois`  : list of region-name fragments (or a named-set key handled by
              load_roi_set).
    `match` : "exact" (default) -> leaf must equal a fragment; "substring" ->
              a dataset matches if any fragment is a substring of its leaf name.
    Region lookup uses the tail of the dataset path, so both "subj01/V1" and
    flat "V1" layouts work.
    """
    if mask_path is None:
        mask_path = hf_hub_download(REPO_ID, MASK_FILE, repo_type="dataset")

    subj_key = f"subj{subj:02d}"
    wanted = [r.lower() for r in rois]
    combined = None
    found = []

    def leaf_matches(leaf):
        if match == "exact":
            return leaf in wanted
        return any(frag in leaf for frag in wanted)

    with h5py.File(mask_path, "r") as f:
        def visit(name, obj):
            nonlocal combined
            if not isinstance(obj, h5py.Dataset):
                return
            parts = name.split("/")
            leaf = parts[-1].lower()
            if not leaf_matches(leaf):
                return
            # if the path is subject-scoped, it must match this subject.
            if len(parts) > 1 and parts[0].lower().startswith("subj") \
                    and parts[0].lower() != subj_key:
                return
            m = np.asarray(obj[()]).astype(bool).ravel()
            combined = m if combined is None else (combined | m)
            found.append(name)

        f.visititems(visit)

    if combined is None:
        raise ValueError(
            f"None of {rois} found for {subj_key} (match={match}). "
            f"Run inspect_roi_masks.py to list valid region names."
        )
    print(f"Matched region datasets: {found}")
    print(f"ROI union selects {int(combined.sum())} / {combined.size} nsdgeneral voxels")
    return combined


def load_roi_set(subj: int, set_name: str, betas_path=None, mask_path=None,
                 match="exact"):
    """
    Load betas for one of the named ROI_SETS ("early_v1v3", "v4_color",
    "concept"). Thin wrapper over load_roi_betas.
    """
    if set_name not in ROI_SETS:
        raise KeyError(f"Unknown set '{set_name}'. Options: {list(ROI_SETS)}")
    print(f"\n=== ROI set '{set_name}' -> fragments {ROI_SETS[set_name]} ===")
    return load_roi_betas(subj, ROI_SETS[set_name],
                          betas_path=betas_path, mask_path=mask_path,
                          match=match)


def load_roi_betas(subj: int, rois, betas_path=None, mask_path=None,
                   match="exact"):
    """
    Download (if needed) subject betas, subselect ROI voxels, return
    (betas_roi, roi_mask).  betas_roi shape = (n_trials, n_roi_voxels), float32.
    """
    roi_mask = load_roi_masks(subj, rois, mask_path=mask_path, match=match)

    if betas_path is None:
        print(f"Downloading {_betas_filename(subj)} (this is the large file)...")
        betas_path = hf_hub_download(
            REPO_ID, _betas_filename(subj), repo_type="dataset"
        )

    with h5py.File(betas_path, "r") as f:
        key = _first_dataset_key(f)
        dset = f[key]
        print(f"betas dataset '{key}' shape={dset.shape} dtype={dset.dtype}")
        n_vox = dset.shape[1]
        if n_vox != roi_mask.size:
            raise ValueError(
                f"Voxel-count mismatch: betas have {n_vox} voxels but ROI mask "
                f"is length {roi_mask.size}. Masks and betas are out of sync."
            )
        # h5py fancy-indexing with a boolean mask reads only the wanted columns.
        betas_roi = dset[:, roi_mask].astype(np.float32)

    print(f"Loaded ROI betas: {betas_roi.shape}  "
          f"(mean={betas_roi.mean():.3f}, std={betas_roi.std():.3f})")
    return betas_roi, roi_mask


def main():
    p = argparse.ArgumentParser(description="Load NSD betas for specific ROIs.")
    p.add_argument("--subj", type=int, default=1, help="Subject number 1-8")
    p.add_argument("--rois", nargs="+",
                   help="Raw region-name fragments, e.g. V1 V2")
    p.add_argument("--set", dest="set_name", choices=list(ROI_SETS),
                   help="Named ROI set to load")
    p.add_argument("--all-sets", action="store_true",
                   help="Load all three named sets separately for this subject")
    p.add_argument("--match", choices=["substring", "exact"], default="exact")
    p.add_argument("--preview", action="store_true",
                   help="Print a small slice of the loaded betas")
    args = p.parse_args()

    results = {}
    if args.all_sets:
        # Download the subject betas once, reuse across all three sets.
        betas_path = hf_hub_download(
            REPO_ID, _betas_filename(args.subj), repo_type="dataset")
        for name in ROI_SETS:
            b, m = load_roi_set(args.subj, name, betas_path=betas_path,
                                match=args.match)
            results[name] = b
    elif args.set_name:
        results[args.set_name] = load_roi_set(
            args.subj, args.set_name, match=args.match)[0]
    elif args.rois:
        results["custom"] = load_roi_betas(
            args.subj, args.rois, match=args.match)[0]
    else:
        p.error("Provide one of --rois, --set, or --all-sets")

    print("\n=== Summary ===")
    for name, b in results.items():
        print(f"{name:12s} betas shape={b.shape}")
        if args.preview:
            print(f"  preview {name} [:2,:6]:\n{b[:2, :6]}")


if __name__ == "__main__":
    main()
