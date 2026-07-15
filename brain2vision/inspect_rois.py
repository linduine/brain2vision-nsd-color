"""
inspect_roi_masks.py
--------------------
Download the small brain_region_masks.hdf5 (~0.8 MB) from the MindEye2 NSD
release and print every ROI/region available per subject, with voxel counts.

Run this ONCE before using roi.py so you know which region names to
request. Nothing large is downloaded here.

The betas in this release are stored per subject as a single FLATTENED array of
"nsdgeneral" voxels (~13k-16k per subject). brain_region_masks.hdf5 contains,
for each subject, boolean masks over that same flattened voxel axis. Selecting
an ROI = picking the True columns of one of these masks. That is how you use a
sub-ROI without training on the whole nsdgeneral region.
"""

from huggingface_hub import hf_hub_download
import h5py
import numpy as np

REPO_ID = "pscotti/mindeyev2"
MASK_FILE = "brain_region_masks.hdf5"


def main():
    path = hf_hub_download(repo_id=REPO_ID, filename=MASK_FILE, repo_type="dataset")
    print(f"Downloaded: {path}\n")

    with h5py.File(path, "r") as f:
        # The file is a small tree. We walk it and report every dataset,
        # its shape, dtype, and (if boolean-ish) how many voxels it selects.
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                arr = obj[()]
                n_true = ""
                try:
                    a = np.asarray(arr)
                    if a.dtype == bool or set(np.unique(a)).issubset({0, 1}):
                        n_true = f"  -> selects {int(a.sum())} voxels"
                except Exception:
                    pass
                print(f"{name:45s} shape={obj.shape} dtype={obj.dtype}{n_true}")

        print("=== Contents of brain_region_masks.hdf5 ===")
        f.visititems(walk)
        print("\nTop-level keys:", list(f.keys()))


if __name__ == "__main__":
    main()
