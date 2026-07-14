"""
download_data.py
================
Fetch the data this repo needs into ./data. By running this you confirm you
have read and accepted the NSD Terms of Use (https://naturalscenesdataset.org/)
and the COCO terms (https://cocodataset.org/). See DATA_TERMS.md.

Small shared files (a few MB) are always fetched. Per-subject betas (~1.5 GB
each) are fetched only for the subjects you request.

Usage
-----
    pip install -e .
    python download_data.py --subjects 1            # smoke test: one subject
    python download_data.py --subjects 1 2 5 7       # the 40-session subjects
    python download_data.py --small-only             # just the small files
"""

import os
import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "pscotti/mindeyev2"
S3_BASE = "https://natural-scenes-dataset.s3.amazonaws.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SMALL_HF_FILES = [
    "brain_region_masks.hdf5",   # ROI masks over nsdgeneral
    "shared1000.npy",            # held-out test image ids
]


def _link_into_data(path):
    """Symlink a hf-cached file into ./data for convenience (best effort)."""
    dst = os.path.join(DATA_DIR, os.path.basename(path))
    try:
        if not os.path.exists(dst):
            os.symlink(path, dst)
    except OSError:
        pass
    return dst


def fetch_small():
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in SMALL_HF_FILES:
        p = hf_hub_download(REPO_ID, f, repo_type="dataset")
        _link_into_data(p)
        print(f"  {f} -> {p}")
    # NSD image-info CSV (from NSD S3) for bounding boxes
    import requests
    csv_dst = os.path.join(DATA_DIR, "nsd_stim_info_merged.csv")
    if not os.path.exists(csv_dst):
        url = f"{S3_BASE}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
        print(f"  downloading {url}")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(csv_dst, "wb") as fh:
                for c in r.iter_content(1 << 20):
                    fh.write(c)
    print(f"  nsd_stim_info_merged.csv -> {csv_dst}")


def fetch_subject_betas(subj):
    fn = f"betas_all_subj{subj:02d}_fp32_renorm.hdf5"
    print(f"Downloading {fn} (~1.5 GB)...")
    p = hf_hub_download(REPO_ID, fn, repo_type="dataset")
    _link_into_data(p)
    print(f"  {fn} -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=[],
                    help="Subjects whose betas to download (e.g. 1 2 5 7)")
    ap.add_argument("--images", action="store_true",
                    help="Also fetch coco_images_224_float16.hdf5 (~22 GB)")
    ap.add_argument("--small-only", action="store_true")
    args = ap.parse_args()

    print("By running this you accept the NSD and COCO data terms "
          "(see DATA_TERMS.md).")
    fetch_small()
    if args.small_only:
        return
    for s in args.subjects:
        fetch_subject_betas(s)
    if args.images:
        print("Downloading coco_images_224_float16.hdf5 (~22 GB)...")
        p = hf_hub_download(REPO_ID, "coco_images_224_float16.hdf5",
                            repo_type="dataset")
        _link_into_data(p)
        print(f"  images -> {p}")

    print("\nDone. Next: build targets, then run an experiment (see README).")


if __name__ == "__main__":
    main()
