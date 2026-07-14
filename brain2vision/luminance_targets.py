"""
luminance_targets.py
====================
Build a per-image LUMINANCE (brightness) distribution target: an n-bin
histogram of pixel brightness from dark (bin 0) to bright (bin n-1). This is the
achromatic counterpart to color_targets.py, used to test whether early visual
cortex — which looked luminance-driven in the color analysis (best at black and
white) — actually decodes image brightness better than V4 / higher visual.

Brightness uses the standard luma weighting: 0.299 R + 0.587 G + 0.114 B.

Usage
-----
    python -m brain2vision.luminance_targets \
        --images data/coco_images_224_float16.hdf5 \
        --out data/luminance_targets.npy --bins 11
"""

import argparse
import numpy as np

BIN_LABELS = None  # set at runtime as ["L0", "L1", ...] for --bins bins


def image_luminance_hist(arr, bins=11):
    """arr: (H,W,3) uint8/float -> length-`bins` brightness distribution."""
    a = np.asarray(arr).astype(np.float32)
    if a.max() > 1.001:
        a = a / 255.0
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    h, _ = np.histogram(lum, bins=bins, range=(0.0, 1.0))
    return (h / h.sum()).astype(np.float32)


def run(h5path, out, bins=11, nsd_ids=None, batch=256):
    import h5py
    with h5py.File(h5path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        dset = f[key]
        n = dset.shape[0]
        ids = list(range(n)) if nsd_ids is None else sorted(nsd_ids)
        out_vecs = np.zeros((len(ids), bins), dtype=np.float32)
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            arr = np.asarray(dset[chunk])
            if arr.ndim == 4 and arr.shape[1] == 3:
                arr = arr.transpose(0, 2, 3, 1)
            for j, im in enumerate(arr):
                out_vecs[i + j] = image_luminance_hist(im, bins=bins)
            print(f"  processed {min(i + batch, len(ids))}/{len(ids)}")
    np.save(out, out_vecs)
    np.save(out.replace(".npy", "_ids.npy"), np.asarray(ids))
    print(f"Saved {out} shape={out_vecs.shape} ({bins} brightness bins)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="data/luminance_targets.npy")
    p.add_argument("--bins", type=int, default=11)
    p.add_argument("--nsd-ids", nargs="+", type=int)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    run(args.images, args.out, bins=args.bins,
        nsd_ids=args.nsd_ids, batch=args.batch)


if __name__ == "__main__":
    main()
