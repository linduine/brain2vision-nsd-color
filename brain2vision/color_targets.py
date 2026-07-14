"""
extract_color_targets.py
========================
Build the COLOR target vectors for the "which colors is the subject seeing?"
experiment. For each NSD image we compute an interpretable 11-way basic-color
distribution: the fraction of pixels that fall into each of the 11 English
basic color categories (Berlin & Kay):

    red, orange, yellow, green, blue, purple, pink, brown, black, white, gray

Each image -> an 11-dim vector that sums to 1 (a soft label over colors). This
is what the V4 decoder predicts. It's more informative than a single dominant
color and stays fully interpretable ("30% green, 20% blue, ...").

Method
------
Pixels are converted RGB->HSV (vectorized, numpy only) and assigned to a color
bin by simple, transparent rules on hue/saturation/value. Approximate but good
enough for a first experiment; swap in a Lab nearest-reference scheme later if
you want perceptual accuracy.

Usage
-----
    pip install h5py numpy
    python extract_color_targets.py --images coco_images_224_float16.hdf5 \
        --out color_targets.npy
Outputs color_targets.npy (n_images, 11) and color_targets_ids.npy (NSD ids),
plus prints the color order.
"""

import argparse
import numpy as np
# h5py is imported lazily inside run() so the color functions below can be
# reused (e.g. by train_v4_color.py) without requiring h5py.

COLOR_NAMES = ["red", "orange", "yellow", "green", "blue",
               "purple", "pink", "brown", "black", "white", "gray"]
NAME_TO_IDX = {c: i for i, c in enumerate(COLOR_NAMES)}


def rgb_to_hsv_np(rgb):
    """rgb in [0,1], shape (...,3) -> hsv in [0,1] (h in [0,1))."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn + 1e-12
    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) + 360) % 360
    mask = mx == g
    h[mask] = (60 * ((b[mask] - r[mask]) / diff[mask]) + 120) % 360
    mask = mx == b
    h[mask] = (60 * ((r[mask] - g[mask]) / diff[mask]) + 240) % 360
    s = np.where(mx == 0, 0, (mx - mn) / (mx + 1e-12))
    v = mx
    return h / 360.0, s, v


def classify_pixels(h, s, v):
    """Assign each pixel (arrays of h[0,1], s[0,1], v[0,1]) to a color index."""
    idx = np.full(h.shape, NAME_TO_IDX["gray"], dtype=np.int8)
    hue = h * 360.0

    # achromatic first
    idx[v < 0.2] = NAME_TO_IDX["black"]
    achrom = s < 0.15
    idx[achrom & (v >= 0.2) & (v < 0.75)] = NAME_TO_IDX["gray"]
    idx[achrom & (v >= 0.75)] = NAME_TO_IDX["white"]

    chrom = (s >= 0.15) & (v >= 0.2)

    def band(lo, hi):
        return chrom & (hue >= lo) & (hue < hi)

    # brown = dark/desaturated orange-red
    brown = chrom & (hue < 45) & (v < 0.6) & (s > 0.3)
    # pink = light red/magenta
    pink = chrom & (((hue < 20) | (hue >= 330)) & (v > 0.8) & (s < 0.5))

    idx[band(0, 15) | band(345, 360)] = NAME_TO_IDX["red"]
    idx[band(15, 45)] = NAME_TO_IDX["orange"]
    idx[band(45, 70)] = NAME_TO_IDX["yellow"]
    idx[band(70, 170)] = NAME_TO_IDX["green"]
    idx[band(170, 260)] = NAME_TO_IDX["blue"]
    idx[band(260, 300)] = NAME_TO_IDX["purple"]
    idx[band(300, 345)] = NAME_TO_IDX["pink"]
    # apply the two overrides last
    idx[brown] = NAME_TO_IDX["brown"]
    idx[pink] = NAME_TO_IDX["pink"]
    return idx


def image_color_hist(arr):
    """arr: (H,W,3) uint8 or float -> length-11 distribution summing to 1."""
    a = arr.astype(np.float32)
    if a.max() > 1.001:
        a = a / 255.0
    h, s, v = rgb_to_hsv_np(a)
    idx = classify_pixels(h, s, v)
    counts = np.bincount(idx.ravel(), minlength=len(COLOR_NAMES)).astype(np.float32)
    return counts / counts.sum()


def run(h5path, out, nsd_ids=None, batch=256):
    import h5py
    with h5py.File(h5path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        dset = f[key]
        n = dset.shape[0]
        ids = list(range(n)) if nsd_ids is None else sorted(nsd_ids)
        out_vecs = np.zeros((len(ids), len(COLOR_NAMES)), dtype=np.float32)
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            arr = np.asarray(dset[chunk])
            if arr.ndim == 4 and arr.shape[1] == 3:      # (N,3,H,W)->(N,H,W,3)
                arr = arr.transpose(0, 2, 3, 1)
            for j, im in enumerate(arr):
                out_vecs[i + j] = image_color_hist(im)
            print(f"  processed {min(i + batch, len(ids))}/{len(ids)}")

    np.save(out, out_vecs)
    np.save(out.replace(".npy", "_ids.npy"), np.asarray(ids))
    print(f"Saved {out} shape={out_vecs.shape}")
    print("Color order:", COLOR_NAMES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--nsd-ids", nargs="+", type=int)
    p.add_argument("--out", default="color_targets.npy")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    run(args.images, args.out, nsd_ids=args.nsd_ids, batch=args.batch)


if __name__ == "__main__":
    main()
