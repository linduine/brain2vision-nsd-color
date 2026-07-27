"""
perceptual_color_targets.py
===========================
NEXT-STEP / SCAFFOLD — not part of the validated result.

Build a PERCEPTUAL color target using the van de Weijer et al. learned
color-name model (CVPR 2007, "Learning Color Names from Real-World Images"; IEEE
TIP 2009, "Learning Color Names for Real-World Applications"). Instead of the
hand-written HSV rules in color_targets.py — which dump desaturated pixels (a
washed-out sky) into white/gray — this uses a lookup learned from how people
actually name colors, so pale sky-blue keeps real "blue" probability.

The model is a 32768 x 11 table `w2c` mapping quantized RGB (32 bins/channel) to
a probability over the 11 basic colors. Per image we look up each pixel's 11-way
name distribution and average — same output shape/format as color_targets.py, in
the SAME color order, so it drops straight into replicate_subjects / color_decode.

Getting the table (one-time)
----------------------------
The `w2c` table ships with van de Weijer's code (and is widely mirrored, e.g. in
CV tracking repos as `w2c.mat`). Convert it to `data/w2c.npy` once:

    # from w2c.mat (columns already in van de Weijer order):
    import scipy.io, numpy as np
    w = scipy.io.loadmat("w2c.mat")["w2c"]        # (32768, 11)
    np.save("data/w2c.npy", w.astype("float32"))

Usage
-----
    python -m brain2vision.perceptual_color_targets \
        --images data/coco_images_224_float16.hdf5 \
        --w2c data/w2c.npy --out data/color_targets_perceptual.npy

Then decode exactly like the HSV target:
    python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 \
        --target data/color_targets_perceptual.npy --out roi_percept_8subj.png
"""

import argparse
import numpy as np

# van de Weijer w2c column order:
W2C_ORDER = ["black", "blue", "brown", "grey", "green", "orange",
             "pink", "purple", "red", "white", "yellow"]
# our canonical order (matches color_targets.COLOR_NAMES):
OUR_ORDER = ["red", "orange", "yellow", "green", "blue", "purple",
             "pink", "brown", "black", "white", "gray"]
# index map so the output columns are in OUR order (gray <-> grey):
_REMAP = [W2C_ORDER.index("grey" if c == "gray" else c) for c in OUR_ORDER]


def _rgb_to_w2c_index(rgb_uint8):
    """rgb_uint8: (...,3) in 0..255 -> flat bin index in 0..32767 (32 bins/ch)."""
    a = rgb_uint8.astype(np.int32)          # widen before arithmetic (avoid uint8 overflow)
    r = a[..., 0] >> 3
    g = a[..., 1] >> 3
    b = a[..., 2] >> 3
    return r + 32 * g + 1024 * b


def image_perceptual_hist(arr, w2c):
    """arr: (H,W,3) uint8/float -> length-11 perceptual color distribution
    (in OUR color order), the mean of per-pixel name distributions."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        a = (a * 255 if a.max() <= 1.001 else a)
        a = np.clip(a, 0, 255).astype(np.uint8)
    idx = _rgb_to_w2c_index(a).ravel()             # (H*W,)
    probs = w2c[idx]                                # (H*W, 11) in w2c order
    mean = probs.mean(0)                            # (11,)
    mean = mean[_REMAP]                             # reorder to OUR order
    s = mean.sum()
    return (mean / s).astype(np.float32) if s > 0 else mean.astype(np.float32)


def run(h5path, w2c_path, out, nsd_ids=None, batch=256):
    import h5py
    w2c = np.load(w2c_path).astype(np.float32)
    assert w2c.shape == (32768, 11), f"w2c must be (32768, 11), got {w2c.shape}"
    with h5py.File(h5path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        dset = f[key]
        n = dset.shape[0]
        ids = list(range(n)) if nsd_ids is None else sorted(nsd_ids)
        out_vecs = np.zeros((len(ids), 11), dtype=np.float32)
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            arr = np.asarray(dset[chunk])
            if arr.ndim == 4 and arr.shape[1] == 3:
                arr = arr.transpose(0, 2, 3, 1)
            for j, im in enumerate(arr):
                out_vecs[i + j] = image_perceptual_hist(im, w2c)
            print(f"  processed {min(i + batch, len(ids))}/{len(ids)}")
    np.save(out, out_vecs)
    np.save(out.replace(".npy", "_ids.npy"), np.asarray(ids))
    print(f"Saved {out} shape={out_vecs.shape} (perceptual color names, our order)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--w2c", default="data/w2c.npy",
                   help="van de Weijer 32768x11 color-name table (.npy)")
    p.add_argument("--out", default="data/color_targets_perceptual.npy")
    p.add_argument("--nsd-ids", nargs="+", type=int)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    run(args.images, args.w2c, args.out, nsd_ids=args.nsd_ids, batch=args.batch)


if __name__ == "__main__":
    main()
