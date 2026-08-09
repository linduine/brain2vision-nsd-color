"""
synthetic_targets.py
=====================
Build decoding targets for the NSD-synthetic colour control, computed DIRECTLY
from the stimulus pixels (so nothing depends on a separate metadata/label file).

Primary target: HUE of the isoluminant chromatic pink-noise images.
    The 64 chromatic images span 16 evenly-spaced hues at (near-)constant
    luminance, so the right target is the hue ANGLE, represented as
    (cos, sin) -- a circular encoding that a linear decoder can fit and that
    avoids the 0deg/360deg wrap-around problem of a raw angle. We report
    decoding as R^2 on the 2-D (cos, sin) target.

Also produced (for continuity with the NSD-core experiment):
    * the 11-way basic-colour histogram (same bins as color_targets.py);
    * a per-image saturation score + a boolean CHROMATIC-subset mask, so the
      hue decode can be restricted to the images that actually carry colour.

Stimuli (RAW NSD, S3 bucket "natural-scenes-dataset")
-----------------------------------------------------
    nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_colorstimuli.hdf5
The colour stimuli file holds the RGB versions of the chromatic images (the
grayscale nsdsynthetic_stimuli.hdf5 has no colour and would give zero hue).
Pass whichever file you have with --images; the code auto-detects the dataset
key and channel order, and flags images with no chromatic content.
VERIFY the exact filename against the NSD manual; override with --images.

Usage
-----
    pip install h5py numpy
    python -m brain2vision.synthetic_targets \
        --images nsdsynthetic_colorstimuli.hdf5 --out data/synth
Outputs (prefix from --out):
    <out>_hue.npy         (n_images, 2)  cos/sin of mean hue  (0 where achromatic)
    <out>_huedeg.npy      (n_images,)    mean hue in degrees  (nan where achromatic)
    <out>_color.npy       (n_images, 11) basic-colour histogram
    <out>_saturation.npy  (n_images,)    mean chroma score
    <out>_chromatic.npy   (n_images,)    bool: image carries colour
    <out>_ids.npy         (n_images,)    1-based synthetic image numbers
"""

import argparse
import numpy as np

from brain2vision.color_targets import rgb_to_hsv_np, image_color_hist, COLOR_NAMES

# A pixel counts as "chromatic" if its saturation exceeds this; an image is
# chromatic if enough of its pixels are.
PIXEL_SAT_MIN = 0.10
IMAGE_CHROMA_FRAC_MIN = 0.20


def image_hue(arr):
    """
    (H,W,3) -> (cos_mean, sin_mean, mean_hue_deg_or_nan, chroma_score, is_chromatic).

    Mean hue is the SATURATION-WEIGHTED circular mean over chromatic pixels, so
    grey pixels (no meaningful hue) don't dilute the estimate. Returns zeros /
    nan for images with essentially no colour.
    """
    a = arr.astype(np.float32)
    if a.max() > 1.001:
        a = a / 255.0
    h, s, v = rgb_to_hsv_np(a)             # h in [0,1)
    ang = h.ravel() * 2.0 * np.pi          # radians
    sat = s.ravel()
    chrom = sat >= PIXEL_SAT_MIN
    chroma_score = float(sat.mean())
    is_chrom = bool(chrom.mean() >= IMAGE_CHROMA_FRAC_MIN)
    if not is_chrom or chrom.sum() == 0:
        return 0.0, 0.0, np.nan, chroma_score, is_chrom
    w = sat[chrom]
    c = np.average(np.cos(ang[chrom]), weights=w)
    sN = np.average(np.sin(ang[chrom]), weights=w)
    mag = np.hypot(c, sN) + 1e-12
    hue_deg = (np.degrees(np.arctan2(sN, c)) + 360.0) % 360.0
    # return the *unit* direction (cos,sin) as the decoding target
    return c / mag, sN / mag, float(hue_deg), chroma_score, is_chrom


def run(h5path, out_prefix, batch=64):
    import h5py
    with h5py.File(h5path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        dset = f[key]
        n = dset.shape[0]
        print(f"stimuli dataset '{key}' shape={dset.shape}")
        hue = np.zeros((n, 2), np.float32)
        huedeg = np.full(n, np.nan, np.float32)
        color = np.zeros((n, len(COLOR_NAMES)), np.float32)
        sat = np.zeros(n, np.float32)
        chrom = np.zeros(n, bool)
        for i in range(0, n, batch):
            arr = np.asarray(dset[i:i + batch])
            if arr.ndim == 4 and arr.shape[1] == 3:        # (N,3,H,W)->(N,H,W,3)
                arr = arr.transpose(0, 2, 3, 1)
            for j, im in enumerate(arr):
                c, sN, hd, cs, ic = image_hue(im)
                hue[i + j] = (c, sN); huedeg[i + j] = hd
                sat[i + j] = cs; chrom[i + j] = ic
                color[i + j] = image_color_hist(im)
            print(f"  processed {min(i + batch, n)}/{n}")

    ids = np.arange(1, n + 1)
    np.save(f"{out_prefix}_hue.npy", hue)
    np.save(f"{out_prefix}_huedeg.npy", huedeg)
    np.save(f"{out_prefix}_color.npy", color)
    np.save(f"{out_prefix}_saturation.npy", sat)
    np.save(f"{out_prefix}_chromatic.npy", chrom)
    np.save(f"{out_prefix}_ids.npy", ids)
    print(f"\nSaved targets with prefix '{out_prefix}_*.npy'")
    print(f"  chromatic images: {int(chrom.sum())}/{n} "
          f"(expected ~64 for nsdsynthetic colour stimuli)")
    if chrom.sum():
        print(f"  hue range over chromatic imgs: "
              f"{np.nanmin(huedeg):.0f}..{np.nanmax(huedeg):.0f} deg")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True,
                   help="nsdsynthetic colour stimuli hdf5")
    p.add_argument("--out", default="data/synth", help="output prefix")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    run(args.images, args.out, batch=args.batch)


if __name__ == "__main__":
    main()
