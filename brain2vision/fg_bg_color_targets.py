"""
fg_bg_color_targets.py
======================
Split each image's COLOUR into a FOREGROUND part (pixels inside any COCO object
bounding box -- the attended objects) and a BACKGROUND part (pixels outside all
boxes), and build a separate 11-way colour histogram for each.

Why
---
Residualising colour against object *presence* collapsed colour decoding in
every ROI, but that test can't separate "colour was semantic-bound" from "we
removed the colour of the attended foreground and left weakly-represented
background colour" (the attention confound). This module adjudicates: decode
foreground colour vs background colour per ROI.

    * foreground decodes well, background near zero  -> attention/foreground
      account supported (the brain codes the colour of what it's looking at);
    * background decodes about as well as foreground  -> attention is not the
      driver.

Method
------
Boxes come from bboxes.py in the 425x425 NSD stimulus frame; we scale them to the
image resolution and take the union as the foreground mask. Colour bins reuse
color_targets.py. Outputs align 1:1 with color_targets (same ids), so the
existing replicate_subjects decodes them unchanged.

Caveat: bounding boxes are rectangular and over-include some background inside
each box, so "foreground" is an approximation of "attended object". Good enough
to test the attention account; not a pixel-perfect segmentation.

Usage
-----
    pip install h5py numpy pandas requests
    python -m brain2vision.fg_bg_color_targets \
        --images coco_images_224_float16.hdf5 --out data/fgbg
    # then decode each per ROI, across subjects:
    python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 \
        --target data/fgbg_fg.npy --out roi_fgcolor_8subj.png
    python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 \
        --target data/fgbg_bg.npy --out roi_bgcolor_8subj.png

Outputs: <out>_fg.npy, <out>_fg_ids.npy, <out>_bg.npy, <out>_bg_ids.npy,
and <out>_coverage.npy (foreground pixel fraction per image, for QC).
"""

import argparse
import numpy as np

from brain2vision.color_targets import rgb_to_hsv_np, classify_pixels, COLOR_NAMES
from brain2vision.bboxes import build as build_boxes, STIM_SIZE


def _load_image(dset, gid):
    """One image as (H,W,3) float in [0,1], handling channel order."""
    im = np.asarray(dset[gid]).astype(np.float32)
    if im.ndim == 3 and im.shape[0] == 3:
        im = im.transpose(1, 2, 0)
    if im.max() > 1.001:
        im = im / 255.0
    return np.clip(im, 0, 1)


def preview(images_h5, ids, out, cache="coco_cache", min_box_frac=0.0, largest_only=False):
    """
    Save a figure showing, for each id: the image with object boxes outlined,
    the foreground-only pixels, and the background-only pixels -- so you can see
    exactly what the fg/bg colour split is measuring (and how much area each is).
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    boxes_by_id = build_boxes(nsd_ids=ids, out=f"{out}_boxes.json", cache=cache)
    with h5py.File(images_h5, "r") as f:
        key = next(k for k in f if isinstance(f[k], h5py.Dataset))
        dset = f[key]
        fig, axes = plt.subplots(len(ids), 3, figsize=(9, 3 * len(ids)), squeeze=False)
        for i, gid in enumerate(ids):
            im = _load_image(dset, gid)
            R = im.shape[0]
            allb = boxes_by_id.get(gid, [])
            m = foreground_mask(allb, R, min_frac=min_box_frac, largest_only=largest_only)
            if largest_only and allb:
                kept = [max(allb, key=lambda b: b["bbox_xywh"][2] * b["bbox_xywh"][3])]
            else:
                kept = [b for b in allb
                        if (b["bbox_xywh"][2] * b["bbox_xywh"][3]) / STIM_SIZE ** 2 >= min_box_frac]
            axes[i, 0].imshow(im); axes[i, 0].set_title(f"image {gid} + boxes", fontsize=9)
            for b in kept:
                x, y, w, h = [v * R / STIM_SIZE for v in b["bbox_xywh"]]
                axes[i, 0].add_patch(Rectangle((x, y), w, h, fill=False,
                                               edgecolor="lime", lw=1.5))
            axes[i, 1].imshow(im * m[..., None])
            axes[i, 1].set_title(f"foreground ({100*m.mean():.0f}% of pixels)", fontsize=9)
            axes[i, 2].imshow(im * (~m)[..., None])
            axes[i, 2].set_title(f"background ({100*(1-m.mean()):.0f}%)", fontsize=9)
            for a in axes[i]:
                a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


def foreground_mask(boxes, R, min_frac=0.0, largest_only=False):
    """
    Union of object boxes as a boolean (R,R) mask, boxes given in 425 frame.

    min_frac      : drop boxes smaller than this fraction of the image area
                    (removes the scattered small/distant "background objects" that
                    COCO still labels).
    largest_only  : keep only the single biggest box -- the dominant, most-likely-
                    attended object -- so a real background survives even when many
                    objects overlap.
    """
    m = np.zeros((R, R), dtype=bool)
    scale = R / float(STIM_SIZE)
    if not boxes:
        return m
    kept = boxes
    if largest_only:
        kept = [max(boxes, key=lambda b: b["bbox_xywh"][2] * b["bbox_xywh"][3])]
    for b in kept:
        x, y, w, h = b["bbox_xywh"]
        if (w * h) / float(STIM_SIZE ** 2) < min_frac:
            continue
        x0 = max(0, int(round(x * scale))); y0 = max(0, int(round(y * scale)))
        x1 = min(R, int(round((x + w) * scale))); y1 = min(R, int(round((y + h) * scale)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def _hist(idx_map, mask):
    """11-way colour histogram over the masked pixels (sums to 1, or all-zero)."""
    sel = idx_map[mask]
    counts = np.bincount(sel.ravel(), minlength=len(COLOR_NAMES)).astype(np.float32)
    tot = counts.sum()
    return counts / tot if tot > 0 else counts


def _n_real_objects(boxes, min_frac):
    """How many boxes are 'real' objects (area >= min_frac of the image)."""
    return sum((b["bbox_xywh"][2] * b["bbox_xywh"][3]) / STIM_SIZE ** 2 >= min_frac
               for b in boxes)


def run(images_h5, out_prefix, nsd_ids=None, cache="coco_cache", batch=256,
        min_box_frac=0.0, largest_only=False, single_object=False,
        coverage_min=0.0, coverage_max=1.0):
    """
    Build fg/bg colour targets. With the "clean image" filters, only unambiguous
    images are kept -- so the fg/bg contrast is a fair test rather than being
    muddied by multi-object scenes (which object is 'the foreground'?):

      single_object=True         : keep only images with exactly ONE real object
                                   (drops the two-buses / two-girls ambiguity);
      coverage_min/max           : keep only images whose foreground covers a
                                   sensible fraction (a real foreground AND a real
                                   background both present).
    Only kept images are written, so replicate_subjects decodes just those.
    """
    import h5py
    with h5py.File(images_h5, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        n = f[key].shape[0]
    ids = list(range(n)) if nsd_ids is None else sorted(int(i) for i in nsd_ids)
    print(f"building boxes for {len(ids)} images...")
    boxes_by_id = build_boxes(nsd_ids=ids, out=f"{out_prefix}_boxes.json", cache=cache)

    fg_l, bg_l, cov_l, kept = [], [], [], []
    n_seen = 0
    with h5py.File(images_h5, "r") as f:
        dset = f[key]
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            arr = np.asarray(dset[chunk])
            if arr.ndim == 4 and arr.shape[1] == 3:          # (N,3,H,W)->(N,H,W,3)
                arr = arr.transpose(0, 2, 3, 1)
            for j, im in enumerate(arr):
                n_seen += 1
                gid = chunk[j]
                boxes = boxes_by_id.get(gid, [])
                if single_object and _n_real_objects(boxes, min_box_frac) != 1:
                    continue                                  # ambiguous / no clear object
                a = im.astype(np.float32)
                if a.max() > 1.001:
                    a = a / 255.0
                R = a.shape[0]
                m = foreground_mask(boxes, R, min_frac=min_box_frac, largest_only=largest_only)
                c = float(m.mean())
                if not (coverage_min <= c <= coverage_max):
                    continue                                  # no real fg or no real bg
                h, s, v = rgb_to_hsv_np(a)
                idx_map = classify_pixels(h, s, v)
                fg_l.append(_hist(idx_map, m)); bg_l.append(_hist(idx_map, ~m))
                cov_l.append(c); kept.append(gid)
            print(f"  processed {min(i + batch, len(ids))}/{len(ids)}  (kept {len(kept)})")

    fg = np.asarray(fg_l, np.float32); bg = np.asarray(bg_l, np.float32)
    ids_arr = np.asarray(kept); cov = np.asarray(cov_l, np.float32)
    np.save(f"{out_prefix}_fg.npy", fg); np.save(f"{out_prefix}_fg_ids.npy", ids_arr)
    np.save(f"{out_prefix}_bg.npy", bg); np.save(f"{out_prefix}_bg_ids.npy", ids_arr)
    np.save(f"{out_prefix}_coverage.npy", cov)
    filt = []
    if single_object: filt.append("single-object")
    if min_box_frac: filt.append(f"min-box>={min_box_frac}")
    if coverage_min > 0 or coverage_max < 1: filt.append(f"coverage {coverage_min}-{coverage_max}")
    print(f"\nSaved {out_prefix}_fg.npy / _bg.npy  (shape {fg.shape})")
    print(f"  kept {len(kept)}/{n_seen} images"
          + (f"  [filters: {', '.join(filt)}]" if filt else "")
          + (f"  (mean foreground coverage {cov.mean():.2f})" if len(cov) else ""))
    print(f"  colour order: {COLOR_NAMES}")
    print("Next: decode _fg.npy and _bg.npy per ROI with replicate_subjects and compare.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="image hdf5 (same as color_targets)")
    p.add_argument("--out", default="data/fgbg", help="output prefix")
    p.add_argument("--nsd-ids", nargs="+", type=int)
    p.add_argument("--cache", default="coco_cache")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--preview-ids", nargs="+", type=int,
                   help="just render the fg/bg split for these image ids "
                        "(saves <out>_preview.png), don't build full targets")
    p.add_argument("--min-box-frac", type=float, default=0.0,
                   help="drop object boxes smaller than this fraction of the image "
                        "(removes scattered background objects; try 0.05)")
    p.add_argument("--largest-object", action="store_true",
                   help="foreground = only the single biggest object (dominant/"
                        "attended); leaves a real background even when objects overlap")
    p.add_argument("--single-object", action="store_true",
                   help="CLEAN images only: keep only images with exactly ONE real "
                        "object (drops ambiguous multi-object scenes)")
    p.add_argument("--coverage-min", type=float, default=0.0,
                   help="clean filter: minimum foreground coverage (real object present)")
    p.add_argument("--coverage-max", type=float, default=1.0,
                   help="clean filter: maximum foreground coverage (real background present)")
    args = p.parse_args()
    if args.preview_ids:
        preview(args.images, args.preview_ids, f"{args.out}_preview.png", cache=args.cache,
                min_box_frac=args.min_box_frac, largest_only=args.largest_object)
    else:
        run(args.images, args.out, nsd_ids=args.nsd_ids, cache=args.cache, batch=args.batch,
            min_box_frac=args.min_box_frac, largest_only=args.largest_object,
            single_object=args.single_object,
            coverage_min=args.coverage_min, coverage_max=args.coverage_max)


if __name__ == "__main__":
    main()
