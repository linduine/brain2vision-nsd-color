#!/usr/bin/env python3
"""
luminance_hue_test.py
=====================
Within a given kind of object, does its luminance distinguish its hue?

The question
------------
If a reconstruction predicts "red jacket" from semantics but the brain-decoded
luminance is inconsistent with red, that disagreement could flag an atypically
coloured object. That requires luminance to carry hue information *for a given
object*, a red car versus a blue car, which is the comparison this script makes.

What the previous version got wrong
-----------------------------------
It compared luminance across objects of DIFFERENT categories: a blue car against
a green tree against a brown table. Hue was then confounded with object identity
and with scene context, blue things sit in bright outdoor scenes, brown things
indoors, so "blue objects are bright" came out, which is a fact about skies, not
about blue. That is the same content–colour confound the manuscript is about,
reappearing inside the control analysis.

Conditioning on category removes it. Within `car`, blue means a blue car.

Method
------
For each image in the clean single-dominant-object subset: take the largest box,
compute mean relative luminance inside it (Y = 0.2126R + 0.7152G + 0.0722B) and
outside it, and the dominant basic colour term of the box, using the same
`classify_pixels` rule as the manuscript's colour target. Record the COCO
category.

Then, keeping only objects whose dominant term is chromatic:

  1. per category, mean luminance for each hue, and the spread across hues
  2. pooled after subtracting each category's mean luminance, so between-category
     brightness cancels and only within-category variation remains
  3. hue classification from luminance alone against the majority-class baseline

What would count as a positive result
-------------------------------------
Within-category hue means that separate by more than a few percent of the
luminance range, in a consistent direction across categories, and a
classification gain that survives the pooled centred test. A gain of one or two
points is not usable for correcting a reconstruction.

Caveats that cap this from above
--------------------------------
Boxes are rectangular and include background, so the object's luminance is
diluted towards the scene's; `--centre-frac 0.5` restricts to the central part of
the box, which is more likely to be object. And the "dominant term of the box" is
a coarse stand-in for "the object's colour" without pixel-accurate segmentation.
Both push the measured effect down, so this is a floor, not a ceiling.

Usage
-----
The image array is not redistributed here; pass its path explicitly.

    python luminance_hue_test.py --images <coco_images_224_float16.hdf5> \
        --limit 4000 --cache lum_hue.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from brain2vision.color_targets import classify_pixels, rgb_to_hsv_np, COLOR_NAMES
from brain2vision.bboxes import STIM_SIZE

ACHRO = {"black", "white", "gray"}
CHROM = [c for c in COLOR_NAMES if c not in ACHRO]
COEF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)   # BT.709


def _load_image(dset, gid):
    im = np.asarray(dset[gid]).astype(np.float32)
    if im.ndim == 3 and im.shape[0] == 3:
        im = im.transpose(1, 2, 0)
    if im.max() > 1.001:
        im = im / 255.0
    return np.clip(im, 0, 1)


def _box_mask(box, R, centre_frac=1.0):
    """Boolean (R,R) mask for a box given in the 425-pixel stimulus frame.

    centre_frac < 1 shrinks the box about its centre, which trades area for a
    higher chance that the pixels belong to the object rather than behind it.
    """
    x, y, w, h = [v * R / STIM_SIZE for v in box]
    if centre_frac < 1.0:
        cx, cy = x + w / 2, y + h / 2
        w, h = w * centre_frac, h * centre_frac
        x, y = cx - w / 2, cy - h / 2
    m = np.zeros((R, R), bool)
    y0, y1 = max(int(np.floor(y)), 0), min(int(np.ceil(y + h)), R)
    x0, x1 = max(int(np.floor(x)), 0), min(int(np.ceil(x + w)), R)
    m[y0:y1, x0:x1] = True
    return m


def dominant_hue(rgb_pixels):
    h, s, v = rgb_to_hsv_np(rgb_pixels[None, :, :])
    idx = classify_pixels(h, s, v).ravel()
    return COLOR_NAMES[int(np.argmax(np.bincount(idx, minlength=len(COLOR_NAMES))))]


def extract(args):
    import h5py

    if not os.path.exists(args.images):
        raise SystemExit(
            f"image file not found: {args.images}\n"
            "Pass --images with the path to the MindEye2 image array. It is not "
            "redistributed with this repository; see the MindEye2 release.")

    ids = np.load(args.ids)
    if args.limit:
        ids = ids[: args.limit]
    boxes_by_id = json.load(open(args.boxes))

    gid, cat, hue, y_obj, y_bg = [], [], [], [], []
    with h5py.File(args.images, "r") as f:
        dset = f[next(k for k in f if isinstance(f[k], h5py.Dataset))]
        for n, g in enumerate(ids):
            bl = boxes_by_id.get(str(int(g))) or boxes_by_id.get(int(g))
            if not bl:
                continue
            b = max(bl, key=lambda d: d["bbox_xywh"][2] * d["bbox_xywh"][3])
            im = _load_image(dset, int(g))
            R = im.shape[0]
            m = _box_mask(b["bbox_xywh"], R, args.centre_frac)
            out = ~_box_mask(b["bbox_xywh"], R, 1.0)      # background = outside full box
            if m.sum() < 50 or out.sum() < 50:
                continue
            Y = im @ COEF
            gid.append(int(g)); cat.append(b["category"]); hue.append(dominant_hue(im[m]))
            y_obj.append(float(Y[m].mean())); y_bg.append(float(Y[out].mean()))
            if n and n % 2000 == 0:
                print(f"  {n}/{len(ids)}")
    return (np.array(gid), np.array(cat), np.array(hue),
            np.array(y_obj), np.array(y_bg))


def loo_acc(x, lab, seed=0, folds=5):
    """1-D Gaussian class-conditional classifier, cross-validated."""
    classes = np.unique(lab)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    correct = 0
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        mu, sd, pri = {}, {}, {}
        for c in classes:
            v = x[tr][lab[tr] == c]
            if len(v) < 5:
                continue
            mu[c], sd[c], pri[c] = v.mean(), v.std() + 1e-6, len(v) / len(tr)
        for i in te:
            best, bc = -np.inf, classes[0]
            for c in mu:
                ll = (-0.5 * ((x[i] - mu[c]) / sd[c]) ** 2
                      - np.log(sd[c]) + np.log(pri[c]))
                if ll > best:
                    best, bc = ll, c
            correct += (bc == lab[i])
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="path to the MindEye2 image array (not redistributed here)")
    ap.add_argument("--boxes", default="data/fgbg_clean_boxes.json")
    ap.add_argument("--ids", default="data/fgbg_clean_fg_ids.npy")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--centre-frac", type=float, default=1.0,
                    help="shrink each box about its centre; 0.5 halves each side")
    ap.add_argument("--cache", default=None,
                    help="npz to write/reuse, so the image pass runs once")
    ap.add_argument("--min-per-cat", type=int, default=60)
    ap.add_argument("--min-per-hue", type=int, default=15)
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        gid, cat, hue, y_obj, y_bg = (z["gid"], z["cat"], z["hue"], z["y_obj"], z["y_bg"])
        print(f"reusing {args.cache} ({len(gid):,} objects)")
    else:
        gid, cat, hue, y_obj, y_bg = extract(args)
        if args.cache:
            np.savez(args.cache, gid=gid, cat=cat, hue=hue, y_obj=y_obj, y_bg=y_bg)
            print(f"wrote {args.cache}")

    keep = np.isin(hue, CHROM)
    cat, hue, y_obj, y_bg = cat[keep], hue[keep], y_obj[keep], y_bg[keep]
    print(f"\n{len(hue):,} objects with a chromatic dominant term, "
          f"{len(np.unique(cat))} categories\n")

    # ---- 1. within each category ------------------------------------------
    print("Within-category luminance by hue (categories with enough instances)")
    print(f"{'category':14s}{'n':>5s}   " + "".join(f"{c[:6]:>8s}" for c in CHROM)
          + f"{'spread':>9s}")
    spreads, usable = [], 0
    for c in sorted(set(cat)):
        k = cat == c
        if k.sum() < args.min_per_cat:
            continue
        means, cells = [], []
        for hcol in CHROM:
            kk = k & (hue == hcol)
            if kk.sum() >= args.min_per_hue:
                means.append(y_obj[kk].mean()); cells.append(f"{y_obj[kk].mean():8.3f}")
            else:
                cells.append(f"{'·':>8s}")
        if len(means) < 2:
            continue
        usable += 1
        sp = max(means) - min(means); spreads.append(sp)
        print(f"{c[:14]:14s}{k.sum():5d}   " + "".join(cells) + f"{sp:9.3f}")
    if spreads:
        print(f"\n  {usable} categories usable; median within-category spread across "
              f"hues = {np.median(spreads):.3f}")
        print(f"  (the full luminance range in this sample is "
              f"{y_obj.min():.2f}–{y_obj.max():.2f})")

    # ---- 2. pooled, centred within category --------------------------------
    big = np.array([ (cat == c).sum() >= args.min_per_cat for c in cat ])
    cc, hh, yy = cat[big], hue[big], y_obj[big].copy()
    for c in np.unique(cc):                       # remove each category's mean
        yy[cc == c] -= yy[cc == c].mean()
    _, counts = np.unique(hh, return_counts=True)
    base = counts.max() / counts.sum()
    print(f"\nPooled within-category (n = {len(hh):,}, "
          f"{len(np.unique(hh))} hues, between-category brightness removed)")
    print(f"  majority-class baseline        : {base:.3f}")
    print(f"  hue from centred luminance     : {loo_acc(yy, hh):.3f}")
    print(f"  hue from raw luminance         : {loo_acc(y_obj[big], hh):.3f}")
    se = np.sqrt(base * (1 - base) / len(hh))
    print(f"  (1 SE on an accuracy at baseline ≈ {se:.3f})")
    print("\nA gain of a few SE is real but not usable for correcting a colour;")
    print("what would matter is separation of the per-hue means above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
