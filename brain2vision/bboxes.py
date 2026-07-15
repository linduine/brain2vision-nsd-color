"""
bboxes.py
====================
Link each NSD image to its COCO object annotations and express every bounding
box in the NSD STIMULUS FRAME -- i.e. the 425x425 center-cropped+resized image
the subject actually saw, not the original COCO image.

Why the transform matters
-------------------------
NSD took each COCO image, center-cropped it to a square, and resized to
425x425. Raw COCO boxes are in original-image pixels, so they must be cropped
and rescaled to line up with the stimulus. nsd_stim_info_merged.csv gives the
per-image `cropBox` needed to do this.

Inputs (downloaded automatically unless paths are given)
--------------------------------------------------------
* NSD image info : nsddata/experiments/nsd/nsd_stim_info_merged.csv   (S3)
* COCO 2017 instance annotations : instances_train2017.json /
  instances_val2017.json  (from cocodataset.org; ~250 MB zip)

Output
------
JSON keyed by NSD id (0-based, aligns with the 73k image order used by the
MindEye2 hdf5 files and nsd_stimuli.hdf5). Each value is a list of boxes:
    {"category": "dog", "bbox_xywh": [x, y, w, h]}   # in 425x425 coords

Usage
-----
    pip install pandas numpy requests
    python -m brain2vision.bboxes --out nsd_bboxes.json            # all 73k
    python -m brain2vision.bboxes --nsd-ids 0 1 2 3 --out demo.json

NOTE: cropBox is interpreted as (top, bottom, left, right) fractions removed
from each side (NSD manual convention). Sanity-check a few boxes visually
before trusting at scale -- draw them on the corresponding nsd_stimuli image.
"""

import os
import io
import json
import zipfile
import argparse
import ast
import numpy as np
import pandas as pd
import requests

S3_BASE = "https://natural-scenes-dataset.s3.amazonaws.com"
STIM_INFO_URL = f"{S3_BASE}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
STIM_SIZE = 425  # NSD presented-stimulus resolution (pixels, square)


def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest + ".part", "wb") as fh:
            for c in r.iter_content(1 << 20):
                fh.write(c)
    os.replace(dest + ".part", dest)
    return dest


def _ensure_coco_annotations(cache="coco_cache"):
    """Download + unzip COCO 2017 instances json files if not present."""
    os.makedirs(cache, exist_ok=True)
    train = os.path.join(cache, "annotations", "instances_train2017.json")
    val = os.path.join(cache, "annotations", "instances_val2017.json")
    if os.path.exists(train) and os.path.exists(val):
        return train, val
    zip_path = _download(COCO_ANN_ZIP, os.path.join(cache, "ann.zip"))
    print("unzipping COCO annotations...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache)
    return train, val


def _load_coco(json_path):
    """Return (imgId->[anns], imgId->(W,H), catId->name) from a COCO json."""
    with open(json_path) as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    sizes = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
    anns = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return anns, sizes, cats


def _parse_cropbox(raw):
    """cropBox stored as a string tuple '(top, bottom, left, right)'."""
    if isinstance(raw, (tuple, list)):
        return tuple(float(x) for x in raw)
    return tuple(float(x) for x in ast.literal_eval(str(raw)))


def transform_bbox(bbox_xywh, W, H, cropbox):
    """
    Map a COCO bbox (original pixels) into the 425x425 NSD stimulus frame.
    Returns None if the box lies fully outside the crop.
    """
    top, bottom, left, right = cropbox
    x0 = left * W            # crop origin in original pixels
    y0 = top * H
    crop_w = W - (left + right) * W
    crop_h = H - (top + bottom) * H
    # NSD crops the longer dimension to a square, so crop_w ~= crop_h.
    sx = STIM_SIZE / crop_w
    sy = STIM_SIZE / crop_h

    x, y, w, h = bbox_xywh
    nx = (x - x0) * sx
    ny = (y - y0) * sy
    nw = w * sx
    nh = h * sy
    # clip to the visible stimulus and drop if nothing remains
    x1 = max(0.0, nx); y1 = max(0.0, ny)
    x2 = min(STIM_SIZE, nx + nw); y2 = min(STIM_SIZE, ny + nh)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def build(nsd_ids=None, out="nsd_bboxes.json", cache="coco_cache"):
    stim_csv = _download(STIM_INFO_URL, os.path.join(cache, "nsd_stim_info_merged.csv"))
    info = pd.read_csv(stim_csv)
    # nsdId in the file is typically 0-based already; align to file row order.
    if "nsdId" not in info.columns:
        info["nsdId"] = np.arange(len(info))

    train_json, val_json = _ensure_coco_annotations(cache)
    coco = {
        "train2017": _load_coco(train_json),
        "val2017": _load_coco(val_json),
    }

    if nsd_ids is None:
        rows = info.itertuples(index=False)
    else:
        idset = set(int(i) for i in nsd_ids)
        rows = info[info["nsdId"].isin(idset)].itertuples(index=False)

    result = {}
    n_done = 0
    for row in rows:
        r = row._asdict()
        nsd_id = int(r["nsdId"])
        coco_id = int(r["cocoId"])
        split = str(r["cocoSplit"])
        cropbox = _parse_cropbox(r["cropBox"])

        anns_map, sizes_map, cats = coco.get(split, ({}, {}, {}))
        W, H = sizes_map.get(coco_id, (None, None))
        boxes = []
        if W is not None:
            for a in anns_map.get(coco_id, []):
                tb = transform_bbox(a["bbox"], W, H, cropbox)
                if tb is not None:
                    boxes.append({"category": cats.get(a["category_id"], "?"),
                                  "bbox_xywh": tb})
        result[nsd_id] = boxes
        n_done += 1
        if n_done % 5000 == 0:
            print(f"  processed {n_done} images")

    with open(out, "w") as f:
        json.dump(result, f)
    print(f"Wrote {out} ({len(result)} images, boxes in 425x425 NSD frame)")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nsd-ids", nargs="+", type=int,
                   help="Specific 0-based NSD ids (default: all 73k)")
    p.add_argument("--out", default="nsd_bboxes.json")
    p.add_argument("--cache", default="coco_cache")
    args = p.parse_args()
    build(nsd_ids=args.nsd_ids, out=args.out, cache=args.cache)


if __name__ == "__main__":
    main()
