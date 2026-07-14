"""
visualize_bbox.py
=================
Sanity-check the COCO -> NSD-stimulus-frame bounding boxes produced by
build_coco_bboxes.py by drawing them on the actual stimulus image.

If the boxes line up with the objects, the cropBox transform is correct. If
they're shifted/scaled, the cropBox convention needs flipping (see notes in
build_coco_bboxes.py).

Usage
-----
    pip install h5py numpy pillow
    python visualize_bbox.py --images coco_images_224_float16.hdf5 \
        --bboxes nsd_bboxes.json --nsd-id 3 --out check_3.png
"""

import json
import argparse
import numpy as np
import h5py
from PIL import Image, ImageDraw

STIM_SIZE = 425  # frame the boxes are expressed in (build_coco_bboxes output)


def _load_image(h5path, nsd_id):
    with h5py.File(h5path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        arr = np.asarray(f[key][nsd_id])
    if arr.ndim == 3 and arr.shape[0] == 3:      # (3,H,W) -> (H,W,3)
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.001:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def visualize(h5path, bbox_json, nsd_id, out):
    img = _load_image(h5path, nsd_id)
    # boxes are in 425 coords; scale to whatever resolution the image is.
    scale = img.width / STIM_SIZE
    with open(bbox_json) as f:
        boxes = json.load(f).get(str(nsd_id), [])

    draw = ImageDraw.Draw(img)
    for b in boxes:
        x, y, w, h = [v * scale for v in b["bbox_xywh"]]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
        draw.text((x + 2, y + 2), b["category"], fill=(255, 255, 0))

    img.save(out)
    print(f"nsd_id={nsd_id}: drew {len(boxes)} boxes -> {out}")
    for b in boxes:
        print(f"   {b['category']:15s} {b['bbox_xywh']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="HDF5 of NSD images")
    p.add_argument("--bboxes", required=True, help="JSON from build_coco_bboxes.py")
    p.add_argument("--nsd-id", type=int, required=True)
    p.add_argument("--out", default="bbox_check.png")
    args = p.parse_args()
    visualize(args.images, args.bboxes, args.nsd_id, args.out)


if __name__ == "__main__":
    main()
