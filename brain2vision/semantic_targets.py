"""
semantic_targets.py
===================
Build a per-image SEMANTIC feature that is colour-free by construction: the
presence of each COCO object category (80-dim binary vector, "which objects are
in this image", regardless of their colour).

This is the semantic axis for the chromatic-vs-semantic variance partitioning
(semantic_residual.py). It must be colour-free: CLIP *image* embeddings would
leak colour (CLIP sees the pixels), so residualising colour against them would
remove the brain's chromatic signal too. COCO category presence is about *what
is there*, not what colour it is, so it isolates semantics cleanly.

(A colour-light alternative is CLIP *text* embeddings of the COCO captions, the
clip_targets module, `--captions`, since captions rarely mention colour.)

Reuses the COCO-annotation loading from bboxes.py.

Usage
-----
    pip install pandas numpy requests
    python -m brain2vision.semantic_targets --out data/semantic_targets.npy
Outputs semantic_targets.npy (n_images, 80), _ids.npy, and _categories.txt.
"""

import os
import argparse
import numpy as np

from brain2vision.bboxes import (_download, _ensure_coco_annotations, _load_coco,
                                 STIM_INFO_URL)


def run(out, nsd_ids=None, cache="coco_cache"):
    import pandas as pd
    stim = _download(STIM_INFO_URL, os.path.join(cache, "nsd_stim_info_merged.csv"))
    info = pd.read_csv(stim)
    if "nsdId" not in info.columns:
        info["nsdId"] = np.arange(len(info))

    train_json, val_json = _ensure_coco_annotations(cache)
    coco = {"train2017": _load_coco(train_json), "val2017": _load_coco(val_json)}

    # fixed category_id -> column mapping (80 COCO categories, non-contiguous ids)
    cats = coco["train2017"][2]                      # id -> name
    cat_ids = sorted(cats.keys())
    col = {cid: i for i, cid in enumerate(cat_ids)}
    ncat = len(cat_ids)

    rows = (info if nsd_ids is None
            else info[info["nsdId"].isin(set(int(i) for i in nsd_ids))])
    ids, vecs = [], []
    for r in rows.itertuples(index=False):
        d = r._asdict()
        nid, cid, split = int(d["nsdId"]), int(d["cocoId"]), str(d["cocoSplit"])
        anns_map, _, _ = coco.get(split, ({}, {}, {}))
        v = np.zeros(ncat, dtype=np.float32)
        for a in anns_map.get(cid, []):
            if a["category_id"] in col:
                v[col[a["category_id"]]] = 1.0      # presence (binary)
        ids.append(nid); vecs.append(v)
        if len(ids) % 5000 == 0:
            print(f"  processed {len(ids)} images")

    V = np.asarray(vecs, dtype=np.float32); ids = np.asarray(ids)
    np.save(out, V); np.save(out.replace(".npy", "_ids.npy"), ids)
    with open(out.replace(".npy", "_categories.txt"), "w") as f:
        f.write("\n".join(cats[c] for c in cat_ids))
    print(f"Saved {out} shape={V.shape} ({ncat} COCO categories, colour-free)")
    print(f"  mean categories/image: {V.sum(1).mean():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/semantic_targets.npy")
    p.add_argument("--nsd-ids", nargs="+", type=int)
    p.add_argument("--cache", default="coco_cache")
    args = p.parse_args()
    run(args.out, nsd_ids=args.nsd_ids, cache=args.cache)


if __name__ == "__main__":
    main()
