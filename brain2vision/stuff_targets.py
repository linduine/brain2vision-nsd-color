"""
stuff_targets.py
================
Build a per-image STUFF feature: the presence of each COCO-Stuff category
(91-dim binary vector, "what amorphous background material is in this image").

Why this exists
---------------
`semantic_targets.py` encodes the 80 COCO *thing* categories: discrete objects,
from `person` to `toothbrush`. It contains no sky, no grass, no road, no water,
no wall. But the manuscript motivates residualisation with exactly those:

    "skies are blue, foliage green, soil and wood brown"

None of those three is a COCO thing category, so the existing residualisation
cannot remove the correlations used to justify it. COCO-Stuff supplies them:
`sky-other`, `clouds`, `grass`, `tree`, `bush`, `leaves`, `dirt`, `mud`, `sand`,
`wood`, `sea`, `river`, `snow`, `road`, and the wall/floor/ceiling families.

Encoding is binary presence, deliberately matching the thing vector, so that the
three residualisations differ only in *which content* is removed and not in how
it is represented. Binary is the weaker encoding, area fraction would carry more
information, which makes any collapse it produces a lower bound.

Three residualisations follow from this:
  * things only  (80-dim, the existing analysis)
  * stuff only   (91-dim, this module)
  * things+stuff (171-dim, `--combine`)

Prediction worth stating before looking: if the decodable colour signal is
content-predictable, thing-residualisation should hit higher visual cortex hardest,
while stuff-residualisation should hit early, retinotopic, peripherally-weighted
cortex hardest, because backgrounds are mostly stuff, and the foreground/
background control already shows early cortex decodes backgrounds best
(R² = 0.033 vs 0.018).

Memory note
-----------
`stuff_train2017.json` carries per-annotation segmentations and is large; a plain
`json.load` of it needs several GB of RAM. Only category presence is required
here, so if `ijson` is installed this module streams the file instead
(`pip install ijson`). Without it, it falls back to `json.load` and says so.

Usage
-----
    python -m brain2vision.stuff_targets --out data/stuff_targets.npy
    python -m brain2vision.stuff_targets --combine data/semantic_targets.npy \
        --out data/thingstuff_targets.npy
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile

import numpy as np

from brain2vision.bboxes import _download, STIM_INFO_URL

COCO_STUFF_ZIP = ("http://images.cocodataset.org/annotations/"
                  "stuff_annotations_trainval2017.zip")

# COCO-Stuff reserves id 183 / "other" as a catch-all for pixels that are stuff
# but not any named class. It is excluded: as a regressor it would absorb
# unrelated content and is not a semantic category in the sense we want.
EXCLUDE_NAMES = {"other"}


def _ensure_stuff_annotations(cache="coco_cache"):
    """Download + unzip the COCO-Stuff annotation jsons if not present."""
    os.makedirs(cache, exist_ok=True)
    train = os.path.join(cache, "annotations", "stuff_train2017.json")
    val = os.path.join(cache, "annotations", "stuff_val2017.json")
    if os.path.exists(train) and os.path.exists(val):
        return train, val
    zip_path = _download(COCO_STUFF_ZIP, os.path.join(cache, "stuff_ann.zip"))
    print("unzipping COCO-Stuff annotations (this takes a minute)...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache)
    if not (os.path.exists(train) and os.path.exists(val)):
        raise FileNotFoundError(
            f"expected {train} and {val} after unzipping {zip_path}; "
            f"found: {os.listdir(os.path.join(cache, 'annotations'))}")
    return train, val


def _presence_and_categories(json_path):
    """Return (imgId -> set(category_id), {category_id: name}).

    Streams with ijson when available; only category ids per image are kept, so
    the segmentations never need to be materialised.
    """
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None

    if ijson is None:
        print(f"  [ijson not installed -> loading {os.path.basename(json_path)} "
              f"whole; expect several GB of RAM. `pip install ijson` to stream.]")
        with open(json_path) as f:
            data = json.load(f)
        cats = {c["id"]: c["name"] for c in data["categories"]}
        pres: dict[int, set] = {}
        for a in data["annotations"]:
            pres.setdefault(a["image_id"], set()).add(a["category_id"])
        return pres, cats

    cats = {}
    with open(json_path, "rb") as f:
        for c in ijson.items(f, "categories.item"):
            cats[int(c["id"])] = c["name"]
    pres = {}
    with open(json_path, "rb") as f:
        for a in ijson.items(f, "annotations.item"):
            pres.setdefault(int(a["image_id"]), set()).add(int(a["category_id"]))
    return pres, cats


def run(out, nsd_ids=None, cache="coco_cache", combine=None):
    import pandas as pd

    stim = _download(STIM_INFO_URL, os.path.join(cache, "nsd_stim_info_merged.csv"))
    info = pd.read_csv(stim)
    if "nsdId" not in info.columns:
        info["nsdId"] = np.arange(len(info))

    train_json, val_json = _ensure_stuff_annotations(cache)
    print("reading stuff annotations...")
    pres_tr, cats_tr = _presence_and_categories(train_json)
    pres_va, cats_va = _presence_and_categories(val_json)
    presence = {"train2017": pres_tr, "val2017": pres_va}

    cats = dict(cats_tr); cats.update(cats_va)
    keep_ids = sorted(cid for cid, name in cats.items()
                      if name not in EXCLUDE_NAMES)
    dropped = sorted(name for cid, name in cats.items() if name in EXCLUDE_NAMES)
    col = {cid: i for i, cid in enumerate(keep_ids)}
    names = [cats[cid] for cid in keep_ids]
    print(f"  {len(keep_ids)} stuff categories kept"
          + (f"; dropped catch-all: {dropped}" if dropped else ""))

    rows = (info if nsd_ids is None
            else info[info["nsdId"].isin(set(int(i) for i in nsd_ids))])
    ids, vecs, missing = [], [], 0
    for r in rows.itertuples(index=False):
        d = r._asdict()
        nid, cid, split = int(d["nsdId"]), int(d["cocoId"]), str(d["cocoSplit"])
        per_img = presence.get(split, {})
        v = np.zeros(len(keep_ids), dtype=np.float32)
        anns = per_img.get(cid)
        if anns is None:
            missing += 1
        else:
            for a in anns:
                if a in col:
                    v[col[a]] = 1.0
        ids.append(nid); vecs.append(v)

    X = np.asarray(vecs, dtype=np.float32)
    ids = np.asarray(ids)
    print(f"  built {X.shape} stuff-presence matrix; "
          f"{missing} images had no stuff annotation "
          f"({100 * missing / max(len(ids), 1):.1f}%)")
    print(f"  mean categories per image: {X.sum(1).mean():.2f}")

    if combine:
        T = np.load(combine)
        tid = np.load(combine.replace(".npy", "_ids.npy"))
        if not np.array_equal(tid, ids):
            order = {int(v): i for i, v in enumerate(tid)}
            sel = np.array([order[int(v)] for v in ids])
            T = T[sel]
        X = np.concatenate([T, X], axis=1)
        tnames = [l.strip() for l in
                  open(combine.replace(".npy", "_categories.txt")) if l.strip()]
        names = tnames + names
        print(f"  combined with {combine}: {X.shape}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, X)
    np.save(out.replace(".npy", "_ids.npy"), ids)
    with open(out.replace(".npy", "_categories.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"Saved {out} shape={X.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/stuff_targets.npy")
    p.add_argument("--cache", default="coco_cache")
    p.add_argument("--nsd-ids", nargs="+", type=int)
    p.add_argument("--combine", default=None,
                   help="path to semantic_targets.npy; concatenate things+stuff")
    a = p.parse_args()
    run(a.out, nsd_ids=a.nsd_ids, cache=a.cache, combine=a.combine)


if __name__ == "__main__":
    main()
