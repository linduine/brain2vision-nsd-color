"""
check_shared.py
===============
What IS the held-out set?

check_split.py showed that for subj03 the test shards contain 930 images, that
none of them appear in shared1000.npy, and that train and test share no images
at all. Two readings are possible and they have different consequences:

  (a) The test shards ARE the shared-1000 trials, but shared1000.npy is indexed
      in a different id space than the behav image ids, so the comparison in
      check_split.py compared apples to oranges. Methods is right; one file's
      convention is misread.

  (b) The webdataset's train/test split is NOT the shared-1000 split at all —
      it is a per-participant held-out subset of that participant's own images.
      The evaluation is still clean (no image appears on both sides), but the
      Methods sentence "the held-out test set comprised trials whose image
      belonged to the shared-1000 set viewed by all participants" is wrong, and
      different participants are being tested on DIFFERENT images.

The decisive test is between-participant: the shared-1000 images are by
definition seen by every participant, so if the test shards are the shared set,
two participants' test image sets must nearly coincide. If the split is
per-participant, they will barely overlap.

Run:
    python check_shared.py --subj-a 3 --subj-b 4
"""

import argparse
from collections import Counter

import numpy as np

from brain2vision.roi import REPO_ID
from check_split import scan


def describe_shared1000():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO_ID, "shared1000.npy", repo_type="dataset")
    a = np.load(p)
    print("\n1. WHAT IS IN shared1000.npy?")
    print(f"   shape {a.shape}   dtype {a.dtype}")
    flat = np.asarray(a).ravel()
    print(f"   min {flat.min()}   max {flat.max()}   unique {len(np.unique(flat)):,}")
    print(f"   first 10 values: {flat[:10].tolist()}")
    if a.dtype == bool or set(np.unique(flat).tolist()) <= {0, 1}:
        print("   -> looks like a BOOLEAN MASK over the 73k images, not a list of ids.")
        print("      Any code doing set(load(...).astype(int)) would get {0, 1} and")
        print("      silently mark almost nothing as held out.")
        ids = set(np.flatnonzero(flat).tolist())
        print(f"      as a mask it selects {len(ids):,} images")
    else:
        ids = set(flat.astype(int).tolist())
        print(f"   -> looks like a LIST OF IDS ({len(ids):,} values)")
    return ids


def test_images(subj):
    entries, _ = scan(subj)
    ids = [i for k, _, i in entries if k == "test"]
    return set(ids), Counter(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj-a", type=int, default=3)
    ap.add_argument("--subj-b", type=int, default=4)
    args = ap.parse_args()

    print("=" * 70)
    print("WHAT IS THE HELD-OUT SET?")
    print("=" * 70)

    shared = describe_shared1000()

    a, ca = test_images(args.subj_a)
    b, cb = test_images(args.subj_b)

    print(f"\n2. TEST IMAGES PER PARTICIPANT")
    print(f"   subj{args.subj_a:02d}: {len(a):,} images, "
          f"{sum(ca.values()):,} trials ({sum(ca.values())/max(len(a),1):.2f} per image)")
    print(f"   subj{args.subj_b:02d}: {len(b):,} images, "
          f"{sum(cb.values()):,} trials ({sum(cb.values())/max(len(b),1):.2f} per image)")

    print(f"\n3. DO THE TWO PARTICIPANTS SHARE THEIR TEST IMAGES?")
    inter = a & b
    frac = len(inter) / max(min(len(a), len(b)), 1)
    print(f"   images in common: {len(inter):,}  ({100*frac:.1f}% of the smaller set)")
    if frac > 0.9:
        print("   -> SHARED test set: consistent with the shared-1000 protocol.")
        print("      Reading (a): shared1000.npy is in a different id space.")
    elif frac < 0.1:
        print("   -> DISJOINT test sets: each participant is tested on their own")
        print("      images. Reading (b): the split is NOT the shared-1000 split,")
        print("      and the Methods sentence must be corrected.")
    else:
        print("   -> partial overlap; inspect further before concluding.")

    print(f"\n4. AGAINST shared1000.npy (as interpreted above)")
    for tag, s in ((f"subj{args.subj_a:02d}", a), (f"subj{args.subj_b:02d}", b)):
        print(f"   {tag} test images in shared1000: {len(s & shared):,} / {len(s):,}")
    print(f"   union of both test sets in shared1000: "
          f"{len((a | b) & shared):,} / {len(a | b):,}")

    print("\n5. WHAT THIS DOES NOT CHANGE")
    print("   check_split.py found zero images and zero trials on both sides of")
    print("   the split for subj03. Whatever the held-out set turns out to BE,")
    print("   the evaluation was still performed on unseen images.")


if __name__ == "__main__":
    main()
