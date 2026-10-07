"""
check_split.py
==============
Decisive diagnostic for the train/test split.

check_betas.py established that the behav scan returns MORE trials than the
betas matrix has rows, and that the number of unique (betas row, image id)
pairs equals the number of betas rows exactly. So the surplus is re-read
trials, not extra data. The question that actually matters is WHERE the extra
copies land:

  (a) both copies in the TEST shards
      -> harmless bookkeeping: some held-out trials are counted twice, which
         re-weights images within the test set but changes nothing about
         training. Reported trial counts are wrong; results are not.

  (b) one copy in a TRAIN shard, one in a TEST shard
      -> the same trial is in both training and test data. That is leakage,
         it inflates R², and every decoding number would have to be re-run.

This script answers that directly, and separately checks IMAGE-level leakage:
whether any image id appears in both the train and the test shards, which is
the property the Methods actually claims ("splitting by image ... so there is
no image leakage").

Run:
    python check_split.py [--subj 3]
"""

import argparse
import io
import tarfile
from collections import Counter, defaultdict

import numpy as np

from brain2vision.color_decode import IMG_COL, BETAS_COL, SHARED_COL
from brain2vision.roi import REPO_ID


def scan(subj, all_shards=False):
    """
    Return list of (shard_kind, betas_row, image_id) for every behav entry.

    By default reads the same shards as the analysis pipeline (see
    _select_shard_dirs: every directory present, minus a legacy one whose
    'new_' replacement also exists). all_shards=True reads everything,
    reproducing the pre-fix behaviour.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from brain2vision.color_decode import _select_shard_dirs
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    subj_tag = f"subj{subj:02d}"
    tars = sorted(f for f in files
                  if f.startswith(f"wds/{subj_tag}/") and f.endswith(".tar"))
    if not all_shards:
        keep = _select_shard_dirs(files, subj_tag)
        tars = [f for f in tars
                if any(f.startswith(f"wds/{subj_tag}/{d}/") for d in keep)]
    if not tars:
        raise FileNotFoundError(f"no wds tars for subj{subj:02d}")

    entries = []
    per_shard = Counter()
    flag_vs_dir = Counter()
    for rel in tars:
        local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
        kind = "test" if "test" in rel.split("/")[2].lower() else "train"
        with tarfile.open(local) as tf:
            for m in tf.getmembers():
                base = m.name.rsplit("/", 1)[-1].lower()
                if not base.endswith("behav.npy"):
                    continue
                if "past" in base or "future" in base or "old" in base:
                    continue
                behav = np.atleast_2d(
                    np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True))
                r, i = int(behav[0, BETAS_COL]), int(behav[0, IMG_COL])
                if r < 0 or i < 0:
                    continue
                # Does the documented shared1000 column agree with the directory?
                flag = (int(behav[0, SHARED_COL])
                        if behav.shape[1] > SHARED_COL else -1)
                flag_vs_dir[(kind, flag)] += 1
                entries.append((kind, r, i))
                per_shard[rel] += 1

    print("\n  directory vs documented shared1000 flag (behav column 16):")
    for (kind, flag), n in sorted(flag_vs_dir.items()):
        label = {0: "not shared", 1: "shared1000", -1: "column absent"}.get(flag, str(flag))
        print(f"    {kind:5s} shards, flag={label:14s} {n:,}")
    disagree = flag_vs_dir[("train", 1)] + flag_vs_dir[("test", 0)]
    print(f"    -> the two definitions {'AGREE' if disagree == 0 else f'DISAGREE on {disagree:,} trials'}")
    return entries, per_shard, flag_vs_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=3)
    args = ap.parse_args()

    entries, per_shard, _ = scan(args.subj)
    print("=" * 70)
    print(f"SPLIT DIAGNOSTIC: subj{args.subj:02d}")
    print("=" * 70)

    train = [(r, i) for k, r, i in entries if k == "train"]
    test = [(r, i) for k, r, i in entries if k == "test"]
    print(f"\nbehav entries   total {len(entries):,}   "
          f"train {len(train):,}   test {len(test):,}")
    print(f"unique (row,id) total {len({(r, i) for _, r, i in entries}):,}   "
          f"train {len(set(train)):,}   test {len(set(test)):,}")

    # ---------------------------------------------------------------- 1
    print("\n1. WHERE DO THE DUPLICATE COPIES LAND?")
    where = defaultdict(set)
    for k, r, i in entries:
        where[(r, i)].add(k)
    dup_counts = Counter()
    for k, r, i in entries:
        dup_counts[(r, i)] += 1
    dups = {p for p, c in dup_counts.items() if c > 1}
    both = {p for p in dups if where[p] == {"train", "test"}}
    only_test = {p for p in dups if where[p] == {"test"}}
    only_train = {p for p in dups if where[p] == {"train"}}
    print(f"  duplicated trials              {len(dups):,}")
    print(f"    both copies in TEST shards   {len(only_test):,}")
    print(f"    both copies in TRAIN shards  {len(only_train):,}")
    print(f"    one TRAIN + one TEST         {len(both):,}   <-- leakage if > 0")

    # ---------------------------------------------------------------- 2
    print("\n2. TRIAL-LEVEL LEAKAGE (same betas row on both sides)")
    rows_tr = {r for r, _ in train}
    rows_te = {r for r, _ in test}
    shared_rows = rows_tr & rows_te
    print(f"  betas rows in train {len(rows_tr):,}   in test {len(rows_te):,}")
    print(f"  rows on BOTH sides  {len(shared_rows):,}"
          f"   -> {'CLEAN' if not shared_rows else '**LEAKAGE**'}")

    # ---------------------------------------------------------------- 3
    print("\n3. IMAGE-LEVEL LEAKAGE (same image on both sides)")
    ids_tr = {i for _, i in train}
    ids_te = {i for _, i in test}
    shared_ids = ids_tr & ids_te
    print(f"  images in train {len(ids_tr):,}   in test {len(ids_te):,}")
    print(f"  images on BOTH sides {len(shared_ids):,}"
          f"   -> {'CLEAN' if not shared_ids else '**LEAKAGE**'}")

    # ---------------------------------------------------------------- 4
    print("\n4. DO THE TEST SHARDS CORRESPOND TO THE SHARED-1000 SET?")
    try:
        # shared1000.npy is a length-73,000 BOOLEAN MASK, not an id list.
        from huggingface_hub import hf_hub_download
        a = np.load(hf_hub_download(
            REPO_ID, "shared1000.npy", repo_type="dataset")).ravel()
        shared = (set(np.flatnonzero(a).tolist()) if a.dtype == bool
                  else set(a.astype(int).tolist()))
        print(f"  test images that are shared-1000     "
              f"{len(ids_te & shared):,} / {len(ids_te):,}")
        print(f"  train images that are shared-1000    "
              f"{len(ids_tr & shared):,} / {len(ids_tr):,}"
              f"   -> {'CLEAN' if not (ids_tr & shared) else '**shared images in TRAIN**'}")
    except Exception as e:
        print(f"  skipped ({type(e).__name__}: {e})")

    # ---------------------------------------------------------------- 5
    print("\n5. WHAT THE ANALYSIS ACTUALLY TRAINED AND TESTED ON")
    print(f"  training trials as counted by the pipeline  {len(train):,}")
    print(f"  held-out trials as counted by the pipeline  {len(test):,}")
    print(f"  held-out trials after de-duplication        {len(set(test)):,}")
    print("\n  (NSD design: 750 trials/session; the shared-1000 images are")
    print("   distributed across sessions, so a participant with S sessions")
    print("   has roughly 3000*S/40 shared trials.)")


if __name__ == "__main__":
    main()
