"""
train_v4_color.py
=================
First experiment: predict the COLOR content of the seen image from V4 betas.

Pipeline
--------
  V4 betas (per trial)  --ridge/MLP-->  11-way basic-color distribution
                                        (from extract_color_targets.py)

V4 is the natural ROI for this: it is strongly color-selective. We predict the
soft color histogram (red/orange/.../white/gray) and evaluate how well V4
activity tracks the colors present.

Data alignment (IMPORTANT)
--------------------------
The MindEye2 betas (betas_all_subjXX_fp32_renorm.hdf5) are NOT stored in image
order. Each trial's row index and its 73k image id come from the `behav` arrays
in the webdataset:
    wds/subjXX/{new_train,new_test}/*.tar   ->   *.behav.npy
    behav[0, IMG_COL]   = 73k image id (== NSD id, 0-based)   [default col 0]
    behav[0, BETAS_COL] = row index into betas_all             [default col 5]
These default columns follow the MindEye2 dataloader. VERIFY once against the
MindEye2 training notebook / your data: after loading, ids must be in
[0, 73000) and rows in [0, n_betas_rows). A quick check is printed.

Alternatively (raw-NSD path) pass --ids-npy / --betas-npy directly if you built
the alignment yourself deterministically from nsd_stim_info.

Split
-----
Test set = trials whose image is in the shared-1000 set (shared1000.npy from
pscotti/mindeyev2). Train = everything else. This mirrors the standard NSD
held-out protocol and avoids image leakage between train and test.

Usage
-----
    pip install scikit-learn h5py numpy huggingface_hub
    # 1) make color targets first (see extract_color_targets.py)
    # 2) train (auto-downloads betas, masks, behav, shared1000)
    python train_v4_color.py --subj 1 --color-targets color_targets.npy \
        --model ridge
"""

import os
import io
import glob
import tarfile
import argparse
import numpy as np

from brain2vision.roi import load_roi_masks, _betas_filename, _first_dataset_key, REPO_ID

IMG_COL = 0      # behav column holding the 73k image id  (verify!)
BETAS_COL = 5    # behav column holding the betas row index (verify!)


# --------------------------------------------------------------------------- #
# Alignment: read behav from the MindEye2 webdataset
# --------------------------------------------------------------------------- #
def read_behav_alignment(subj, cache_dir=None):
    """
    Return (betas_rows, nsd_ids, is_test) arrays, one entry per trial, by
    scanning the subject's webdataset behav files on Hugging Face.
    """
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    subj_tag = f"subj{subj:02d}"
    tars = [f for f in files
            if f.startswith(f"wds/{subj_tag}/") and f.endswith(".tar")]
    if not tars:
        raise FileNotFoundError(
            f"No wds tars for {subj_tag}; check repo layout with list_repo_files.")

    rows, ids, test = [], [], []
    for rel in sorted(tars):
        local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
        is_test = "test" in rel.lower()
        with tarfile.open(local) as tf:
            for m in tf.getmembers():
                base = m.name.rsplit("/", 1)[-1].lower()
                # Keep ONLY the current-trial behav. Each sample also stores
                # past_/future_/old_ behav (neighbouring trials, padded with -1);
                # those all end in "behav.npy" too, so exclude them explicitly.
                if not base.endswith("behav.npy"):
                    continue
                if "past" in base or "future" in base or "old" in base:
                    continue
                behav = np.load(io.BytesIO(tf.extractfile(m).read()),
                                allow_pickle=True)
                behav = np.atleast_2d(behav)
                ids.append(int(behav[0, IMG_COL]))
                rows.append(int(behav[0, BETAS_COL]))
                test.append(is_test)
    rows = np.asarray(rows); ids = np.asarray(ids); test = np.asarray(test)
    valid = (rows >= 0) & (ids >= 0)          # drop any residual -1 padding
    rows, ids, test = rows[valid], ids[valid], test[valid]
    print(f"Alignment: {len(rows)} trials | id range [{ids.min()},{ids.max()}] "
          f"| row range [{rows.min()},{rows.max()}] | test trials {test.sum()}")
    if ids.max() >= 73000 or rows.min() < 0:
        print("  !! id/row ranges look off -- re-check IMG_COL/BETAS_COL.")
    return rows, ids, test


# --------------------------------------------------------------------------- #
# Assemble X (V4 betas) and y (color targets)
# --------------------------------------------------------------------------- #
def build_xy(subj, color_targets_npy, rois=("V4",), ids_npy=None, betas_npy=None):
    color = np.load(color_targets_npy)
    ct_ids = np.load(color_targets_npy.replace(".npy", "_ids.npy"))
    id_to_row = {int(i): k for k, i in enumerate(ct_ids)}

    if ids_npy and betas_npy:                      # raw-NSD path (user-supplied)
        nsd_ids = np.load(ids_npy)
        X = np.load(betas_npy).astype(np.float32)
        is_test = _shared1000_mask(nsd_ids)
        betas_rows = np.arange(len(nsd_ids))
    else:                                           # MindEye2 path
        betas_rows, nsd_ids, is_test = read_behav_alignment(subj)
        from huggingface_hub import hf_hub_download
        import h5py
        roi_mask = load_roi_masks(subj, list(rois))  # boolean over nsdgeneral
        bpath = hf_hub_download(REPO_ID, _betas_filename(subj), repo_type="dataset")
        with h5py.File(bpath, "r") as f:
            dset = f[_first_dataset_key(f)]
            # Some h5py builds reject ANY index array on the row axis (even a
            # sorted, unique one). So we pass h5py ONLY plain slices: read the
            # dataset in row-chunks, keep just the ROI columns (NumPy boolean
            # index on the in-memory chunk), and assemble the full ROI matrix.
            n_all = dset.shape[0]
            n_roi = int(roi_mask.sum())
            roi_all = np.empty((n_all, n_roi), dtype=np.float32)
            step = 2000
            for i in range(0, n_all, step):
                chunk = dset[i:i + step]              # plain slice -> always OK
                roi_all[i:i + step] = chunk[:, roi_mask]
        # all fancy selection happens in NumPy, where any order/duplicates are fine
        X = roi_all[betas_rows].astype(np.float32)

    # y from color targets, dropping trials whose image lacks a target
    keep = np.array([i in id_to_row for i in nsd_ids])
    X, nsd_ids, is_test = X[keep], nsd_ids[keep], is_test[keep]
    y = np.stack([color[id_to_row[int(i)]] for i in nsd_ids])
    print(f"X={X.shape}  y={y.shape}  test={int(is_test.sum())}")
    return X, y, is_test


def _shared1000_mask(nsd_ids):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO_ID, "shared1000.npy", repo_type="dataset")
    shared = set(np.load(p).astype(int).tolist())
    return np.array([int(i) in shared for i in nsd_ids])


# --------------------------------------------------------------------------- #
# Models + evaluation
# --------------------------------------------------------------------------- #
def train_eval(X, y, is_test, model="ridge", alpha=1000.0, labels=None):
    # `labels` names the target columns (default: the 11 colors). Passing a
    # different list lets this decode any target, e.g. luminance bins.
    if labels is None:
        from brain2vision.color_targets import COLOR_NAMES
        labels = COLOR_NAMES
    Xtr, ytr = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]

    # z-score voxels on train stats
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    if model == "ridge":
        # Tune alpha per call via efficient leave-one-out CV, so ROIs with very
        # different voxel counts each get appropriate regularization. A fixed
        # alpha over-penalizes small ROIs and under-penalizes large ones, which
        # is a dimensionality confound when comparing ROIs of different sizes.
        from sklearn.linear_model import RidgeCV
        alphas = np.logspace(1, 6, 12)
        reg = RidgeCV(alphas=alphas).fit(Xtr, ytr)
        pred = reg.predict(Xte)
        print(f"RidgeCV selected alpha = {float(np.atleast_1d(reg.alpha_)[0]):.1f}")
    elif model == "mlp":
        from sklearn.neural_network import MLPRegressor
        reg = MLPRegressor(hidden_layer_sizes=(256,), max_iter=200).fit(Xtr, ytr)
        pred = reg.predict(Xte)
    else:
        raise ValueError(model)

    # metrics
    from sklearn.metrics import r2_score
    r2 = r2_score(yte, pred, multioutput="raw_values")
    overall_r2 = r2_score(yte, pred)
    top1 = (pred.argmax(1) == yte.argmax(1)).mean()   # dominant-bin accuracy

    print(f"\n=== decode ({model}) ===")
    print(f"overall R^2 : {overall_r2:.3f}")
    print(f"dominant-bin top-1 acc : {top1:.3f} "
          f"(chance ~= {1/len(labels):.3f})")
    print("per-target R^2:")
    for name, val in zip(labels, r2):
        print(f"   {name:8s} {val:+.3f}")
    return {"overall_r2": overall_r2, "top1": top1,
            "per_color_r2": dict(zip(labels, r2.tolist()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subj", type=int, default=1)
    p.add_argument("--color-targets", required=True)
    p.add_argument("--model", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--alpha", type=float, default=1000.0)
    p.add_argument("--ids-npy", help="(raw-NSD path) nsd_id per betas row")
    p.add_argument("--betas-npy", help="(raw-NSD path) V4 betas array")
    args = p.parse_args()

    X, y, is_test = build_xy(args.subj, args.color_targets,
                             ids_npy=args.ids_npy, betas_npy=args.betas_npy)
    train_eval(X, y, is_test, model=args.model, alpha=args.alpha)


if __name__ == "__main__":
    main()
