"""
color_decode.py
=================
Decode an image target (color distribution, luminance distribution, ...) from a
chosen ROI's single-trial betas, and evaluate on held-out images.

Pipeline
--------
  ROI betas (per trial)  --RidgeCV/MLP-->  target distribution
                                           (e.g. color_targets.py, luminance_targets.py)

The ROI defaults to V4 but is configurable (`rois=`). For a fair comparison
across ROIs of different sizes, see compare_rois.py / replicate_subjects.py,
which match voxel counts. The decoder itself is target-agnostic via `labels=`.

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
    # 1) make color targets first (see color_targets.py)
    # 2) train (auto-downloads betas, masks, behav, shared1000)
    python -m brain2vision.color_decode --subj 1 --color-targets color_targets.npy \
        --model ridge
"""

import io
import gc
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
# Assemble X (ROI betas) and y (target)
# --------------------------------------------------------------------------- #
def build_xy(subj, color_targets_npy, rois=("V4",), ids_npy=None, betas_npy=None,
             return_ids=False):
    import gc
    color = np.load(color_targets_npy)
    ct_ids = np.load(color_targets_npy.replace(".npy", "_ids.npy"))
    id_to_row = {int(i): k for k, i in enumerate(ct_ids)}

    if ids_npy and betas_npy:                      # raw-NSD path (user-supplied)
        nsd_ids = np.load(ids_npy)
        is_test = _shared1000_mask(nsd_ids)
        keep = np.array([int(i) in id_to_row for i in nsd_ids])   # filter first
        nsd_ids, is_test = nsd_ids[keep], is_test[keep]
        X = np.load(betas_npy, mmap_mode="r")[keep].astype(np.float32, copy=False)
    else:                                           # MindEye2 path
        betas_rows, nsd_ids, is_test = read_behav_alignment(subj)
        # Drop trials whose image has no colour target BEFORE materialising X,
        # so we never build rows we'd only discard (halves peak memory).
        keep = np.array([int(i) in id_to_row for i in nsd_ids])
        betas_rows, nsd_ids, is_test = betas_rows[keep], nsd_ids[keep], is_test[keep]
        from huggingface_hub import hf_hub_download
        import h5py
        roi_mask = load_roi_masks(subj, list(rois))   # boolean over nsdgeneral
        bpath = hf_hub_download(REPO_ID, _betas_filename(subj), repo_type="dataset")
        with h5py.File(bpath, "r") as f:
            dset = f[_first_dataset_key(f)]
            # h5py can reject index arrays on the row axis, so read plain row
            # slices and keep only ROI columns on the in-memory chunk.
            n_all = dset.shape[0]; n_roi = int(roi_mask.sum())
            roi_all = np.empty((n_all, n_roi), dtype=np.float32)
            step = 2000
            for i in range(0, n_all, step):
                roi_all[i:i + step] = dset[i:i + step][:, roi_mask]
        X = roi_all[betas_rows]        # already float32; kept trials only
        del roi_all; gc.collect()      # free the ~1 GB full-ROI matrix now

    y = np.stack([color[id_to_row[int(i)]] for i in nsd_ids]).astype(np.float32)
    print(f"X={X.shape}  y={y.shape}  test={int(is_test.sum())}")
    return (X, y, is_test, nsd_ids) if return_ids else (X, y, is_test)


def _shared1000_mask(nsd_ids):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO_ID, "shared1000.npy", repo_type="dataset")
    shared = set(np.load(p).astype(int).tolist())
    return np.array([int(i) in shared for i in nsd_ids])


# --------------------------------------------------------------------------- #
# Models + evaluation
# --------------------------------------------------------------------------- #
def train_eval(X, y, is_test, model="ridge", alpha=1000.0, labels=None,
               r2_weighting="uniform"):
    # `labels` names the target columns (default: the 11 colors). Passing a
    # different list lets this decode any target, e.g. luminance bins.
    #
    # r2_weighting controls how the per-target R^2 values are pooled into the
    # overall R^2:
    #   "uniform"  -> plain mean (default; but near-absent targets such as the
    #                 "purple" colour bin have ~zero variance, are impossible to
    #                 predict, and their large negative R^2 can dominate the mean);
    #   "variance" -> variance-weighted (sklearn), so a colour contributes in
    #                 proportion to how much it actually varies -> rare, near-
    #                 constant colours barely count. Much more stable on small /
    #                 selected image subsets.
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
        reg = RidgeCV(alphas=np.logspace(1, 6, 12)).fit(Xtr, ytr)
        pred = reg.predict(Xte)
        print(f"RidgeCV selected alpha = {float(np.atleast_1d(reg.alpha_)[0]):.1f}")
    elif model == "elasticnet":
        # Sparse LINEAR decoder: shrinks AND zeros out voxels. Robustness check
        # within linear models — does a different regularizer give the same answer?
        import inspect
        from sklearn.linear_model import MultiTaskElasticNetCV
        kw = dict(l1_ratio=0.5, cv=3, max_iter=3000, n_jobs=1)  # n_jobs=1: avoid per-core data copies
        # 'n_alphas' was renamed to 'alphas' (now accepts an int) in recent sklearn
        params = inspect.signature(MultiTaskElasticNetCV).parameters
        kw["n_alphas" if "n_alphas" in params else "alphas"] = 8
        reg = MultiTaskElasticNetCV(**kw).fit(Xtr, ytr)
        pred = reg.predict(Xte)
    elif model == "kernel":
        # NONLINEAR via an RBF-kernel approximation (Nystroem) + ridge. The
        # Nystroem features scale to many trials where an exact kernel matrix
        # (O(n^2)) would not, while still capturing curved relationships a linear
        # decoder misses.
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import RidgeCV
        ny = Nystroem(kernel="rbf", gamma=1.0 / Xtr.shape[1],
                      n_components=min(500, Xtr.shape[0]), random_state=0)
        Ztr = ny.fit_transform(Xtr); Zte = ny.transform(Xte)
        reg = RidgeCV(alphas=np.logspace(1, 6, 12)).fit(Ztr, ytr)
        pred = reg.predict(Zte)
    elif model == "svr":
        # Linear Support Vector Regression -- the canonical MVPA decoder. Like
        # ridge it is LINEAR with L2 regularization, but uses an epsilon-
        # insensitive loss instead of squared error. Included because a linear
        # SVM is the most widely cited MVPA decoder; expect it to track ridge
        # closely (both are L2-linear). One SVR per colour (multi-output); the
        # penalty C is picked on an internal validation split.
        from sklearn.svm import LinearSVR
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score as _r2

        def _svr(C):
            return MultiOutputRegressor(
                LinearSVR(C=C, loss="squared_epsilon_insensitive", dual=False, max_iter=20000, tol=1e-4, random_state=0))

        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0)
        best_C, best_s = None, -np.inf
        for C in (0.01, 0.1, 1.0, 10.0):
            s = _r2(yv, _svr(C).fit(Xa, ya).predict(Xv),
                    multioutput="variance_weighted")
            if s > best_s:
                best_s, best_C = s, C
        reg = _svr(best_C).fit(Xtr, ytr)               # refit on full train split
        pred = reg.predict(Xte)
        print(f"LinearSVR selected C = {best_C} (val R2 = {best_s:+.3f})")
    elif model == "mlp":
        # NONLINEAR: a small, REGULARIZED neural net. An unregularized MLP
        # overfits badly on a few hundred high-dimensional trials (that is why
        # the (256,)/default-penalty version scored a large negative R2). The
        # "fair" MLP here (a) uses a small hidden layer, (b) picks the L2 penalty
        # alpha on an internal validation split, and (c) uses early stopping with
        # patience. If even this cannot beat predicting the mean, that is
        # evidence against exploitable nonlinear structure -- not an untuned-model
        # artifact.
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score as _r2

        def _mlp(a):
            return MLPRegressor(hidden_layer_sizes=(64,), alpha=a,
                                max_iter=800, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=20,
                                random_state=0)

        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0)
        best_a, best_s = None, -np.inf
        for a in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            s = _r2(yv, _mlp(a).fit(Xa, ya).predict(Xv),
                    multioutput="variance_weighted")   # robust to near-absent bins
            if s > best_s:
                best_s, best_a = s, a
        reg = _mlp(best_a).fit(Xtr, ytr)               # refit on full train split
        pred = reg.predict(Xte)
        print(f"MLP selected alpha = {best_a} (val R2 = {best_s:+.3f})")
    else:
        raise ValueError(f"unknown model '{model}' "
                         "(ridge | elasticnet | kernel | mlp)")

    # metrics
    from sklearn.metrics import r2_score
    r2 = r2_score(yte, pred, multioutput="raw_values")
    mo = "variance_weighted" if r2_weighting == "variance" else "uniform_average"
    overall_r2 = float(r2_score(yte, pred, multioutput=mo))
    top1 = (pred.argmax(1) == yte.argmax(1)).mean()   # dominant-bin accuracy

    print(f"\n=== decode ({model}, R2 pooling={r2_weighting}) ===")
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
    p.add_argument("--model",
                   choices=["ridge", "elasticnet", "kernel", "svr", "mlp"],
                   default="ridge")
    p.add_argument("--alpha", type=float, default=1000.0)
    p.add_argument("--ids-npy", help="(raw-NSD path) nsd_id per betas row")
    p.add_argument("--betas-npy", help="(raw-NSD path) V4 betas array")
    args = p.parse_args()

    X, y, is_test = build_xy(args.subj, args.color_targets,
                             ids_npy=args.ids_npy, betas_npy=args.betas_npy)
    train_eval(X, y, is_test, model=args.model, alpha=args.alpha)


if __name__ == "__main__":
    main()
