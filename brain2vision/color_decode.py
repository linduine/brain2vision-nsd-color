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

# Behav column layout, documented in the MindEye2 dataset README
# (https://huggingface.co/datasets/pscotti/mindeyev2, README.md):
#   0  cocoidx  = 73KID - 1   (0-based index into the 73k NSD images)
#   5  global_trial              (used by the MindEye2 dataloader to index betas)
#  16  shared1000                (1 if this trial's image is in the shared-1000 set)
IMG_COL = 0
BETAS_COL = 5
SHARED_COL = 16

# The README also warns:
#   'Always use "new_test" instead of "test" in the wds folders, "test" refers
#    to using the old NSD data from before they released the full set of
#    scanning sessions.'
# Matching on the substring "test" therefore picks up the LEGACY shards as well
# as the current ones, and reads those trials twice. But the repo does not
# necessarily ship BOTH a "train" and a "new_train": whichever names are
# actually present must be discovered, not assumed. Hard-coding
# ("new_train", "new_test") selects only held-out shards when the training
# directory happens to be called "train", leaving zero training trials.
LEGACY_PAIRS = {"test": "new_test", "train": "new_train"}


def _select_shard_dirs(files, subj_tag, verbose=True):
    """
    Return the webdataset directory names to read for one participant.

    Keeps every directory present, except a legacy directory whose superseding
    'new_' counterpart also exists (per the MindEye2 README).
    """
    dirs = sorted({f.split("/")[2] for f in files
                   if f.startswith(f"wds/{subj_tag}/") and f.endswith(".tar")
                   and len(f.split("/")) > 3})
    drop = {old for old, new in LEGACY_PAIRS.items() if old in dirs and new in dirs}
    keep = [d for d in dirs if d not in drop]
    if verbose:
        print(f"  wds dirs for {subj_tag}: found {dirs}")
        if drop:
            print(f"    dropping superseded {sorted(drop)} "
                  f"(the 'new_' versions are present)")
        print(f"    reading {keep}")
    return keep


# --------------------------------------------------------------------------- #
# Alignment: read behav from the MindEye2 webdataset
# --------------------------------------------------------------------------- #
def read_behav_alignment(subj, cache_dir=None, legacy_shards=False):
    """
    Return (betas_rows, nsd_ids, is_test) arrays, one entry per trial, by
    scanning the subject's webdataset behav files on Hugging Face.

    Superseded shard directories are skipped (see _select_shard_dirs). The
    held-out flag is taken from the behav shared1000 column rather than from
    the shard name, so the split is defined by the data, not by a filename.

    legacy_shards=True restores the old behaviour (scan every tar, flag by
    filename) for reproducing pre-fix results.
    """
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    subj_tag = f"subj{subj:02d}"
    tars = [f for f in files
            if f.startswith(f"wds/{subj_tag}/") and f.endswith(".tar")]
    if not legacy_shards:
        keep_dirs = _select_shard_dirs(files, subj_tag)
        tars = [f for f in tars
                if any(f.startswith(f"wds/{subj_tag}/{d}/") for d in keep_dirs)]
    if not tars:
        raise FileNotFoundError(
            f"No wds tars for {subj_tag}; check repo layout with list_repo_files.")

    rows, ids, test = [], [], []
    for rel in sorted(tars):
        local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
        shard_is_test = "test" in rel.rsplit("/", 2)[-2].lower()
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
                # Prefer the documented shared1000 flag; fall back to the shard
                # name only if the column is missing or padded with -1.
                flag = (int(behav[0, SHARED_COL])
                        if behav.shape[1] > SHARED_COL else -1)
                test.append(bool(flag == 1) if flag in (0, 1) else shard_is_test)
    rows = np.asarray(rows); ids = np.asarray(ids); test = np.asarray(test)
    valid = (rows >= 0) & (ids >= 0)          # drop any residual -1 padding
    rows, ids, test = rows[valid], ids[valid], test[valid]

    n_dup = len(rows) - len({(int(r), int(i)) for r, i in zip(rows, ids)})
    n_test, n_train = int(test.sum()), int((~test).sum())
    print(f"Alignment: {len(rows)} trials | id range [{ids.min()},{ids.max()}] "
          f"| row range [{rows.min()},{rows.max()}] | test trials {n_test}")
    print(f"  shards: {len(tars)} tars | train trials {n_train} "
          f"| duplicate (row,id) reads: {n_dup}")
    if n_dup:
        print("  !! trials read more than once -- check the shard selection.")
    if ids.max() >= 73000 or rows.min() < 0:
        print("  !! id/row ranges look off -- re-check IMG_COL/BETAS_COL.")

    # Fail here, with a diagnosis, rather than several frames deep in sklearn.
    if n_train == 0 or n_test == 0:
        raise RuntimeError(
            f"subj{subj:02d}: the split has {n_train} training and {n_test} "
            f"held-out trials, one side is empty, so nothing can be fitted.\n"
            f"  shards read: {sorted({t.split('/')[2] for t in tars})}\n"
            f"  Every trial was classified the same way. Check that the shard "
            f"selection includes a training directory, and that behav column "
            f"{SHARED_COL} is the shared1000 flag in this release.")
    return rows, ids, test


# --------------------------------------------------------------------------- #
# Assemble X (ROI betas) and y (target)
# --------------------------------------------------------------------------- #
# Callers that decode several ROIs for the same participant (replicate_subjects,
# reliability) previously called build_xy once per ROI, which re-scanned the
# webdataset and re-read the whole betas file every time, three times the I/O
# for the same bytes. build_xy_multi does that work once per participant; the
# alignment is additionally memoised because it is small and expensive to fetch.
_ALIGN_CACHE = {}


def _alignment(subj):
    """read_behav_alignment, memoised per participant within a process."""
    if subj not in _ALIGN_CACHE:
        _ALIGN_CACHE[subj] = read_behav_alignment(subj)
    else:
        rows, ids, test = _ALIGN_CACHE[subj]
        print(f"Alignment: reusing cached scan for subj{subj:02d} "
              f"({len(rows):,} trials, {int(test.sum()):,} held out)")
    return _ALIGN_CACHE[subj]


def _targets(color_targets_npy):
    color = np.load(color_targets_npy)
    ct_ids = np.load(color_targets_npy.replace(".npy", "_ids.npy"))
    return color, {int(i): k for k, i in enumerate(ct_ids)}


def build_xy_multi(subj, color_targets_npy, roi_sets, return_ids=False):
    """
    Build (X, y, is_test) for SEVERAL ROIs of one participant, reading the
    alignment and the betas file once.

    roi_sets : dict {name: [region fragments]}, e.g. ROI_SETS from roi.py.
    Returns  : dict {name: (X, y, is_test[, ids])}

    Memory note: the betas are read once into the UNION of the requested ROI
    masks, then each ROI is a column slice of that. Peak usage is therefore the
    union matrix plus one ROI matrix, rather than one ROI matrix at a time, a
    modest increase in exchange for reading the file once instead of N times.
    """
    import gc
    import h5py
    from huggingface_hub import hf_hub_download

    color, id_to_row = _targets(color_targets_npy)
    betas_rows, nsd_ids, is_test = _alignment(subj)
    keep = np.array([int(i) in id_to_row for i in nsd_ids])
    betas_rows, nsd_ids, is_test = betas_rows[keep], nsd_ids[keep], is_test[keep]

    masks = {name: load_roi_masks(subj, list(frags))
             for name, frags in roi_sets.items()}
    union = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        union |= m
    # column index of each ROI's voxels within the union matrix
    union_idx = np.flatnonzero(union)
    pos = {v: k for k, v in enumerate(union_idx)}
    cols = {name: np.array([pos[v] for v in np.flatnonzero(m)], dtype=int)
            for name, m in masks.items()}

    bpath = hf_hub_download(REPO_ID, _betas_filename(subj), repo_type="dataset")
    with h5py.File(bpath, "r") as f:
        dset = f[_first_dataset_key(f)]
        n_all = dset.shape[0]
        allvox = np.empty((n_all, union.sum()), dtype=np.float32)
        step = 2000
        for i in range(0, n_all, step):
            allvox[i:i + step] = dset[i:i + step][:, union]
    print(f"  read betas once: {n_all:,} x {int(union.sum()):,} voxels "
          f"(union of {', '.join(roi_sets)})")

    y = np.stack([color[id_to_row[int(i)]] for i in nsd_ids]).astype(np.float32)

    out = {}
    for name, c in cols.items():
        X = allvox[np.ix_(betas_rows, c)]
        print(f"  {name}: X={X.shape}  y={y.shape}  test={int(is_test.sum())}")
        out[name] = (X, y, is_test, nsd_ids) if return_ids else (X, y, is_test)
    del allvox
    gc.collect()
    return out


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
        betas_rows, nsd_ids, is_test = _alignment(subj)
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
    """
    Boolean mask: which of `nsd_ids` are in the shared-1000 set?

    NOTE ON THE FILE FORMAT. shared1000.npy is a length-73,000 BOOLEAN MASK over
    the 73k images, not a list of 1,000 image ids. Reading it as a list of ids
    (set(load(...).astype(int))) yields {0, 1} and silently marks essentially
    nothing as held out. The repository also ships test_73k_images.npy, which
    IS the 1,000-element id list; either can be used, but they are different
    shapes and must not be confused.
    """
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO_ID, "shared1000.npy", repo_type="dataset")
    a = np.load(p).ravel()
    if a.dtype == bool or (a.size == 73000 and set(np.unique(a).tolist()) <= {0, 1}):
        shared = set(np.flatnonzero(a).tolist())          # mask -> ids
    else:
        shared = set(a.astype(int).tolist())              # already ids
    if len(shared) != 1000:
        print(f"  !! shared1000 resolved to {len(shared)} images, expected 1000.")
    return np.array([int(i) in shared for i in nsd_ids])


# --------------------------------------------------------------------------- #
# Models + evaluation
# --------------------------------------------------------------------------- #
def train_eval(X, y, is_test, model="ridge", alpha=1000.0, labels=None,
               r2_weighting="uniform", test_ids=None, return_pred=False):
    # `labels` names the target columns (default: the 11 colors). Passing a
    # different list lets this decode any target, e.g. luminance bins.
    #
    # return_pred is ADDITIVE and changes nothing about the fit, the metrics or
    # the existing keys. When True the returned dict also carries "pred" and
    # "yte" -- the held-out predictions and targets, in test-row order -- so a
    # caller can score arbitrary SUBSETS of the held-out set (e.g. dark vs
    # bright images) without refitting and without duplicating the
    # standardisation and alpha-selection logic, which must stay identical to
    # the published runs.
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
        # within linear models, does a different regularizer give the same answer?
        # PERFORMANCE (does not change the objective or its optimum):
        #   n_jobs   the 3 CV folds run in parallel. The old comment here said
        #            n_jobs=1 avoided "per-core data copies", true when X was a
        #            whole ROI, but after voxel matching Xtr is k=397 columns,
        #            i.e. ~34 MB, so three workers cost ~100 MB. Freed 2-3x.
        #   selection="random"  coordinate descent visits coefficients in random
        #            order rather than cyclically. Neighbouring voxels are highly
        #            correlated and cyclic CD zig-zags on correlated designs;
        #            randomised order is the standard remedy. random_state fixes
        #            it, so runs stay reproducible.
        # Both are solver-level choices: same penalised least-squares problem,
        # same minimiser, only the route there differs. Verified against the
        # subjects computed under the previous settings (see PROVENANCE).
        import inspect
        from sklearn.linear_model import MultiTaskElasticNetCV
        kw = dict(l1_ratio=0.5, cv=3, max_iter=3000, n_jobs=3,
                  selection="random", random_state=0)
        # 'n_alphas' was renamed to 'alphas' (now accepts an int) in recent sklearn
        params = inspect.signature(MultiTaskElasticNetCV).parameters
        kw["n_alphas" if "n_alphas" in params else "alphas"] = 8
        kw = {k: v for k, v in kw.items() if k in params}   # tolerate older sklearn
        reg = MultiTaskElasticNetCV(**kw).fit(Xtr, ytr)
        pred = reg.predict(Xte)
        print(f"MultiTaskElasticNetCV selected alpha = {reg.alpha_:.5g}  "
              f"(path {reg.alphas_.min():.3g}..{reg.alphas_.max():.3g})")
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

    # Optional: the same R2 computed over IMAGES rather than trials.
    #
    # The held-out set contains several trials per image (2,371 trials over 930
    # images for participant 3), so the number of independent stimuli is smaller
    # than the number of rows R2 is computed over. Reviewer comment [19] asks
    # whether that matters. Passing `test_ids` adds an image-averaged figure
    # computed from the SAME predictions, so the only difference between the two
    # numbers is the aggregation and nothing else can drift between them.
    #
    # A target is a property of the image, so every repeat of an image carries an
    # identical target row. That is asserted rather than assumed, if it were
    # ever false, averaging the targets would be silently wrong.
    overall_r2_byimage = overall_r2_byimage_fixedw = None
    if test_ids is not None:
        tid = np.asarray(test_ids)
        if len(tid) != len(yte):
            raise ValueError(f"test_ids has {len(tid)} entries, "
                             f"the held-out set has {len(yte)}")
        uniq = np.unique(tid)
        yi = np.stack([yte[tid == g].mean(0) for g in uniq])
        pi = np.stack([pred[tid == g].mean(0) for g in uniq])
        spread = max(float(np.ptp(yte[tid == g], axis=0).max()) for g in uniq)
        if spread > 1e-9:
            raise ValueError(f"targets differ between repeats of the same image "
                             f"(max within-image range {spread:.2e}); averaging "
                             f"them would be wrong")
        overall_r2_byimage = float(r2_score(yi, pi, multioutput=mo))

        # And the same thing with the POOLING WEIGHTS HELD FIXED.
        #
        # "variance_weighted" derives its weights from whichever y_true it is
        # given: sklearn uses each column's total sum of squares. At trial level
        # those come from yte, where an image appears once per presentation; at
        # image level from yi, where it appears once. Repeat counts are NOT
        # uniform, participants 5 and 7 have exactly 3 trials per image, but 4
        # and 8 have 2,188 over 907 images and 6 has 2,371 over 930, so for
        # those participants the two figures differ in how the eleven colour
        # bins are weighted against each other, not only in aggregation.
        #
        # Two variables moving at once is what this comparison exists to avoid.
        # This third figure re-pools the image-level per-column R2 using the
        # TRIAL-level weights, so aggregation is the only thing that changes.
        # Where repeats are uniform it is identical to the line above by
        # construction, which is a free check that the weighting logic is right.
        w_trial = ((yte - yte.mean(0)) ** 2).sum(0)
        r2i_raw = r2_score(yi, pi, multioutput="raw_values")
        overall_r2_byimage_fixedw = float(
            (r2i_raw * w_trial).sum() / max(w_trial.sum(), 1e-12))

    print(f"\n=== decode ({model}, R2 pooling={r2_weighting}) ===")
    print(f"overall R^2 : {overall_r2:.3f}")
    print(f"dominant-bin top-1 acc : {top1:.3f} "
          f"(chance ~= {1/len(labels):.3f})")
    print("per-target R^2:")
    for name, val in zip(labels, r2):
        print(f"   {name:8s} {val:+.3f}")
    if overall_r2_byimage is not None:
        print(f"overall R^2 (averaged within image, n={len(uniq)} images "
              f"vs {len(yte)} trials) : {overall_r2_byimage:.3f}")
        print(f"overall R^2 (image-averaged, trial-level pooling weights) : "
              f"{overall_r2_byimage_fixedw:.3f}")
    out = {"overall_r2": overall_r2, "top1": top1,
           "per_color_r2": dict(zip(labels, r2.tolist())),
           "overall_r2_byimage": overall_r2_byimage,
           "overall_r2_byimage_fixedw": overall_r2_byimage_fixedw}
    if return_pred:
        out["pred"] = pred
        out["yte"] = yte
    return out


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
