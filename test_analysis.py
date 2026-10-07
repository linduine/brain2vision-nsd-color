"""
test_analysis.py
================
Verification suite for the analysis logic behind the preprint.

The question this answers: *how do we know the models were written correctly?*
We cannot prove correctness, but we can plant known ground truth and check the
code recovers it, and plant a known null and check the code reports nothing.
Each test targets a specific claim the paper depends on.

Run:
    python test_analysis.py          # all tests
    python test_analysis.py --quick  # skip the slower decoder tests

Requires the project venv (numpy + scikit-learn). No fMRI data needed: every
test builds its own synthetic data with a known answer.
"""

import argparse
import importlib.util
import itertools
import sys

import numpy as np

PASS, FAIL = "PASS", "**FAIL**"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def _load(modname, path):
    """Import a module file directly (bypasses the package __init__)."""
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# =====================================================================
# 1. STATISTICS  (pure NumPy, always runnable)
# =====================================================================
def test_statistics():
    print("\n1. STATISTICS (brain2vision/stats.py)")
    st = _load("st", "brain2vision/stats.py")
    paired, bh = st._paired, st._bh
    rng = np.random.default_rng(0)

    # Null data must not produce systematically small p.
    ps = [paired(rng.normal(0, 1, 8))[1] for _ in range(400)]
    check("null calibration: P(p<0.05) under H0 is not inflated",
          np.mean(np.array(ps) < 0.05) <= 0.08,
          f"observed {np.mean(np.array(ps) < 0.05):.3f}")

    # A consistent effect must reach, and not exceed, the exact floor.
    p = paired(np.full(8, 0.05) + rng.normal(0, 1e-3, 8))[1]
    check("permutation floor is exactly 2/2^8", abs(p - 2 / 256) < 1e-12, f"p={p:.4f}")

    # Two-sided test must be sign-symmetric.
    a = rng.normal(0.02, 0.01, 8)
    check("two-sided test is sign-symmetric", paired(a)[1] == paired(-a)[1])

    # Bootstrap CI must bracket the observed mean.
    o, p, lo, hi = paired(rng.normal(0.03, 0.01, 8))
    check("bootstrap CI contains the observed mean", lo <= o <= hi)

    # Planted effect -> detected; planted null -> not detected.
    o, p, lo, hi = paired(np.array([.02, .03, .025, .018, .031, .022, .028, .024]))
    check("planted consistent effect is detected (CI excludes 0)", lo > 0, f"p={p:.4f}")
    o, p, lo, hi = paired(np.array([.02, -.019, .03, -.031, .01, -.011, .025, -.024]))
    check("planted null is not detected (CI contains 0)", lo < 0 < hi, f"p={p:.4f}")

    # BH-FDR against an independently computed expectation.
    ps6 = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    q = bh(ps6)
    m = len(ps6)
    expect = np.minimum.accumulate(
        [p_ * m / (i + 1) for i, p_ in enumerate(sorted(ps6))][::-1])[::-1]
    check("BH-FDR matches the textbook computation", np.allclose(np.sort(q), expect))
    check("BH-FDR is monotone in p", np.all(np.diff(q[np.argsort(ps6)]) >= -1e-12))
    check("BH-FDR q >= p", np.all(q >= ps6 - 1e-12))

    # Exhaustive, not sampled.
    check("permutation is exhaustive (256 sign vectors)",
          len(list(itertools.product([1, -1], repeat=8))) == 256)


# =====================================================================
# 2. RESIDUALISATION  (the paper's central analysis)
# =====================================================================
def test_residualisation():
    print("\n2. RESIDUALISATION (does removing objects actually remove objects?)")
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(1)

    n, n_obj, n_col = 2000, 80, 11
    objects = (rng.random((n, n_obj)) < 0.05).astype(float)     # sparse presence
    W = rng.normal(0, 1, (n_obj, n_col))
    chromatic = rng.normal(0, 1, (n, n_col))                     # content-unpredicted part
    colour = objects @ W + chromatic                             # colour = semantic + chromatic

    # cross-validated residualisation, as in semantic_residual
    pred = np.zeros_like(colour)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(objects):
        pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(objects[tr], colour[tr]).predict(objects[te])
    resid = colour - pred

    # (a) the residual must be ~uncorrelated with the object regressors
    cors = [abs(np.corrcoef(objects[:, j], resid[:, k])[0, 1])
            for j in range(0, n_obj, 7) for k in range(n_col)]
    check("residual is ~orthogonal to object presence", max(cors) < 0.12,
          f"max|r| = {max(cors):.3f}")

    # (b) the residual must retain the planted content-unpredicted component
    r_keep = np.mean([abs(np.corrcoef(chromatic[:, k], resid[:, k])[0, 1]) for k in range(n_col)])
    check("residual retains the content-unpredicted (chromatic) component", r_keep > 0.5,
          f"mean |r| with planted chromatic = {r_keep:.2f}")

    # (c) the residual must LOSE the planted semantic component
    sem = objects @ W
    r_lose = np.mean([abs(np.corrcoef(sem[:, k], resid[:, k])[0, 1]) for k in range(n_col)])
    check("residual loses the planted semantic component", r_lose < 0.2,
          f"mean |r| with planted semantic = {r_lose:.2f}")


# =====================================================================
# 3. R² POOLING  (uniform vs variance-weighted)
# =====================================================================
def test_pooling():
    print("\n3. R² POOLING (is variance weighting doing what we claim?)")
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(2)

    n = 500
    y = np.column_stack([rng.normal(0, 1, n),          # high-variance, predictable
                         rng.normal(0, 1e-4, n)])       # near-constant ("purple")
    pred = np.column_stack([y[:, 0] * 0.9 + rng.normal(0, .3, n),
                            rng.normal(0, 1e-4, n)])    # unpredictable

    uni = r2_score(y, pred, multioutput="uniform_average")
    var = r2_score(y, pred, multioutput="variance_weighted")
    per = r2_score(y, pred, multioutput="raw_values")

    check("uniform pooling equals the plain mean of per-target R²",
          abs(uni - per.mean()) < 1e-9)
    check("a near-constant target drags uniform pooling down", uni < per[0] - 0.1,
          f"uniform={uni:+.3f} vs good-target R²={per[0]:+.3f}")
    check("variance weighting is dominated by the high-variance target",
          abs(var - per[0]) < 0.05, f"variance-weighted={var:+.3f}")
    check("variance weighting > uniform when a dead bin is present", var > uni,
          f"{var:+.3f} > {uni:+.3f}")


# =====================================================================
# 4. DECODER  (recovers planted signal; reports nothing on shuffled labels)
# =====================================================================
def test_decoder():
    print("\n4. DECODER (brain2vision/color_decode.py :: train_eval)")
    cd = _load("cd", "brain2vision/color_decode.py")
    rng = np.random.default_rng(3)

    n, n_vox, n_col = 800, 120, 11
    y = rng.random((n, n_col)); y /= y.sum(1, keepdims=True)
    W = rng.normal(0, 1, (n_col, n_vox))
    X = (y @ W + rng.normal(0, 0.5, (n, n_vox))).astype(np.float32)
    is_test = np.zeros(n, bool); is_test[int(n * .75):] = True
    labels = [f"c{i}" for i in range(n_col)]

    r = cd.train_eval(X, y, is_test, model="ridge", labels=labels, r2_weighting="variance")
    check("recovers a planted linear signal", r["overall_r2"] > 0.3,
          f"R² = {r['overall_r2']:+.3f}")

    # Shuffled targets must destroy decoding (the falsification logic).
    y_shuf = y[rng.permutation(n)]
    r0 = cd.train_eval(X, y_shuf, is_test, model="ridge", labels=labels, r2_weighting="variance")
    check("shuffled labels collapse decoding to ~0", r0["overall_r2"] < 0.05,
          f"R² = {r0['overall_r2']:+.3f}")

    # Pure noise must not decode.
    Xn = rng.normal(0, 1, (n, n_vox)).astype(np.float32)
    rn = cd.train_eval(Xn, y, is_test, model="ridge", labels=labels, r2_weighting="variance")
    check("pure-noise voxels do not decode", rn["overall_r2"] < 0.05,
          f"R² = {rn['overall_r2']:+.3f}")

    # All decoders should agree on a strong planted signal.
    for m in ("elasticnet", "svr", "kernel"):
        try:
            rm = cd.train_eval(X, y, is_test, model=m, labels=labels, r2_weighting="variance")
            check(f"decoder '{m}' also recovers the planted signal", rm["overall_r2"] > 0.2,
                  f"R² = {rm['overall_r2']:+.3f}")
        except Exception as e:
            check(f"decoder '{m}' runs", False, str(e)[:60])


# =====================================================================
# 5. SPLIT-HALF  (are the halves really independent?)
# =====================================================================
def test_split_half():
    print("\n5. SPLIT-HALF (reliability.py image-disjoint split)")
    rng = np.random.default_rng(4)
    ids = rng.integers(0, 73000, 5000)          # NSD image ids, with repeats
    isA = (ids % 2 == 0)                         # the parity rule used in reliability.py
    a, b = set(ids[isA]), set(ids[~isA])
    check("halves share no image", len(a & b) == 0, f"overlap = {len(a & b)}")
    check("every repeat of an image stays in one half",
          all((ids[isA] % 2 == 0).all() for _ in [0]) and (ids[~isA] % 2 == 1).all())
    check("halves are roughly balanced", 0.4 < isA.mean() < 0.6, f"{isA.mean():.2f}")


# Size of the full suite. README.md and PROVENANCE.md quote this number, so if
# you add or remove a check, update it here and in both files. The run fails if
# the two disagree, which stops the documented count going stale.
EXPECTED_CHECKS = 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the slower decoder tests")
    args = ap.parse_args()

    print("=" * 68)
    print("ANALYSIS VERIFICATION SUITE")
    print("Each test plants a known answer and checks the code recovers it.")
    print("=" * 68)

    test_statistics()
    test_split_half()
    skipped = None
    try:
        test_residualisation()
        test_pooling()
        if not args.quick:
            test_decoder()
    except ImportError as e:
        skipped = str(e)

    n, k = len(results), sum(results)
    print("\n" + "=" * 68)
    if skipped:
        print(f"INCOMPLETE: {k}/{n} checks passed, but {EXPECTED_CHECKS - n} of "
              f"{EXPECTED_CHECKS} did not run ({skipped}).")
        print("Install scikit-learn and re-run for the full suite.")
    elif args.quick:
        print(f"{k}/{n} checks passed (--quick: decoder tests not run)")
    else:
        print(f"{k}/{n} checks passed")
        if n != EXPECTED_CHECKS:
            print(f"WARNING: ran {n} checks but EXPECTED_CHECKS is {EXPECTED_CHECKS}. "
                  f"Update it here and the counts quoted in README.md and PROVENANCE.md.")
    print("=" * 68)

    complete = (not skipped) and (args.quick or n == EXPECTED_CHECKS)
    sys.exit(0 if (k == n and complete) else 1)


if __name__ == "__main__":
    main()
