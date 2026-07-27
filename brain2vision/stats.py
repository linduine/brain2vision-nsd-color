"""
stats.py
========
Formal significance tests for the ROI comparison, from the per-subject summaries
that `replicate_subjects` saves. Replaces the informal "difference ~ 2-3x SEM"
reading with proper inference:

  * paired sign-flip permutation test across subjects (exact: 2^n sign vectors),
  * 95% bootstrap confidence interval on each ROI difference,
  * Benjamini-Hochberg FDR correction across the whole family of tests.

It reports the main-effect contrasts (which ROI decodes color / luminance best)
and — the key one — the *dissociation* (interaction): whether a region is more
color-biased (colorR2 - luminanceR2) than another. That interaction is what makes
the color/luminance crossover a real double dissociation rather than two
coincidental main effects.

Note on n=8: the exact sign-flip test has a two-sided p floor of 2/2^8 = 0.0078,
reached when the observed difference is more extreme than all other sign
combinations — i.e. "as significant as 8 subjects allow", not a marginal value.

Usage
-----
    python -m brain2vision.stats \
        --color roi_color_8subj_summary.npy \
        --luminance roi_luminance_8subj_summary.npy
"""

import argparse
import itertools
import numpy as np

ROIS = ["early_v1v3", "v4_color", "concept"]


def _paired(diff, n_boot=20000, seed=0):
    """Paired sign-flip permutation (exact) + bootstrap CI for a per-subject
    difference vector."""
    diff = np.asarray(diff, float)
    n = len(diff)
    obs = diff.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=n)))  # 2^n x n
    perm = (signs * diff).mean(1)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())          # two-sided
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, p, float(lo), float(hi)


def _bh(ps):
    """Benjamini-Hochberg FDR-adjusted q-values."""
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps)
    q = np.empty(m); prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ps[order[i]] * m / (i + 1))
        q[order[i]] = prev
    return q


def run(color_npy, luminance_npy):
    C = np.load(color_npy, allow_pickle=True).item()
    L = np.load(luminance_npy, allow_pickle=True).item()
    cov = {r: np.array(C["agg"][r]["ov"]) for r in ROIS}   # per-subject overall R2
    lov = {r: np.array(L["agg"][r]["ov"]) for r in ROIS}
    n = len(cov[ROIS[0]])
    print(f"n subjects = {n}   (exact sign-flip p floor = {2/2**n:.4f})\n")

    bias = {r: cov[r] - lov[r] for r in ROIS}              # color-minus-luminance
    tests = []
    for a, b in [("concept", "early_v1v3"), ("concept", "v4_color"),
                 ("early_v1v3", "v4_color")]:
        tests.append((f"COLOR: {a} - {b}", cov[a] - cov[b]))
    for a, b in [("early_v1v3", "concept"), ("early_v1v3", "v4_color"),
                 ("concept", "v4_color")]:
        tests.append((f"LUM:   {a} - {b}", lov[a] - lov[b]))
    for a, b in [("concept", "early_v1v3"), ("v4_color", "early_v1v3")]:
        tests.append((f"BIAS:  {a} - {b} (color-lum)", bias[a] - bias[b]))

    stats = [_paired(d) for _, d in tests]
    qs = _bh([s[1] for s in stats])

    print(f"{'contrast':<38}{'mean d':>9}{'p':>8}{'q(FDR)':>8}   95% CI")
    for (name, _), (obs, p, lo, hi), q in zip(tests, stats, qs):
        print(f"{name:<38}{obs:+9.4f}{p:8.3f}{q:8.3f}   [{lo:+.4f}, {hi:+.4f}]")
    print("\npaired sign-flip permutation p, 20k-bootstrap 95% CI, BH-FDR q "
          f"across {len(tests)} tests.")
    return tests, stats, qs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--color", required=True, help="roi_color_*_summary.npy")
    p.add_argument("--luminance", required=True, help="roi_luminance_*_summary.npy")
    args = p.parse_args()
    run(args.color, args.luminance)


if __name__ == "__main__":
    main()
