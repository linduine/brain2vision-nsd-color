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


def run_fgbg(fg_npy, bg_npy):
    """
    Stats for the foreground-vs-background colour test (the attention control).
    Reports, per ROI:
      * FG-BG (foreground bias): >0 would mean colour decoding favours the
        attended foreground -- the prediction of the attention account;
      * and the retinotopy INTERACTION: whether early visual gains more from the
        (peripheral) background than the foveally-biased concept / V4 do.
    """
    F = np.load(fg_npy, allow_pickle=True).item()
    B = np.load(bg_npy, allow_pickle=True).item()
    fg = {r: np.array(F["agg"][r]["ov"]) for r in ROIS}
    bg = {r: np.array(B["agg"][r]["ov"]) for r in ROIS}
    n = len(fg[ROIS[0]])
    sem = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    print(f"n subjects = {n}   (exact sign-flip p floor = {2/2**n:.4f})\n")
    print(f"{'ROI':<12}{'foreground R2':>16}{'background R2':>16}")
    for r in ROIS:
        print(f"{r:<12}{fg[r].mean():+9.3f}+/-{sem(fg[r]):.3f}"
              f"{bg[r].mean():+9.3f}+/-{sem(bg[r]):.3f}")
    print()

    dbf = {r: bg[r] - fg[r] for r in ROIS}                 # background advantage
    tests = []
    for r in ROIS:                                          # attention: fg>bg?
        tests.append((f"FG-BG: {r} (foreground bias)", fg[r] - bg[r]))
    tests.append(("INTERACTION: (bg-fg) early - concept", dbf["early_v1v3"] - dbf["concept"]))
    tests.append(("INTERACTION: (bg-fg) early - v4",      dbf["early_v1v3"] - dbf["v4_color"]))

    stats = [_paired(d) for _, d in tests]
    qs = _bh([s[1] for s in stats])
    print(f"{'contrast':<40}{'mean d':>9}{'p':>8}{'q(FDR)':>8}   95% CI")
    for (name, _), (obs, p, lo, hi), q in zip(tests, stats, qs):
        print(f"{name:<40}{obs:+9.4f}{p:8.3f}{q:8.3f}   [{lo:+.4f}, {hi:+.4f}]")
    print("\npaired sign-flip permutation p, 20k-bootstrap 95% CI, BH-FDR q "
          f"across {len(tests)} tests.")
    print("Reading: FG-BG ~ 0 (or negative) => colour decoding is NOT foreground-"
          "specific (attention account not supported). Positive early-vs-concept "
          "interaction => early visual gains from the peripheral background "
          "(retinotopy).")
    return tests, stats, qs


def run_residual(raw_npy, residual_npy):
    """
    Test the semantic collapse: is raw-colour decoding significantly higher than
    residual-colour decoding, per ROI (paired across subjects)? Both summaries
    must be the same colour target decoded raw vs residualized-against-semantics.
    """
    R = np.load(raw_npy, allow_pickle=True).item()
    S = np.load(residual_npy, allow_pickle=True).item()
    raw = {r: np.array(R["agg"][r]["ov"]) for r in ROIS}
    res = {r: np.array(S["agg"][r]["ov"]) for r in ROIS}
    n = len(raw[ROIS[0]])
    sem = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    print(f"n subjects = {n}   (exact sign-flip p floor = {2/2**n:.4f})\n")
    print(f"{'ROI':<12}{'raw R2':>14}{'residual R2':>14}")
    for r in ROIS:
        print(f"{r:<12}{raw[r].mean():+9.3f}+/-{sem(raw[r]):.3f}"
              f"{res[r].mean():+9.3f}+/-{sem(res[r]):.3f}")
    print()
    tests = [(f"COLLAPSE: {r} (raw - residual)", raw[r] - res[r]) for r in ROIS]
    stats = [_paired(d) for _, d in tests]
    qs = _bh([s[1] for s in stats])
    print(f"{'contrast':<38}{'mean d':>9}{'p':>8}{'q(FDR)':>8}   95% CI")
    for (name, _), (obs, p, lo, hi), q in zip(tests, stats, qs):
        print(f"{name:<38}{obs:+9.4f}{p:8.3f}{q:8.3f}   [{lo:+.4f}, {hi:+.4f}]")
    print("\nA significant positive 'raw - residual' = the colour advantage drops "
          "significantly when object identity is removed.")
    return tests, stats, qs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--color", help="roi_color_*_summary.npy")
    p.add_argument("--luminance", help="roi_luminance_*_summary.npy")
    p.add_argument("--fg", help="roi_fgcolor_*_summary.npy")
    p.add_argument("--bg", help="roi_bgcolor_*_summary.npy")
    p.add_argument("--raw", help="raw-colour roi_*_summary.npy (with --residual)")
    p.add_argument("--residual", help="residual-colour roi_*_summary.npy (with --raw)")
    args = p.parse_args()
    if args.fg and args.bg:
        run_fgbg(args.fg, args.bg)
    elif args.raw and args.residual:
        run_residual(args.raw, args.residual)
    elif args.color and args.luminance:
        run(args.color, args.luminance)
    else:
        p.error("give --color+--luminance, --fg+--bg, or --raw+--residual")


if __name__ == "__main__":
    main()
