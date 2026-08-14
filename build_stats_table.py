"""
build_stats_table.py
====================
Generate the full group-level statistics table (Supplement Table S3) directly
from the result summaries, so the reported values are computed rather than
transcribed.

Why this exists: the same p-value yields a different Benjamini-Hochberg q
depending on the family it is corrected within, so contrasts must be corrected
together, in one pass, exactly as the table presents them. Assembling the table
by pasting numbers from separate `brain2vision.stats` invocations silently mixes
FDR families and invites transcription slips.

Families are declared explicitly below and corrected separately, matching the
caption ("... FDR control (q) within each family").

Usage:
    python build_stats_table.py                 # the corrected (_v2) results
    python build_stats_table.py --suffix ""     # the pre-fix results
    python build_stats_table.py --json out.json # machine-readable, for the builder
"""

import argparse
import itertools
import json

import numpy as np

ROIS = ["early_v1v3", "v4_color", "concept"]
NICE = {"early_v1v3": "early", "v4_color": "V4", "concept": "higher"}


def paired(diff, n_boot=20000, seed=0):
    """Exact paired sign-flip permutation p + subject-level bootstrap 95% CI.

    Also returns how many participants show the group-mean direction. With n = 8
    the permutation p bottoms out at 2/2**8 = 0.0078, which 17 of the 21
    contrasts reach; the agreement count is what distinguishes them, so it is
    reported alongside p rather than left implicit.
    """
    diff = np.asarray(diff, float)
    n = len(diff)
    obs = diff.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    perm = (signs * diff).mean(1)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    n_agree = int(np.sum(np.sign(diff) == np.sign(obs)))
    return obs, p, float(lo), float(hi), n_agree


def bh(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ps[order[i]] * m / (i + 1))
        q[order[i]] = prev
    return q


def load(stem, suffix):
    f = f"roi_{stem}{suffix}_8subj_summary.npy"
    d = np.load(f, allow_pickle=True).item()
    return {r: np.asarray(d["agg"][r]["ov"], float) for r in ROIS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2",
                    help="'_v2' for the corrected runs, '' for the pre-fix ones")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    s = args.suffix

    col, lum = load("color_vw", s), load("luminance_vw", s)
    res, pcol = load("colorresid_vw", s), load("pcolor_vw", s)
    pres = load("pcolorresid_vw", s)
    fg, bg = load("fgcolor_cleanvw", s), load("bgcolor_cleanvw", s)

    # ---- families, exactly as the caption describes them --------------------
    families = {
        "Dissociation": [
            ("Dissociation: higher − early (colour)", col["concept"] - col["early_v1v3"]),
            ("Dissociation: higher − V4 (colour)", col["concept"] - col["v4_color"]),
            ("Luminance: early − higher", lum["early_v1v3"] - lum["concept"]),
            ("Luminance: early − V4", lum["early_v1v3"] - lum["v4_color"]),
            ("Interaction (colour−luminance): higher − early",
             (col["concept"] - col["early_v1v3"]) - (lum["concept"] - lum["early_v1v3"])),
            ("Interaction (colour−luminance): V4 − early",
             (col["v4_color"] - col["early_v1v3"]) - (lum["v4_color"] - lum["early_v1v3"])),
        ],
        "Residualisation": [
            ("Collapse (physical): early, raw − residual", col["early_v1v3"] - res["early_v1v3"]),
            ("Collapse (physical): V4, raw − residual", col["v4_color"] - res["v4_color"]),
            ("Collapse (physical): higher, raw − residual", col["concept"] - res["concept"]),
            ("Collapse (perceptual): early, raw − residual", pcol["early_v1v3"] - pres["early_v1v3"]),
            ("Collapse (perceptual): V4, raw − residual", pcol["v4_color"] - pres["v4_color"]),
            ("Collapse (perceptual): higher, raw − residual", pcol["concept"] - pres["concept"]),
            ("Interaction (physical): collapse, higher − early",
             (col["concept"] - res["concept"]) - (col["early_v1v3"] - res["early_v1v3"])),
            ("Interaction (perceptual): collapse, higher − early",
             (pcol["concept"] - pres["concept"]) - (pcol["early_v1v3"] - pres["early_v1v3"])),
            ("Residual target (physical): early − higher", res["early_v1v3"] - res["concept"]),
            ("Residual target (perceptual): early − higher", pres["early_v1v3"] - pres["concept"]),
        ],
        "Attention / retinotopy": [
            ("Attention: FG − BG, early", fg["early_v1v3"] - bg["early_v1v3"]),
            ("Attention: FG − BG, V4", fg["v4_color"] - bg["v4_color"]),
            ("Attention: FG − BG, higher visual cortex", fg["concept"] - bg["concept"]),
            ("Retinotopy: (BG−FG) early − higher",
             (bg["early_v1v3"] - fg["early_v1v3"]) - (bg["concept"] - fg["concept"])),
            ("Retinotopy: (BG−FG) early − V4",
             (bg["early_v1v3"] - fg["early_v1v3"]) - (bg["v4_color"] - fg["v4_color"])),
        ],
    }

    n = len(col["concept"])
    print("=" * 100)
    print(f"GROUP-LEVEL STATISTICS  (n = {n}; suffix '{s or 'pre-fix'}')")
    print(f"exact sign-flip floor = 2/2^{n} = {2/2**n:.4f}   FDR applied within each family")
    print("=" * 100)
    print(f"{'Contrast':50s} {'Mean':>8s} {'p':>7s} {'q':>7s} {'n/8':>5s} {'Sig':>5s}  95% CI")

    rows = []
    for fam, items in families.items():
        stats = [paired(v) for _, v in items]
        qs = bh([st[1] for st in stats])
        print(f"\n-- {fam} ({len(items)} tests) " + "-" * (72 - len(fam)))
        for (name, _), (obs, p, lo, hi, n_agree), q in zip(items, stats, qs):
            # Derive the stars from the DISPLAYED (rounded) q, so a reader never
            # sees "q = 0.010" tagged ** against a caption that says ** is q<0.01.
            q_shown = round(q, 3)
            sig = "**" if q_shown < 0.01 else ("*" if q_shown < 0.05 else "n.s.")
            print(f"{name:50s} {obs:+8.3f} {p:7.3f} {q:7.3f} {n_agree:4d}/{n} {sig:>5s}  "
                  f"[{lo:+.3f}, {hi:+.3f}]")
            rows.append({"family": fam, "contrast": name,
                         "mean": round(obs, 4), "p": round(p, 4), "q": round(q, 4),
                         "sig": sig, "ci": [round(lo, 4), round(hi, 4)],
                         "n_agree": n_agree, "n": int(n),
                         # pre-rounded strings, ready to paste into the builder
                         "cells": [name, f"{obs:+.3f}", f"{p:.3f}", f"{q:.3f}",
                                   f"{n_agree}/{n}", sig, f"[{lo:+.3f}, {hi:+.3f}]"]})

    print("\nGroup-mean R² per region (early / V4 / higher):")
    for tag, dd in (("colour", col), ("luminance", lum), ("colour residual", res),
                    ("perceptual", pcol), ("perceptual residual", pres),
                    ("foreground", fg), ("background", bg)):
        print(f"  {tag:22s} " + " / ".join(f"{dd[r].mean():.3f}" for r in ROIS))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
