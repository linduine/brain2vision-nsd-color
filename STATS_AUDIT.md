# Statistical audit — 13 Aug 2026

Independent re-derivation of every group-level statistic from the per-subject
`.npy` summaries, checked against what the manuscript says.

**Verdict: the statistical machinery is correct. Six reported numbers were
wrong.** All six were transcription errors in the text, not computation errors —
`build_stats_table.py` had them right all along.

---

## Part 1 — Methods verified correct

| Check | Method | Result |
|---|---|---|
| Sign-flip permutation | Re-derived all 2⁸ = 256 sign assignments independently | ✅ matches |
| Two-sidedness | `\|perm\| ≥ \|obs\| − 1e-12` counts ties, includes the identity vector | ✅ correct, and conservative |
| p floor | 2/256 = 0.0078125 — both the all-`+1` and all-`−1` vectors always qualify | ✅ as documented |
| Benjamini–Hochberg | Compared against an independent step-up implementation over **2000 random p-vectors** plus edge cases (all-equal, all-ones, zeros) | ✅ identical every time |
| Bootstrap / permutation agreement | Checked all 21 contrasts | ✅ **no disagreements** |
| t-interval / permutation agreement | Checked all 21 contrasts | ✅ **no disagreements** |
| FDR family structure | 6 + 10 + 5 = 21, matching the caption | ✅ |
| Global vs within-family FDR | Recomputed BH across all 21 together | ✅ **zero verdicts change** |

That last row is worth keeping: if a reviewer objects to correcting within
families rather than across all contrasts, the answer is that it makes no
difference. Now stated in Methods.

---

## Part 2 — Six errors found and corrected

| # | Location | Said | Should be |
|---|---|---|---|
| 1 | Results, double dissociation | interaction V4 − early, `q = 0.010` | **`q = 0.012`** |
| 2 | Results, residualisation (both targets) | `all q = 0.008` | **`all q = 0.009`** |
| 3 | Results, perceptual collapse | `all q = 0.008` | **`all q = 0.009`** |
| 4 | Figure 2 caption | `all q = 0.008` | **`all q = 0.009`** |
| 5 | Results, physical collapse in higher visual cortex | CI `[+0.052, +0.063]` | **`[+0.051, +0.063]`** |
| 6 | Results, attention control | V4 foreground−background `−0.000` | **`+0.001`** (wrong sign) |

Errors 2–4 share a cause: **`q = 0.008` never occurs anywhere in the table.**
0.008 is the *p* floor; the smallest *q* in the Residualisation family is 0.009.
The p-value appears to have been copied into the q slot three times. Error 6
flipped the sign of a near-zero value.

None changes a conclusion — every affected contrast keeps its significance class
— but a reviewer who checks the table against the text would find all six.

**Guard added:** the text now quotes only q-values that exist in the computed
table, and this is checkable automatically (Part 5).

---

## Part 3 — The finding that most affects how you report

**Seventeen of the twenty-one contrasts sit exactly at the exact-test floor of
p = 0.0078.**

This is not a coincidence and not a bug. With n = 8 the sign-flip test cannot
return anything smaller, and it returns the floor whenever all eight
participants agree in sign and the effect is large relative to the spread.

The risk is presentational. "p = 0.008" reads like a precisely estimated
p-value, and any reviewer who recognises 2/2⁸ will wonder whether the authors
know what they are reporting. **The honest framing is that p = 0.008 means "as
significant as eight participants permit."**

Sign agreement is the more informative statistic, and it is strong:

| Contrast | Mean ΔR² | p | Participants agreeing |
|---|---|---|---|
| **Interaction (colour−luminance), higher − early** | +0.023 | 0.008 | **8/8** |
| Collapse (physical), higher | +0.057 | 0.008 | **8/8** |
| Interaction (physical): collapse, higher − early | +0.021 | 0.008 | **8/8** |
| Reversal (physical): residual early − higher | +0.009 | 0.008 | **8/8** |
| Retinotopy: (BG−FG) early − higher | +0.009 | 0.008 | **8/8** |
| Luminance: early − higher | +0.011 | 0.016 | 7/8 |
| **Dissociation: higher − early (colour)** | +0.012 | 0.031 | **6/8** |
| Reversal (perceptual): residual early − higher | +0.006 | 0.031 | 6/8 |
| Attention: FG − BG, V4 | +0.001 | 0.812 | 5/8 (null, as intended) |

**Note the pattern.** Your *headline raw contrast* — higher visual cortex decodes
colour better than early — is the **weakest** result in the paper: 6/8
participants, p = 0.031, q = 0.031. Your *interaction* contrasts are all 8/8 at
the floor.

That is an argument for what the manuscript already does: leading on the
dissociation and the residualisation interaction rather than on the raw colour
ordering. It also matches the Limitations note that a minority of participants
decode colour as well or better from early cortex. Worth being deliberate about
in the response to reviewers — the 6/8 contrast is where a sceptic will push.

---

## Part 4 — Bootstrap intervals are narrow (but nothing changes)

The percentile bootstrap CIs are consistently **0.76–0.77× the width of the
corresponding t interval** (df = 7), across all 21 contrasts.

This is exactly what theory predicts, not a bug:

```
sqrt(7/8) × (1.96 / 2.365) = 0.935 × 0.829 = 0.775
```

— the bootstrap uses the biased (1/n) variance and normal-ish tails, while the t
interval uses the unbiased variance and t₇ tails. The agreement with 0.76–0.77
observed is itself evidence the implementation is right.

**Consequence:** at n = 8 these nominal 95% intervals under-cover somewhat.
**Consequence for your conclusions: none.** Every contrast significant by the
permutation test also has a t interval excluding zero, and vice versa. Both are
now stated in Methods, so a reviewer raising it finds the answer already there.

---

## Part 5 — Re-running these checks

```bash
python build_stats_table.py                  # the canonical table
python build_stats_table.py --json out.json  # machine-readable
python count_words.py <build_manuscript.py>  # section word counts
```

The text-vs-table check that caught errors 1–6: extract every `q = 0.xxx` from
the manuscript source and confirm each appears among the computed q-values.
`q = 0.008` should never appear — that value is a p, not a q.

---

## Part 6 — Not checked, and worth knowing

- **Within-subject variance is ignored.** Participants are the unit of analysis
  and each contributes one number per contrast. Standard for group fMRI, but it
  discards per-trial uncertainty.
- **The 11-colour family** (Figure 1 stars, BH across eleven colours) was not
  re-derived here; only the 21 group contrasts were.
- **Reliability (split-half r ≈ 0.9)** and **ncsnr** were not re-derived.
- **Whether the R² metric itself is correctly computed** is a question about
  `color_decode.py`, not about the group statistics, and is covered by the
  planted-ground-truth suite rather than by this audit.
