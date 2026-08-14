# Provenance: which file backs which result

Every number in the preprint (*"Is colour in human visual cortex chromatic or
semantic?"*) traces to a `*_summary.npy` file in the repository root. This
document maps result → file → command, so the analysis can be re-run or audited
without guesswork.

**Last verified:** all values below were recomputed from the `.npy` files and
matched the manuscript.

## What is and is not in this repository

| | in the repo | why |
|---|---|---|
| `roi_*_summary.npy`, `reliability_*`, `ncsnr_*` | **yes** (~220 KB) | derived per-participant R² only; every number in the paper traces to these |
| `stats_table_v2.json`, figures, all analysis and verification code | **yes** | the reproducibility record |
| `data/*.npy` — per-image colour, luminance, semantic and stuff targets | **no** (~136 MB) | derived from the NSD/COCO stimuli, large, and regenerable by the commands in §2 and §7 |
| Raw NSD betas, region masks, COCO images | **no** | the NSD Terms of Use forbid redistribution |

Filenames under `data/` appear throughout this document as *the outputs of the
commands shown*, not as files you will find here. Run the commands and they
appear. Nothing in the analysis requires a file that cannot be regenerated from
the public NSD, COCO and MindEye2 releases.

---

## 1. Naming convention

Result files are named `roi_<target><variant>_<n>subj_summary.npy`:

| Element | Meaning |
|---|---|
| `color` | physical colour target (HSV rules → 11 basic colour terms) |
| `pcolor` | **perceptual** colour target (van de Weijer colour-naming model) |
| `colorresid` / `pcolorresid` | same target, **residualised** against 80-way COCO object presence |
| `luminance` | 11-bin brightness target |
| `fgcolor` / `bgcolor` | colour of the **foreground** (inside object box) / **background** |
| `synth_*` | NSD-synthetic (isoluminant) analyses |
| *(no suffix)* | **uniform** pooling of per-colour R² — superseded |
| `_vw` | **variance-weighted** pooling — *this is what the paper reports* |
| `_clean` / `_cleanvw` | restricted to the clean fg/bg image subset (see §4) |
| `_elasticnet`, `_kernel`, `_svr`, `_mlp`, `_mlp2` | alternative decoders |

> **Why `_vw` matters.** Overall R² averages across the 11 target bins, and the
> colour bins are wildly unequal: purple carries **0.2%** of the total target
> variance and gray/blue about **19%** each — a 115:1 ratio between the most and
> least variable bin. A near-constant bin cannot be predicted, so its held-out R²
> is ~0 or negative; under a uniform mean it nonetheless counts as much as blue.
> Variance weighting lets each colour contribute in proportion to how much it
> actually varies. Measured on the `_v2` runs, uniform pooling understates colour
> decoding by ~0.021 in every region (early 0.032 vs 0.054; V4 0.029 vs 0.049;
> higher 0.044 vs 0.066). **All headline numbers use `_vw`.**

> **Why luminance is also `_vw`.** The double-dissociation interaction subtracts a
> luminance R² from a colour R², so the two targets **must be pooled identically**
> — otherwise the subtraction mixes two scales that differ by ~0.021 for purely
> bookkeeping reasons. Matching them costs almost nothing on the luminance side:
> the 11 brightness bins are evenly populated (variance ratio 2.7:1, versus 115:1
> for colour), so uniform and variance pooling agree to ~0.0017 (early 0.0285 vs
> 0.0263; higher 0.0171 vs 0.0156). The interaction is +0.0236 under uniform
> pooling and **+0.0231** under variance weighting — the reported value.
>
> Note also that the pooling shift is near-constant across regions (+0.0218 /
> +0.0206 / +0.0220 for colour), so it largely cancels in any region *difference*.
> Pooling affects absolute values, not the contrasts the paper reports.

---

## 2. Files that back the paper (the canonical set)

| Paper element | File | Values (early / V4 / higher) |
|---|---|---|
| Fig 1, Table 1 "Ridge" | `roi_color_vw_v2_8subj_summary.npy` | 0.054 / 0.049 / 0.066 |
| Table 1 "Elastic-net" | `roi_color_elasticnet_v2_8subj_summary.npy` | 0.053 / 0.047 / 0.065 |
| Table 1 "Linear SVR" | `roi_color_svr_v2_8subj_summary.npy` | 0.049 / 0.045 / 0.061 |
| Table 1 "RBF kernel" | `roi_color_kernel_v2_8subj_summary.npy` | 0.045 / 0.038 / 0.052 |
| Table 1 "MLP" | `roi_color_mlp2_v2_8subj_summary.npy` | 0.051 / 0.046 / 0.065 |
| Luminance / dissociation | `roi_luminance_vw_v2_8subj_summary.npy` | 0.026 / 0.010 / 0.016 |
| Fig 2, collapse (physical) | `roi_colorresid_vw_v2_8subj_summary.npy` | 0.018 / 0.012 / 0.009 |
| Fig 2, raw perceptual | `roi_pcolor_vw_v2_8subj_summary.npy` | 0.066 / 0.068 / 0.090 |
| Fig 2, collapse (perceptual) | `roi_pcolorresid_vw_v2_8subj_summary.npy` | 0.018 / 0.016 / 0.012 |
| Fig 3, attention (foreground) | `roi_fgcolor_cleanvw_v2_8subj_summary.npy` | 0.018 / 0.020 / 0.022 |
| Fig 3, attention (background) | `roi_bgcolor_cleanvw_v2_8subj_summary.npy` | 0.033 / 0.019 / 0.028 |
| S7, synthetic hue null | `roi_synth_hue_8subj_summary.npy` | −0.117 / −0.073 / −0.066 |
| S7, positive control (identity) | `roi_synth_identity_8subj_summary.npy` | 0.158 / 0.088 / 0.046 |
| S7, positive control (pattern) | `roi_synth_pattern_8subj_summary.npy` | 0.217 / 0.164 / 0.128 |
| S7, positive control (chromatic ID) | `roi_synth_chromID_8subj_summary.npy` | 0.159 / 0.114 / 0.116 |
| S3, split-half reliability | `reliability_v2_8subj_summary.npy` | r = 0.90 / 0.88 / 0.84 |
| S4, signal quality | `ncsnr_quality_v2_ncsnr.npy` | ncsnr 0.412 / 0.353 / 0.246 |

> **`_v2` = computed after the webdataset shard fix (§4b).** The files without
> `_v2` are the pre-fix runs; they differ by ≤0.0008 in every group mean and
> change no ordering or conclusion (`python compare_runs.py`), but the paper
> cites `_v2` throughout. The NSD-synthetic controls have no `_v2` because they
> never used the affected loader.

---

## 3. Superseded files — **do not cite these**

Kept for history; they are *not* the reported analyses.

| File | Why superseded |
|---|---|
| `roi_color_8subj_summary.npy` (0.032/0.029/0.045) | uniform pooling; the website's original numbers |
| `roi_luminance_8subj_summary.npy` (0.029/0.012/0.017) | uniform pooling |
| `roi_colorresid_8subj_summary.npy` | uniform pooling |
| `roi_color_7subj_summary.npy` | earlier 7-subject run |
| `roi_color_mlp_8subj_summary.npy` (−0.268/−0.261/−0.308) | **unregularised** MLP; overfit. Replaced by `mlp2` |
| `roi_fgcolor_vw` / `roi_bgcolor_vw` | full 73k image set — Methods describe the *clean* subset, so the paper uses `cleanvw` |
| `roi_fgcolor_clean` / `roi_bgcolor_clean` | clean subset but *uniform* pooling |
| `roi_fgcolor_8subj` / `roi_bgcolor_8subj` | full set, uniform pooling |
| `roi_synth_hue_avg*` | trial-averaged variants of the synthetic hue analysis |
| all `*_8subj_summary.npy` without `_v2` | pre-fix webdataset shard selection (§4b); kept for the before/after comparison in `compare_runs.py` |

> ⚠️ **The fg/bg family is the easiest to confuse** — four variants exist that
> differ in both image subset and R² pooling, and they do not all agree in sign.
> The paper uses **`_cleanvw`** (clean subset + variance weighting).

---

## 4. The clean foreground/background subset

Built by `fg_bg_color_targets` with a single-dominant-object filter and coverage
bounds. Recoverable from the saved targets:

- `data/fgbg_clean_fg.npy` / `data/fgbg_clean_bg.npy` — **16,808 images** (of 73,000, 23%)
- `data/fgbg_clean_coverage.npy` — box coverage ranges **0.10 – 0.70** of the frame
  (mean 0.32), confirming the coverage filter
- Full (unfiltered) versions: `data/fgbg_fg.npy` / `data/fgbg_bg.npy` — 73,000 images

---

## 4a. Region naming

The three ROI keys in `roi.py` are `early_v1v3`, `v4_color` and **`concept`**.
`concept` is our own shorthand, not a term from the literature — it is the
`higher_vis` mask, i.e. **everything in `nsdgeneral` beyond V1–V4** (participant
1: 11,067 of 15,724 voxels, 70%). The field's term for this territory is
**higher visual cortex**; "conceptual" as a regional label usually refers to
anterior temporal cortex, which is *not* in this mask.

The manuscript therefore says **higher visual cortex** throughout. The code key,
the `agg[...]` dictionary keys and the result filenames still say `concept`, and
are left unchanged so existing `.npy` files keep working.

---

## 4b. Webdataset shard selection (affects every decoding run)

The MindEye2 webdataset ships **three** directories per participant:
`wds/subjXX/{train, test, new_test}`. There is no `new_train`. The README warns:

> *"Always use `new_test` instead of `test` in the wds folders, `test` refers to
> using the old NSD data from before they released the full set of scanning
> sessions."*

Selecting shards by the substring `"test"` matches both `test` and `new_test`,
so every held-out trial is read twice. For participant 3 that inflated the
held-out set from 2,371 to 4,484 trials (2,113 duplicates). Training data was
unaffected, and there was no leakage — but the held-out average was weighted
~2x toward the duplicated images.

**Fixed:** `_select_shard_dirs` reads the actual listing and drops a legacy
directory only when its `new_` replacement exists → `train` + `new_test`.
Held-out status now comes from behav column 16 (`shared1000`), which was
verified to agree with the directory partition without exception:

| participant 3 | trials | images |
|---|---|---|
| training (`train`, flag = not shared) | 21,629 | 8,481 |
| held out (`new_test`, flag = shared1000) | 2,371 | 930 |
| total | 24,000 = betas rows | 9,411 |

Overlap between the two partitions: 0 trials, 0 images, 0 shared-1000 images in
training. Results produced **before** this fix carry no `_v2` suffix; see
`rerun_after_split_fix.sh` and `compare_runs.py`.

---

## 4c. Solver settings (elastic-net)

The elastic-net robustness runs use `MultiTaskElasticNetCV(l1_ratio=0.5, cv=3,
alphas=8, max_iter=3000)`. Two **solver-level** settings were changed after the
`_v2` reruns began, purely for speed:

| setting | before | after | why |
|---|---|---|---|
| `n_jobs` | 1 | 3 | the 3 CV folds run in parallel. The original comment said this avoided "per-core data copies", which held when `X` was a whole ROI; after voxel matching `Xtr` is k=397 columns (~34 MB), so 3 workers cost ~100 MB. |
| `selection` | cyclic (default) | `"random"`, `random_state=0` | neighbouring voxels are highly correlated and cyclic coordinate descent zig-zags on correlated designs. |

Neither changes the objective, the penalty, or its minimiser — only the route to
it. **Verified rather than assumed:** `test_elasticnet_speedup.py` refits a
participant already computed under the old settings and compares per ROI, with a
pass threshold of |Δ| < 0.002 (an order of magnitude below the smallest reported
effect, +0.006). It passed.

Coordinate descent does not converge within `max_iter=3000` on these data
(ConvergenceWarning, duality gap ~1.3–7x tolerance). This is expected for
correlated fMRI predictors and is disclosed in the supplement. Raising
`max_iter` is the wrong lever: every fit already runs the full 3000 sweeps, so
raising it multiplies runtime without changing the reported conclusion.

---

## 5. Commands

```bash
# ---- targets -------------------------------------------------------------
python -m brain2vision.color_targets       --images data/coco_images_224_float16.hdf5 --out data/color_targets.npy
python -m brain2vision.luminance_targets   --images data/coco_images_224_float16.hdf5 --out data/luminance_targets.npy
python -m brain2vision.perceptual_color_targets --images data/coco_images_224_float16.hdf5 \
       --w2c data/w2c.npy --out data/color_targets_perceptual.npy
python -m brain2vision.semantic_targets    # 80-way COCO object presence
python -m brain2vision.semantic_residual   # residualise colour against objects (5-fold CV ridge)
python -m brain2vision.fg_bg_color_targets --images data/coco_images_224_float16.hdf5 \
       --single-object --coverage-min 0.10 --coverage-max 0.70 --out data/fgbg_clean

# ---- main decoding (all reported runs use --r2-weighting variance) --------
S="--subjects 1 2 3 4 5 6 7 8 --r2-weighting variance"
python -m brain2vision.replicate_subjects $S --target data/color_targets.npy            --out roi_color_vw_8subj.png
python -m brain2vision.replicate_subjects $S --target data/luminance_targets.npy \
       --labels L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10                                       --out roi_luminance_vw_8subj.png
python -m brain2vision.replicate_subjects $S --target data/color_targets_residual.npy   --out roi_colorresid_vw_8subj.png
python -m brain2vision.replicate_subjects $S --target data/color_targets_perceptual.npy --out roi_pcolor_vw_8subj.png
python -m brain2vision.replicate_subjects $S --target data/color_targets_perceptual_residual.npy --out roi_pcolorresid_vw_8subj.png
python -m brain2vision.replicate_subjects $S --target data/fgbg_clean_fg.npy            --out roi_fgcolor_cleanvw_8subj.png
python -m brain2vision.replicate_subjects $S --target data/fgbg_clean_bg.npy            --out roi_bgcolor_cleanvw_8subj.png

# ---- decoder robustness (nonlinear ones are slow: use --n-draws 3) --------
for M in elasticnet svr kernel mlp; do
  python -m brain2vision.replicate_subjects $S --target data/color_targets.npy \
         --model $M --n-draws 3 --out roi_color_${M}_8subj.png
done

# ---- controls ------------------------------------------------------------
python -m brain2vision.reliability     $S --target data/color_targets.npy --out reliability_8subj.png
python -m brain2vision.ncsnr_quality   --subjects 1 2 3 4 5 6 7 8 \
       --decode-summary roi_color_vw_8subj_summary.npy --out ncsnr_quality.png
python -m brain2vision.synthetic_decode --mode hue      # + --mode identity / pattern

# ---- statistics ----------------------------------------------------------
python -m brain2vision.stats --color roi_color_vw_8subj_summary.npy \
                             --luminance roi_luminance_vw_8subj_summary.npy
python -m brain2vision.stats --raw roi_color_vw_8subj_summary.npy \
                             --residual roi_colorresid_vw_8subj_summary.npy
python -m brain2vision.stats --fg roi_fgcolor_cleanvw_8subj_summary.npy \
                             --bg roi_bgcolor_cleanvw_8subj_summary.npy
```

### 5.1 How these commands were reconstructed

Runs made after the checkpointing code was added **store their own configuration**
inside the `.npy`. Earlier runs do not, so their flags were recovered from
evidence in the saved outputs. Each row below states which.

| File | `--model` | `--r2-weighting` | `k` | Evidence |
|---|---|---|---|---|
| `roi_color_vw` | ridge | variance | 397 | pooling inferred (see 5.2) |
| `roi_color_elasticnet` | elasticnet | variance | 397 | **stored in file** |
| `roi_color_svr` | svr | variance | 397 | **stored in file** |
| `roi_color_kernel` | kernel | variance | 397 | **stored in file** |
| `roi_color_mlp2` | mlp | variance | 397 | **stored in file** |
| `roi_color_mlp` (superseded) | mlp | variance | 397 | **stored in file** |
| `roi_luminance_vw` | ridge | variance | 397 | **stored in file** |
| `reliability_8subj` | ridge | variance | 397 | **stored in file** |
| `roi_colorresid_vw` | ridge | variance | — | pooling inferred |
| `roi_pcolor_vw`, `roi_pcolorresid_vw` | ridge | variance | — | pooling inferred |
| `roi_fgcolor_cleanvw`, `roi_bgcolor_cleanvw` | ridge | variance | — | pooling inferred; subset from `fgbg_clean_*` |
| `roi_*_8subj` (no suffix) | ridge | **uniform** | — | pooling inferred |
| `roi_fgcolor_clean`, `roi_bgcolor_clean` | ridge | **uniform** | — | pooling inferred |

`--subjects 1 2 3 4 5 6 7 8` for every file except `roi_color_7subj` (n = 7),
confirmed from the `subjects` field. The target is identified by the `labels`
field (11 colour terms vs `L0…L10` luminance bins).

**Not recoverable:** `--n-draws` is not stored and leaves no trace in the output.
Reported linear runs used 25 and nonlinear runs 3; the result is insensitive to
this (draw-to-draw variance is far smaller than between-participant variance).

### 5.2 Reconstructing the pooling flag empirically

`replicate_subjects` saves both the per-target R² (`agg[roi]["per"]`) and the
pooled overall R² (`agg[roi]["ov"]`). So the flag can be tested directly:

```python
import numpy as np
d   = np.load("roi_color_vw_8subj_summary.npy", allow_pickle=True).item()
per = np.array(d["agg"]["concept"]["per"])   # per-colour R², per subject
ov  = np.array(d["agg"]["concept"]["ov"])    # pooled R², per subject
np.allclose(per.mean(1), ov, atol=2e-3)      # True -> uniform; False -> variance-weighted
```

For **colour** targets this is decisive: near-absent bins (purple) make uniform
pooling differ sharply from variance weighting (e.g. higher visual cortex 0.044 vs 0.066).

⚠️ For **luminance** it is *not* decisive — the 11 brightness bins are evenly
populated, so the two poolings agree to ~0.0017 (0.0171 vs 0.0156). The stored
config in `roi_luminance_vw` confirms `variance`. This is also why the
colour-minus-luminance interaction is unchanged (+0.024) under either pooling.

---

## 6. Verification

Correctness of the analysis code is checked by planting a known answer and
confirming the code recovers it — and planting a known null and confirming it
reports nothing. Run:

```bash
python test_analysis.py          # 26 checks, no fMRI data required
```

| What is checked | Result |
|---|---|
| Decoder recovers a planted linear signal | R² = +0.51 |
| Shuffled targets destroy decoding | R² = −0.01 |
| Pure-noise voxels do not decode | R² = −0.01 (RidgeCV selects max regularisation) |
| Residual is orthogonal to object regressors | max \|r\| = 0.013 |
| Residual **retains** a planted object-independent component | r = 0.98 |
| Residual **removes** a planted object-predicted component | r = 0.02 |
| A near-constant bin distorts uniform pooling | uniform +0.015 vs true +0.902 |
| Variance weighting tracks the informative target | +0.902 |
| Permutation test calibrated under the null | P(p<0.05) = 0.062 |
| Permutation floor is exactly 2/2⁸ | p = 0.0078 |
| BH-FDR matches the textbook computation | ✓ (monotone, q ≥ p) |
| Split-half halves are image-disjoint | overlap = 0 |
| All decoders recover the same planted signal | ridge/elastic-net/SVR ≈ 0.47–0.51; kernel 0.20 |

### Data plumbing

`test_analysis.py` verifies the maths; `test_plumbing.py` verifies that the right
numbers reached it, running on the **actual derived targets used in the paper**:

```bash
python test_plumbing.py          # 21 checks
```

| What is checked | Result |
|---|---|
| All seven targets share identical image ids | ✓ |
| coverage·fg + (1−coverage)·bg reconstructs the full-image histogram | median error **0.00000** over 70,079 images |
| Object information reduced in the residual target | max \|r\| **0.208 → 0.002** |
| Same for the perceptual residual | ✓ |
| Residual retains variance (not over-removed) | sd 0.139 → 0.119 |
| Targets are valid distributions (sum to 1, non-negative) | ✓ |
| Residual sums to ~0 per row | −0.00000 |
| Perceptual and physical targets are related but distinct | median r = **+0.76** |
| Clean fg/bg subset size | **16,808** images |
| Clean subset coverage bounds | **[0.10, 0.70]** |
| Clean ids are a subset of the full id set | ✓ |

⚠️ **What the fg/bg reconstruction does and does not show.** The exact
reconstruction (error 0.00000) verifies *arithmetic consistency*: the two
histograms were computed from the same images as the main colour target, the
in-box / out-of-box masking has no gaps or double-counting, and the coverage
values correctly describe the area split. It is a plumbing check on the masking
code.

It is **not** evidence that the split cleanly separates *foreground* from
*background* as perceptual categories. A COCO bounding box is rectangular, so the
"foreground" region contains surrounding background pixels, and the "background"
region may contain other objects. That operationalisation is deliberately coarse
— which is why the paper describes the resulting test as conservative and notes
that pixel-accurate segmentation would sharpen it. The arithmetic is exact; the
construct is approximate.

The clean-subset check independently confirms the filter parameters recorded in
§4, which were originally reconstructed from the saved outputs rather than from a
run log.

Two notes. The kernel decoder scoring lower on a **known-linear** planted signal
independently reproduces the paper's finding that nonlinear decoders do not
exceed linear ones — it is expected behaviour, not a data quirk. And these tests
verify the **analysis logic**, not the data plumbing; the betas↔image alignment is
verified separately by falsification in
[`brain2vision/alignment_check.py`](brain2vision/alignment_check.py), where
permuting the image labels abolishes decoding.

Passing tests do not prove correctness. They fail loudly when something is wrong,
which is the achievable standard.

---

## 7. Statistical conventions

- Group inference treats the **8 participants** as the unit of analysis.
- **Exact paired sign-flip permutation** (2⁸ = 256 sign vectors). Two-sided *p*
  has a floor of 2/256 = **0.008**, so `**` (q < 0.01) is the strongest evidence
  attainable at this sample size and `***` is impossible.
- **95% CI** = 20,000-sample subject-level paired-difference bootstrap.
- **q** = Benjamini–Hochberg FDR within each family of contrasts.
- The decisive tests are **interactions**, not single differences:
  colour-minus-luminance across regions, and the region-by-residualisation
  interaction (does decoding fall *more* in one region than another).

---

## 7a. Per-draw R² and the voxel-sampling asymmetry (added 13 Aug 2026)

Voxel matching gives the decoder k = 397 voxels for every region, but that is
~58% of V4 and only ~3.6% of higher visual cortex. To test whether the
object-bound colour signal could be confined to a small sub-region,
`replicate_subjects._matched_draws` now also returns the **individual per-draw
R² values** (field `ovd` in the summary), not just their mean.

The change is additive: `ov` is still the mean over draws and is byte-identical
to before — verified against the canonical run, `max |diff| = 0.00e+00` in all
three regions.

```bash
# per-draw values (NOTE: --r2-weighting variance is required; the flag
# defaults to "uniform" and omitting it silently runs a different analysis)
python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 \
    --r2-weighting variance \
    --target data/color_targets.npy --out roi_color_vwdraws_8subj.png

python draw_variance.py roi_color_vwdraws_8subj_summary.npy
```

`draw_variance.py` prints the run's config and warns if it differs from the
canonical summary, because the weighting flag was omitted once and produced
plausible-looking but non-comparable numbers.

**Result (Extended Data S9, Table S4).** 192 of 200 draws (96%) from higher
visual cortex exceeded the same participant's near-complete V4 sample; every
draw did so in 6 of 8 participants. Across-draw SD was smaller than
between-participant SEM in every region (0.58 early, 0.26 V4, 0.84 higher).

**What this licenses:** the signal cannot be confined to a small sub-region,
since a random 3.6% sample would then miss it. **What it does not licence:** a
claim that the signal is uniformly or "pervasively" distributed — the result is
not unanimous, and two participants had draws falling marginally below their own
V4 value.

---

## 7b. Scene-content residualisation with COCO-Stuff (added 13 Aug 2026)

The 80 COCO categories are all **things** — discrete objects. They contain no
sky, grass, road, water or wall, which are exactly the correlations the
Introduction uses to motivate residualisation. `brain2vision/stuff_targets.py`
adds the 91 **COCO-Stuff** categories, binary presence, encoded identically to
things so the comparison isolates content type rather than representation.

```bash
python -m brain2vision.stuff_targets --out data/stuff_targets.npy
python -m brain2vision.stuff_targets --combine data/semantic_targets.npy \
    --out data/thingstuff_targets.npy

python -m brain2vision.semantic_residual --color data/color_targets.npy \
    --semantic data/thingstuff_targets.npy --out data/color_targets_bothresid.npy

python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 \
    --r2-weighting variance \
    --target data/color_targets_bothresid.npy --out roi_colorbothresid_vw_8subj.png
```

**Validation of the stuff vector** (before any decoding): each category pulls the
colour it should — grass and tree → green, sky and sea → blue, snow → white,
with effects of +0.05 to +0.21 on the colour histogram. Categories that appear
not to (dirt, wood, clouds) are explained by co-occurrence, e.g. P(tree | dirt) =
0.63 against a base rate of 0.32.

**Image-statistics result** (Extended Data S10, Table S5). Variance-weighted
out-of-sample R²(colour ~ content): things 0.126, stuff 0.146, both 0.174 —
**56% shared**. Object labels partly proxy for scene context: P(snow | skis) =
0.95, P(railroad | train) = 0.73.

**Brain result** (Table S6). Removing all 171 categories left **83%** of the
colour-target variance intact (effective dimensionality 6.0 vs 5.9 raw) and did
not eliminate the reversal: early 0.012, V4 0.007, higher 0.005; early − higher
= +0.007, p = 0.008, 8/8. Decomposed, the effect is **achromatic** (early 0.028
vs higher 0.007; +0.021, p = 0.008, 8/8); the chromatic residual is ≤ 0.003
everywhere.

**Not reported, deliberately.** V4 exceeded early cortex on the *chromatic*
residual at p = 0.039 (6/8) under the 171-category scheme but not under the
80-category one (p = 0.148, 5/8). The contrast was identified after inspecting
the data and does not replicate across schemes. Recorded so the decision not to
report it is auditable.

### The colour/luminance asymmetry (Extended Data S10, Tables S7–S8)

The same 171-category residualisation applied to the **luminance** target. Note
`--labels`: without it the summary records brightness bins under colour names.

```bash
python -m brain2vision.semantic_residual --color data/luminance_targets.npy \
    --semantic data/thingstuff_targets.npy \
    --out data/luminance_targets_bothresid.npy

python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 \
    --r2-weighting variance \
    --labels "L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10" \
    --target data/luminance_targets_bothresid.npy \
    --out roi_lumbothresid_vw_8subj.png
```

Annotated content explains **2.5× more of image colour than of image luminance**
(variance-weighted out-of-sample R² 0.174 vs 0.069), leaving 83% vs 93% of the
respective targets. Decoding the luminance residual: **early 0.0096**
(p = 0.008, 8/8, CI [+0.006, +0.013]), **V4 −0.0008** (p = 0.594) and
**higher +0.0005** (p = 0.680) — neither distinguishable from zero. Every
participant shows early > V4 and early > higher (8/8 each).

So content-independent **luminance** is carried by early retinotopic cortex
alone, while content-independent **hue** is not reliably carried anywhere
(chromatic residual ≤ 0.003 in every region). This asymmetry replaced the
Discussion's earlier claim that chromatic content should be sourced from the
early stream — which these data falsify.

**Labelling fix (13 Aug).** `semantic_residual` previously printed every target's
columns as the 11 colour names, so luminance bins were reported as "red, orange,
yellow…". Column names are now derived from the target filename, overridable with
`--labels`, and the module prints both uniform and variance-weighted R² so the
two are not conflated. Numbers were never affected — only labels.

---

## 7c. The "reversal" was a change of mixture (corrected 13 Aug 2026)

**Nina caught this.** The paper claimed that residualisation *reversed the
regional ordering*. It does not. The aggregate ordering changes, but neither
component ordering does.

The 11-term colour target contains 3 achromatic terms (black, white, grey)
carrying **43%** of its variance. Decomposing by subset:

| target | chromatic early | chromatic higher | achromatic early | achromatic higher |
|---|---|---|---|---|
| raw colour | 0.052 | **0.081** (p=0.008, 8/8) | **0.057** (p=0.023, 7/8) | 0.041 |
| residual, things (80) | 0.007 | 0.007 (n.s., 4/8) | **0.036** (p=0.008, 8/8) | 0.011 |
| residual, +stuff (171) | 0.001 | 0.002 (n.s., 4/8) | **0.028** (p=0.008, 8/8) | 0.007 |

Higher visual cortex leads on chromatic terms and early cortex on achromatic
terms **at every stage, including the raw target**. Residualisation removes the
chromatic component — the one higher visual cortex led on — almost entirely, and
spares the achromatic one. The aggregate flips because the *mixture* changes.

Arithmetic check: raw higher = 0.429×0.041 + 0.571×0.081 = 0.064 ≈ 0.066 ✓;
residual early = 0.429×0.036 + 0.571×0.007 = 0.019 ≈ 0.018 ✓.

**What was rewritten.** "Reversal" language removed from the Abstract,
Significance Statement, Results, Discussion, Figure 4 caption, Extended Data S2
and S10, and the contrast names in `build_stats_table.py` / `make_figures.py`
("Reversal" → "Residual target"). The claim is now that the two arms of the
dissociation differ in whether they survive content removal — which is what the
data show, and is stronger because a reviewer running this decomposition finds
the same thing.

**Reproduce it:**

```python
# per-subject per-colour R² are in agg[roi]["per"]; weights from the target variance
w = np.load('data/color_targets.npy').var(0); w /= w.sum()
ia = [8, 9, 10]                      # black, white, gray
ic = [0,1,2,3,4,5,6,7]               # the eight chromatic terms
# renormalise within each subset before comparing regions
```

---

## 7d. Trial bookkeeping, all eight participants (14 Aug 2026)

`python check_betas.py` — run with the full participant set.

| subj | sessions | trials | training | held out | unique images | duplicate reads |
|---|---|---|---|---|---|---|
| 1 | 40 | 30,000 | 27,000 | 3,000 | 10,000 | 0 |
| 2 | 40 | 30,000 | 27,000 | 3,000 | 10,000 | 0 |
| 3 | 32 | 24,000 | 21,629 | 2,371 | 9,411 | 0 |
| 4 | 30 | 22,500 | 20,312 | 2,188 | 9,209 | 0 |
| 5 | 40 | 30,000 | 27,000 | 3,000 | 10,000 | 0 |
| 6 | 32 | 24,000 | 21,629 | 2,371 | 9,411 | 0 |
| 7 | 40 | 30,000 | 27,000 | 3,000 | 10,000 | 0 |
| 8 | 30 | 22,500 | 20,312 | 2,188 | 9,209 | 0 |
| **total** | | **213,000** | **191,882** | **21,118** | | **0** |

Arithmetic verified: sessions × 750 = trials for every participant, and
training + held out = trials for every participant.

**The legacy-shard fix holds universally.** Every participant reports
`dropping superseded ['test'] (the 'new_' versions are present)` and
`duplicate (row,id) reads: 0`. The bug found on 12 Aug is confirmed fixed across
all eight, not just the ones checked at the time.

Held-out counts fall slightly below nominal for the shorter participants —
2,371 against ~2,400 and 2,188 against ~2,250 — because fewer completed sessions
means fewer shared-1000 repeats. Expected, not an error.

Reported in Methods and Extended Data S11 (Table S10).

---

## 8. Environment

Python 3.10 · scikit-learn 1.9 · NumPy 2.5 · SciPy 1.18 · nibabel 5.4 ·
h5py 3.16 · Matplotlib 3.11

Data: NSD via the MindEye2 release (`pscotti/mindeyev2`) for the main analyses;
raw NSD (AWS `natural-scenes-dataset`, OpenNeuro ds004496) for NSD-synthetic and
ncsnr; COCO annotations for object targets. Per the NSD Terms of Use, raw betas
and masks are **not** redistributed here.
