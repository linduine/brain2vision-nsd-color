# Provenance: which file backs which result

Every number in the preprint (*"Is colour in human visual cortex chromatic or
semantic?"*) traces to a `*_summary.npy` file in the repository root. This
document maps result → file → command, so the analysis can be re-run or audited
without guesswork.

**Last verified:** all values below were recomputed from the `.npy` files and
matched the manuscript.

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

> **Why `_vw` matters.** Several colours (notably purple) are near-absent in most
> images, so a uniform mean over per-colour R² is dominated by unpredictable
> bins. Variance weighting lets each colour contribute in proportion to how much
> it actually varies. **All headline numbers use `_vw`.**

---

## 2. Files that back the paper (the canonical set)

| Paper element | File | Values (early / V4 / concept) |
|---|---|---|
| Fig 1, Table 1 "Ridge" | `roi_color_vw_8subj_summary.npy` | 0.054 / 0.049 / 0.067 |
| Table 1 "Elastic-net" | `roi_color_elasticnet_8subj_summary.npy` | 0.053 / 0.047 / 0.065 |
| Table 1 "Linear SVR" | `roi_color_svr_8subj_summary.npy` | 0.049 / 0.045 / 0.062 |
| Table 1 "RBF kernel" | `roi_color_kernel_8subj_summary.npy` | 0.045 / 0.038 / 0.053 |
| Table 1 "MLP" | `roi_color_mlp2_8subj_summary.npy` | 0.051 / 0.046 / 0.065 |
| Luminance / dissociation | `roi_luminance_vw_8subj_summary.npy` | 0.027 / 0.010 / 0.015 |
| Fig 2, collapse (physical) | `roi_colorresid_vw_8subj_summary.npy` | 0.018 / 0.012 / 0.009 |
| Fig 2, raw perceptual | `roi_pcolor_vw_8subj_summary.npy` | 0.066 / 0.067 / 0.090 |
| Fig 2, collapse (perceptual) | `roi_pcolorresid_vw_8subj_summary.npy` | 0.018 / 0.016 / 0.012 |
| Fig 3, attention (foreground) | `roi_fgcolor_cleanvw_8subj_summary.npy` | 0.017 / 0.019 / 0.022 |
| Fig 3, attention (background) | `roi_bgcolor_cleanvw_8subj_summary.npy` | 0.033 / 0.020 / 0.029 |
| S7, synthetic hue null | `roi_synth_hue_8subj_summary.npy` | −0.117 / −0.073 / −0.066 |
| S7, positive control (identity) | `roi_synth_identity_8subj_summary.npy` | 0.158 / 0.088 / 0.046 |
| S7, positive control (pattern) | `roi_synth_pattern_8subj_summary.npy` | 0.217 / 0.164 / 0.128 |
| S7, positive control (chromatic ID) | `roi_synth_chromID_8subj_summary.npy` | 0.159 / 0.114 / 0.116 |
| S3, split-half reliability | `reliability_8subj_summary.npy` | r = 0.91 / 0.89 / 0.86 |
| S4, signal quality | `ncsnr_quality_ncsnr.npy` | ncsnr 0.412 / 0.353 / 0.246 |

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

> The exact flags for some early runs were not logged. Where a flag is uncertain
> it has been **reconstructed from the saved outputs** (e.g. the fg/bg coverage
> bounds were read back from `data/fgbg_clean_coverage.npy`). Re-running the
> commands above reproduces the reported values.

---

## 6. Statistical conventions

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

## 7. Environment

Python 3.10 · scikit-learn 1.9 · NumPy 2.5 · SciPy 1.18 · nibabel 5.4 ·
h5py 3.16 · Matplotlib 3.11

Data: NSD via the MindEye2 release (`pscotti/mindeyev2`) for the main analyses;
raw NSD (AWS `natural-scenes-dataset`, OpenNeuro ds004496) for NSD-synthetic and
ncsnr; COCO annotations for object targets. Per the NSD Terms of Use, raw betas
and masks are **not** redistributed here.
