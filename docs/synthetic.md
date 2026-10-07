# NSD-synthetic control: color without content

## Why

On NSD-core (natural images) the `concept` pool decoded image color best. But
residualizing color against object presence collapsed that advantage across
*all* ROIs, leaving it ambiguous whether higher visual cortex has a genuine
chromatic code or was reading object/scene identity (and the color of the
attended foreground). Residualization cannot separate these, because it removes
the color/content *shared* variance, which is unattributable.

NSD-synthetic breaks the tie. Among its 284 images are **64 isoluminant
chromatic pink-noise images spanning 16 hues** (Hue01–Hue16). These have no
objects, no figure to attend, and constant luminance, so a decodable hue signal
there is genuinely chromatic: it cannot be a content, attentional-foreground,
or luminance artifact.

The NSD-synthetic data paper (Gifford et al., *Nat. Commun.* 2026) reports that
chromatic noise activates early visual cortex (EVC) more than higher visual
cortex (HVC). Our test is the stricter, voxel-matched *decoding* complement:
can each ROI read out the hue, per unit of cortex?

## Prediction

- If the NSD-core "concept decodes color best" result was structure-bound, the
  concept advantage should **not** reappear here, and early visual cortex should
  decode hue at least as well.
- If the concept pool has a genuine chromatic code, it should still decode hue
  competitively.

## Data (raw NSD, S3 bucket `natural-scenes-dataset`; agree to the NSD terms)

| what | path |
|---|---|
| synthetic betas | `nsddata_betas/ppdata/subjXX/func1pt8mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/betas_nsdsynthetic.hdf5` |
| ROI atlases | `nsddata/ppdata/subjXX/func1pt8mm/roi/prf-visualrois.nii.gz`, `.../streams.nii.gz` |
| color stimuli (per subject) | `nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_colorstimuli_subjXX.hdf5` |
| design (trial→image order) | `nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat` |

Notes on layout, learned the hard way:

- The **betas are single-trial**: 744 rows/subject (93 trials × 8 runs), in
  presentation order, *not* condition-averaged. So the trial→image map must come
  from the design file's ordering (a length-744 field of image numbers 1..284).
  Pass it with `--design`; the loader auto-detects the ordering field.
- The **color stimuli are per-subject** (`..._subjXX.hdf5`), because the
  chromatic images were made isoluminant per participant. The 16 hue classes are
  identical across subjects, so targets built from one subject's file apply to
  all; use per-subject files only if you want maximal rigor.
- Verify exact filenames against the NSD manual; the loaders accept overrides
  (`--betas-name`, `--images`, `--design`) and print the shapes they load.

## Run

```bash
pip install -e ".[color,rawnsd]"   # sklearn, matplotlib, nibabel, scipy(+core)

# download one subject's color stimuli + the design file (public bucket)
aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_colorstimuli_subj01.hdf5 . --no-sign-request
aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat . --no-sign-request

# 1. hue targets from the color stimuli pixels (once; hue is subject-invariant)
python -m brain2vision.synthetic_targets \
    --images nsdsynthetic_colorstimuli_subj01.hdf5 --out data/synth

# 2. voxel-matched hue decode across ROIs and subjects
python -m brain2vision.synthetic_decode --subjects 1 2 3 4 5 6 7 8 \
    --targets data/synth --design nsdsynthetic_expdesign.mat \
    --out roi_synth_hue_8subj.png
```

## Method notes

- Hue is decoded as a circular `(cos, sin)` target (avoids the 0°/360° wrap),
  reported as out-of-sample R².
- Evaluation is **GroupKFold by image**, so an image's two task conditions never
  straddle the train/test split.
- Each ROI is subsampled to a common voxel count `k` (smallest ROI across the
  included subjects) and averaged over random draws, identical matching logic to
  the NSD-core comparison.
- `early_v1v3` and `v4_color` come from the `prf-visualrois` atlas; `concept`
  from the high-level `streams` regions.
