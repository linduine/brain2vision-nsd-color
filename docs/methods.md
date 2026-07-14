# NSD methods notes

Load NSD single-trial betas restricted to specific visual ROIs, instead of the
whole `nsdgeneral` region — plus CLIP / color / bounding-box targets and the
color-decoding experiments. Each item below is a module run with
`python -m brain2vision.<name>`.

| module | purpose |
|--------|---------|
| `inspect_rois`         | list ROI region names + voxel counts per subject |
| `roi`                  | load betas for the 3 named ROI sets (early / V4 / concept) |
| `raw_nsd`              | fine-grained ROIs (FFA, PPA, V1v, …) from raw NSD |
| `bboxes`               | COCO bounding boxes per image, in the NSD 425×425 frame |
| `visualize`            | draw boxes on a stimulus image to sanity-check the transform |
| `clip_targets`         | CLIP image/text embeddings (decoder targets) |
| `color_targets`        | per-image 11-way basic-color distribution (color target) |
| `color_decode`         | decode image colors from V4 betas + evaluate (single subject) |
| `compare_rois`         | color decoding across early/V4/concept ROIs + bar plot |
| `color_shared_subject` | shared-subject V4 color decoder (subjects 1,2,5,7) |

## Setup

```bash
pip install -e .
```

## Step 1 — see which ROIs exist

```bash
python -m brain2vision.inspect_rois
```

Downloads only `brain_region_masks.hdf5` (~0.8 MB) from `pscotti/mindeyev2` and
prints every region name + voxel count per subject. Use these names in step 2.
Regions in this release typically include early visual areas (`V1`, `V2`, `V3`,
`V4`) and coarser "streams" regions (early / midventral / midlateral /
midparietal / ventral / lateral / parietal) — confirm the exact strings from the
script output.

## Step 2 — load betas for chosen ROIs

Three named ROI sets are predefined in `nsd_roi_loader.py` (`ROI_SETS`), loaded
independently:

| set key      | intent                        | region(s) | voxels (subj01) |
|--------------|-------------------------------|-----------|-----------------|
| `early_v1v3` | early visual (low-level)      | `V1`, `V2`, `V3` | ~3,970    |
| `v4_color`   | V4 — color / intermediate form| `V4`      | ~687            |
| `concept`    | all higher visual cortex      | `higher_vis` | ~11,067     |

Verified against `brain_region_masks.hdf5`, which stores per subject a clean
partition: `V1`, `V2`, `V3`, `V4`, `early_vis` (= V1–V4 union), `higher_vis`,
and `nsd_general` (= `early_vis` + `higher_vis`). Matching defaults to exact.

Load one set:

```bash
python -m brain2vision.roi --subj 1 --set early_v1v3 --preview
```

Load all three separately (downloads the subject betas once, reuses it):

```bash
python -m brain2vision.roi --subj 1 --all-sets --preview
```

In code:

```python
from nsd_roi_loader import load_roi_set, ROI_SETS
early,  _ = load_roi_set(subj=1, set_name="early_v1v3")
v4,     _ = load_roi_set(subj=1, set_name="v4_color")
concept,_ = load_roi_set(subj=1, set_name="concept")
```

Or a fully custom fragment list:

```python
from nsd_roi_loader import load_roi_betas
betas, roi_mask = load_roi_betas(subj=1, rois=["V1", "V4"])
```

Matching is case-insensitive substring by default, so `V1` also catches `V1v`
/`V1d`. Pass `match="exact"` (or `--match exact`) if you want strict names.

### ⚠️ The `concept` set is `higher_vis`, not per-category floc ROIs

`brain_region_masks.hdf5` does **not** break the concept level into
`floc-faces` / `floc-places` / `floc-bodies` / `floc-words`. It only provides
`higher_vis` — the union of all higher visual cortex beyond V1–V4. So the
`concept` set is that whole block (~11k voxels for subj01).

If you need to separate faces vs places vs bodies vs words, this HF release
can't do it — use the raw NSD category masks (`floc-*.nii.gz` in
`nsddata/ppdata/subjXX/func1pt8mm/roi/`) and mask the volumetric betas yourself.

**That path is already written: the `raw_nsd` module** (`pip install -e
".[rawnsd]"`). It pulls raw NSD volumetric betas + ROI label volumes from the
public S3 bucket and extracts named sub-regions. Example:

```bash
pip install -e ".[rawnsd]"
# just FFA (both patches), subject 1, first 5 sessions as a smoke test
python -m brain2vision.raw_nsd --subj 1 --atlas floc-faces --regions FFA-1 FFA-2 --sessions 1 2 3 4 5
```

Available atlases and their sub-regions (`--atlas` / `--regions`):

| atlas             | sub-regions |
|-------------------|-------------|
| `prf-visualrois`  | V1v V1d V2v V2d V3v V3d hV4 |
| `floc-faces`      | OFA FFA-1 FFA-2 mTL-faces aTL-faces |
| `floc-places`     | OPA PPA RSC |
| `floc-bodies`     | EBA FBA-1 FBA-2 mTL-bodies |
| `floc-words`      | OWFA VWFA-1 VWFA-2 mfs-words mTL-words |
| `streams`         | early midventral midlateral midparietal ventral lateral parietal |

Use `--regions "*"` for all labeled voxels in an atlas. Caveats: requires
agreeing to NSD terms, downloads full volumetric betas (~1 GB/session,
streamed one session at a time), and the integer→name label maps should be
spot-checked against the `.ctab` files for a subject before you trust a
production run.

## How ROI selection works

Betas are stored per subject as one flattened `nsdgeneral` array
(`betas_all_subjXX_fp32_renorm.hdf5`, shape `(n_trials, n_nsdgeneral_voxels)`).
`brain_region_masks.hdf5` holds boolean masks over that same voxel axis. An ROI
is just the `True` columns of its mask: `betas[:, roi_mask]`. The loader unions
multiple ROIs and asserts mask length == betas voxel count so you can't silently
mis-align.

## Important download tradeoff

This preprocessed repo only ships the **full nsdgeneral** betas per subject
(~1.2–1.9 GB each). You download that one file, then keep only the ROI voxels.
So you avoid downloading all subjects and you train on only the ROI — but the
file you fetch is nsdgeneral-sized.

To download **only** an ROI's voxels (never the full nsdgeneral blob), use the
raw NSD instead:

- Source: AWS S3 bucket `natural-scenes-dataset` or OpenNeuro `ds004496`.
- Volumetric betas: `nsddata_betas/ppdata/subjXX/func1pt8mm/betas_fithrf_GLMdenoise_RR/`
- ROI NIfTI masks: `nsddata/ppdata/subjXX/func1pt8mm/roi/` — e.g.
  `prf-visualrois.nii.gz` (V1v/V1d/V2v/V2d/V3v/V3d/hV4),
  `floc-faces.nii.gz`, `floc-places.nii.gz`, `floc-bodies.nii.gz`,
  `floc-words.nii.gz`, `streams.nii.gz`, plus `nsdgeneral.nii.gz`.
- Load an ROI NIfTI, threshold to the label(s) you want, and index the beta
  volume by that mask. This gives arbitrary ROIs at the cost of downloading full
  volumetric betas.

Raw NSD requires agreeing to the NSD data-use terms. Don't redistribute NSD
betas or COCO images in your GitHub repo — ship this code + a download step and
have users accept the terms themselves.

## CLIP embeddings (decoder targets)

```bash
pip install -e ".[clip]"
```

Image embeddings (default ViT-L/14, light — good for the first experiment):

```bash
python -m brain2vision.clip_targets \
    --images coco_images_224_float16.hdf5 --out clip_vitL_img.npy
```

MindEye2 target (OpenCLIP ViT-bigG/14, heavy):

```bash
python -m brain2vision.clip_targets --images coco_images_224_float16.hdf5 \
    --model ViT-bigG-14 --pretrained laion2b_s39b_b160k --out clip_bigG_img.npy
```

Caption/text embeddings (semantic targets — pass a `{nsd_id: caption}` JSON):

```bash
python -m brain2vision.clip_targets --captions captions_by_nsdid.json \
    --out clip_vitL_text.npy
```

Output is a `.npy` of shape `(n_images, embed_dim)` plus a `_ids.npy` giving the
0-based NSD ids, so rows align to the betas via each subject's image index.
`--tokens` is stubbed (returns pooled embeddings for now); full patch-token
sequences for diffusion priors need a `net.visual` hook — noted in the script.

## Bounding boxes (COCO → NSD stimulus frame)

Every NSD image is a cropped COCO image, so COCO's instance boxes transfer over
— but they must be cropped/rescaled into the 425×425 stimulus the subject saw.
`build_coco_bboxes.py` does this using `nsd_stim_info_merged.csv`'s `cropBox`.

```bash
python -m brain2vision.bboxes --out nsd_bboxes.json          # all 73k images
python -m brain2vision.bboxes --nsd-ids 0 1 2 3 --out demo.json
```

Output JSON keyed by 0-based NSD id:

```json
{ "0": [ {"category": "dog", "bbox_xywh": [x, y, w, h]}, ... ] }
```

Coordinates are in 425×425 pixels (the `nsd_stimuli.hdf5` frame). It auto-
downloads `nsd_stim_info_merged.csv` from NSD and the COCO 2017 annotation zip
from cocodataset.org. Two things to verify before scaling up: the `cropBox`
convention is taken as `(top, bottom, left, right)` fractions — draw a few boxes
on the matching stimulus image to confirm — and NSD ids are treated as 0-based
aligned to the 73k image order used by the MindEye2 hdf5 files.

Draw the boxes to check the transform:

```bash
python -m brain2vision.visualize --images coco_images_224_float16.hdf5 \
    --bboxes nsd_bboxes.json --nsd-id 3 --out check_3.png
```

If the red boxes hug the objects, the crop convention is right. If they're
shifted/scaled, flip the `cropBox` order in `build_coco_bboxes.py`.

## Experiment: decode image colors from V4

V4 is strongly color-selective, so a natural first test is: can we predict the
colors in the seen image from V4 activity alone?

```bash
pip install -e ".[color]"
# 1) build the color targets (11-way basic-color distribution per image)
python -m brain2vision.color_targets --images coco_images_224_float16.hdf5 \
    --out color_targets.npy
# 2) train + evaluate (auto-downloads V4 betas, masks, behav, shared1000)
python -m brain2vision.color_decode --subj 1 --color-targets color_targets.npy --model ridge
```

What it does:

- **Target** — each image → an 11-dim distribution over red/orange/yellow/green/
  blue/purple/pink/brown/black/white/gray (fraction of pixels per basic color).
  Interpretable and richer than a single dominant color. Verified: all 11 pure
  swatches classify correctly.
- **Input** — the V4 voxels (`v4_color` set, ~687 for subj01) per trial.
- **Split** — test = images in the shared-1000 set, train = the rest (standard
  NSD held-out, no image leakage).
- **Model** — ridge regression by default (`--model mlp` for a small MLP).
- **Metrics** — overall and per-color R², plus dominant-color top-1 accuracy vs
  the 1/11 chance baseline.

### ⚠️ One alignment detail to verify once

MindEye2 betas are **not** in image order. `train_v4_color.py` recovers each
trial's image id and betas row from the webdataset `behav` arrays, using the
MindEye2 default columns (`behav[0,0]` = image id, `behav[0,5]` = betas row).
The script prints the id/row ranges and warns if they look wrong — glance at
that line on the first run. If your `behav` layout differs, set `IMG_COL` /
`BETAS_COL` at the top of the script. If you instead built alignment yourself
from `nsd_stim_info` (raw-NSD path), skip all that and pass `--ids-npy` /
`--betas-npy` directly.

### Compare ROIs

Run the same decoder on all three ROI sets and plot per-color R²:

```bash
python -m brain2vision.compare_rois --subj 1 --color-targets color_targets.npy \
    --out roi_color_compare.png
```

Produces a grouped bar chart (early_v1v3 / v4_color / concept × 11 colors) with
overall R² and top-1 in the legend — the clean ROI-comparison figure. Real
result, 4 subjects, matched to 687 voxels (see [`../REPORT.md`](../REPORT.md)):

![color decoding by ROI](../figures/fig1_color_by_roi.png)

### Multi-subject V4 (shared-subject model)

V4 is small per subject, so pool subjects. `train_v4_color_multisubject.py`
gives each subject its own linear projection from V4 into a common latent, with
a single shared readout, trained jointly on subjects 1, 2, 5, 7 (the four with
all 40 sessions):

```bash
pip install -e ".[color]"
python -m brain2vision.color_shared_subject --color-targets color_targets.npy \
    --subjects 1 2 5 7 --latent 512 --epochs 30
```

This genuinely adds training data (the readout learns a subject-invariant
latent→color map) rather than just averaging predictions. It reports per-subject
test R²/top-1 on each subject's shared-1000 set plus the cross-subject average,
and saves the model. Note: it downloads each subject's full nsdgeneral betas
(~1.5 GB × 4) to extract V4, so budget the disk/bandwidth.

### Other ideas

Predict mean hue/saturation instead of the histogram; restrict to the raw-NSD
`hV4` sub-region for a tighter color ROI; or add held-out subjects to test
whether a new subject's projection can be fit while freezing the shared readout.

### Tying it together

`clip_*_ids.npy` and the bbox JSON are both keyed by the same 0-based NSD id,
and each subject's betas carry an image-index array (e.g. `subjXX` columns in
`nsd_stim_info` or the MindEye2 `COCO_73k_subj_indices.hdf5`) that maps a trial
to its NSD id. So per trial you can line up: ROI betas ↔ CLIP target ↔ object
boxes — e.g. test whether concept-ROI decoding tracks the objects present.
