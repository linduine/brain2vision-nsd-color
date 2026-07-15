# Methods & module reference

Technical notes behind the study in [`../REPORT.md`](../REPORT.md): ROI-selective
NSD loading, the color/luminance/CLIP/bbox targets, the decoder, and — most
importantly — the methodology that makes the ROI comparison *fair*.

Each item below is a module run with `python -m brain2vision.<name>`.

| module | purpose |
|--------|---------|
| `inspect_rois`         | list ROI region names + voxel counts per subject |
| `roi`                  | load betas for the 3 named ROI sets (early / V4 / concept) |
| `raw_nsd`              | fine-grained ROIs (FFA, PPA, V1v, …) from raw NSD |
| `bboxes`               | COCO bounding boxes per image, in the NSD 425×425 frame |
| `visualize`            | draw boxes on a stimulus image to sanity-check the transform |
| `clip_targets`         | CLIP image/text embeddings (targets for reconstruction work) |
| `color_targets`        | per-image 11-way basic-color distribution |
| `luminance_targets`    | per-image 11-bin brightness distribution |
| `color_decode`         | decode any target from an ROI + evaluate (single subject) |
| `compare_rois`         | voxel-matched ROI comparison for one subject + plot |
| `replicate_subjects`   | matched comparison across subjects (any target) — used for the findings |
| `color_shared_subject` | shared-subject model: per-subject projection → shared readout |

## Setup

```bash
pip install -e ".[color]"     # core + decoding/plotting deps
# other extras: ".[clip]"  ".[rawnsd]"  ".[all]"
```

Quick offline check that the install works (no data needed): `python smoke_test.py`.

## ROI structure

```bash
python -m brain2vision.inspect_rois
```

Downloads only `brain_region_masks.hdf5` (~0.8 MB) from `pscotti/mindeyev2` and
prints every region + voxel count per subject. Verified contents: this file
stores, **per subject, a clean partition** of the `nsdgeneral` voxels —

```
V1, V2, V3, V4        individual early/intermediate areas
early_vis             = union of V1..V4
higher_vis            = everything in nsdgeneral beyond early_vis
nsd_general           = early_vis + higher_vis  (the full ROI)
```

It does **not** contain the coarse "streams" atlas or the category-selective
floc regions — for those you need raw NSD (see below).

### The three ROI sets

Defined in `brain2vision/roi.py` (`ROI_SETS`), matched to the exact names above:

| set key      | intent                         | region(s)     | voxels (subj01) |
|--------------|--------------------------------|---------------|-----------------|
| `early_v1v3` | early visual (low-level)       | `V1`,`V2`,`V3`| ~3,970          |
| `v4_color`   | V4 — color / intermediate form | `V4`          | ~687            |
| `concept`    | all higher visual cortex       | `higher_vis`  | ~11,067         |

Load one set, or all three (downloads the subject betas once and reuses it):

```bash
python -m brain2vision.roi --subj 1 --set early_v1v3 --preview
python -m brain2vision.roi --subj 1 --all-sets --preview
```

In code:

```python
from brain2vision.roi import load_roi_set, load_roi_betas, ROI_SETS
early, _ = load_roi_set(subj=1, set_name="early_v1v3")
custom, mask = load_roi_betas(subj=1, rois=["V1", "V4"])
```

Region matching defaults to **exact** (name must equal a fragment). Pass
`match="substring"` (or `--match substring`) only if you want loose matching;
for this dataset's flat `V1..V4` naming, exact is correct and unambiguous.

### `concept` is `higher_vis`, not per-category floc ROIs

`brain_region_masks.hdf5` does not break the concept level into
`floc-faces` / `floc-places` / `floc-bodies` / `floc-words`; it only provides
`higher_vis` (all higher visual cortex beyond V1–V4). To separate faces vs
places vs bodies vs words, use the raw-NSD category masks via the `raw_nsd`
module (`pip install -e ".[rawnsd]"`):

```bash
python -m brain2vision.raw_nsd --subj 1 --atlas floc-faces --regions FFA-1 FFA-2 --sessions 1 2 3 4 5
```

| atlas             | sub-regions |
|-------------------|-------------|
| `prf-visualrois`  | V1v V1d V2v V2d V3v V3d hV4 |
| `floc-faces`      | OFA FFA-1 FFA-2 mTL-faces aTL-faces |
| `floc-places`     | OPA PPA RSC |
| `floc-bodies`     | EBA FBA-1 FBA-2 mTL-bodies |
| `floc-words`      | OWFA VWFA-1 VWFA-2 mfs-words mTL-words |
| `streams`         | early midventral midlateral midparietal ventral lateral parietal |

Use `--regions "*"` for all labeled voxels. Caveats: requires NSD terms,
downloads full volumetric betas (~1 GB/session, streamed one at a time), and the
integer→name label maps should be spot-checked against the `.ctab` files.

## How ROI selection works

Betas are stored per subject as one flattened `nsdgeneral` array
(`betas_all_subjXX_fp32_renorm.hdf5`, shape `(n_trials, n_nsdgeneral_voxels)`).
`brain_region_masks.hdf5` holds boolean masks over that same voxel axis; an ROI
is the `True` columns of its mask. The loader unions multiple ROIs and asserts
mask length == betas voxel count so it can't silently mis-align.

Note on reads: some h5py builds reject index arrays on the row axis, so
`color_decode` reads the betas in plain row-slices and does all fancy selection
in NumPy — robust across environments.

## Download tradeoff

The preprocessed repo ships only the **full nsdgeneral** betas per subject
(~1.2–1.9 GB each). You download that one file and keep only the ROI voxels — so
you avoid downloading all subjects, but the file you fetch is nsdgeneral-sized.
To download *only* an ROI's voxels, use raw NSD (`natural-scenes-dataset` S3 /
OpenNeuro `ds004496`) with the ROI NIfTI masks in
`nsddata/ppdata/subjXX/func1pt8mm/roi/` — arbitrary ROIs, at the cost of
downloading full volumetric betas. Never redistribute NSD betas or COCO images;
ship code + a download step and have users accept the terms.

## Targets

**Color** — an 11-way basic-color distribution per image (fraction of pixels in
red/orange/yellow/green/blue/purple/pink/brown/black/white/gray). Interpretable
and richer than a single dominant color; all 11 pure swatches classify correctly.

```bash
python -m brain2vision.color_targets --images data/coco_images_224_float16.hdf5 --out data/color_targets.npy
```

**Luminance** — the achromatic counterpart: an 11-bin brightness distribution
(dark → bright, standard luma weighting). Used to test whether early visual
cortex decodes brightness better than V4 / higher visual.

```bash
python -m brain2vision.luminance_targets --images data/coco_images_224_float16.hdf5 --out data/luminance_targets.npy
```

**CLIP** — image or caption embeddings, targets for reconstruction-style work
(not used in the color/luminance study). Default ViT-L/14; MindEye2's target is
`--model ViT-bigG-14 --pretrained laion2b_s39b_b160k`. Output is `(n_images,
dim)` + a `_ids.npy` of 0-based NSD ids. `--tokens` is stubbed (returns pooled
embeddings; full patch-token sequences need a `net.visual` hook).

All target files are keyed by 0-based NSD id via a `_ids.npy` sidecar.

## Decoding a target from an ROI

```bash
python -m brain2vision.color_decode --subj 1 --color-targets data/color_targets.npy --model ridge
```

- **Input** — the chosen ROI's voxels per trial (default `v4_color`).
- **Model** — ridge regression whose `alpha` is tuned per fit via `RidgeCV`
  (leave-one-out CV over a grid). This matters for fair ROI comparison; see below.
  `--model mlp` swaps in a small MLP.
- **Split** — test = the shared-1000 images, train = the rest (standard NSD
  held-out, no image leakage).
- **Metrics** — overall and per-target R², plus dominant-bin top-1 accuracy vs
  the 1/11 chance baseline.

`train_eval` takes an optional `labels=` list, so the same decoder works for
color, luminance, or any target.

### Alignment detail to verify once

MindEye2 betas are **not** in image order. `color_decode` recovers each trial's
image id and betas row from the webdataset `behav` arrays, using the MindEye2
default columns (`behav[0,0]` = image id, `behav[0,5]` = betas row), and keeps
only the current-trial record (past/future/old neighbor records are excluded,
and −1 padding is dropped). It prints the id/row ranges and warns if they look
off — glance at that line on the first run. If your `behav` layout differs, set
`IMG_COL` / `BETAS_COL` at the top of `color_decode.py`, or supply your own
alignment via `--ids-npy` / `--betas-npy`.

## Fair ROI comparison — the methodology that matters

Comparing ROIs of very different sizes is a trap. Two confounds must be removed
before any ranking means anything (the [`../REPORT.md`](../REPORT.md) shows how
badly the "answer" changes at each step):

1. **Regularization.** A fixed ridge `alpha` over-penalizes small ROIs and
   under-penalizes large ones; big ROIs overfit and score *negative* test R².
   Fix: `RidgeCV` tunes `alpha` per fit (the chosen alpha scales with ROI size).
2. **Voxel count.** Even properly regularized, a bigger ROI decodes better just
   from having more voxels. Fix: subsample every ROI to the smallest one's size
   (V4's ~687), decode, and average over several random draws.
3. **Single-subject noise.** Fix: replicate across subjects and report mean ± SEM.

`compare_rois` applies steps 1–2 for one subject; `replicate_subjects` adds
step 3.

```bash
# one subject, voxel-matched, 10 draws
python -m brain2vision.compare_rois --subj 1 --color-targets data/color_targets.npy --n-draws 10

# across subjects, matched — color, then luminance (any target via --target/--labels)
python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 --target data/color_targets.npy --out roi_color_8subj.png
python -m brain2vision.replicate_subjects --subjects 1 2 3 4 5 6 7 8 --target data/luminance_targets.npy \
    --labels L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10 --out roi_luminance_8subj.png
```

Each ROI is matched to 687 voxels; `replicate_subjects` reports per-subject and
across-subject (mean ± SEM) R²/top-1 and saves a figure plus a `_summary.npy`.

### Result

The clean, replicated finding (all 8 subjects, matched to 687 voxels):

![color decoding by ROI](../figures/fig1_color_by_roi.png)

Higher visual cortex decodes color best; early visual owns luminance/darkness;
V4 shows no special advantage for *raw pixel* color. Full numbers, luminance
control, and interpretation are in [`../REPORT.md`](../REPORT.md).

## Shared-subject model (alternative pooling)

`color_shared_subject` is a different way to use multiple subjects: each subject
gets its own linear projection from V4 into a common latent, feeding a single
shared readout trained jointly on subjects 1, 2, 5, 7. This genuinely pools
*training data* (the readout learns a subject-invariant latent→color map) rather
than averaging predictions.

```bash
python -m brain2vision.color_shared_subject --color-targets data/color_targets.npy \
    --subjects 1 2 5 7 --latent 512 --epochs 30
```

Downloads each subject's full nsdgeneral betas (~1.5 GB × 4) to extract V4.

## CLIP + bounding boxes (for extending toward reconstruction)

Every NSD image is a center-cropped COCO image, so COCO's instance boxes
transfer over once cropped/rescaled into the 425×425 stimulus frame using
`nsd_stim_info_merged.csv`'s `cropBox`.

```bash
python -m brain2vision.bboxes --out data/nsd_bboxes.json          # all 73k
python -m brain2vision.visualize --images data/coco_images_224_float16.hdf5 \
    --bboxes data/nsd_bboxes.json --nsd-id 3 --out check_3.png
```

Output JSON is keyed by 0-based NSD id: `{"0": [{"category": "dog",
"bbox_xywh": [x,y,w,h]}, ...]}`, coordinates in 425×425 pixels. Verify the
`cropBox` convention (`(top, bottom, left, right)` fractions) by drawing a few
boxes with `visualize`; if they're shifted, flip the order in `bboxes.py`.

Because CLIP `_ids.npy`, the bbox JSON, and each subject's betas are all keyed
to the same 0-based NSD id, you can line up per trial: ROI betas ↔ CLIP target ↔
object boxes — the starting point for object-aware or reconstruction analyses.

## Ideas to extend

Add a VAE-latent target (the low-level counterpart to CLIP, for reconstruction);
decode mean hue/saturation instead of the histogram; restrict to the raw-NSD
`hV4` sub-region for a tighter color ROI; or fit a new subject's projection while
freezing the shared readout.
