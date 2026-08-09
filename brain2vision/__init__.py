"""
brain2vision: NSD brain-to-vision utilities.

ROI-selective fMRI loading, CLIP + color + bounding-box targets, and color
decoding experiments on the Natural Scenes Dataset (NSD).

Modules
-------
roi                  ROI-selective betas from the MindEye2 preprocessed release
raw_nsd              fine-grained ROIs (FFA, PPA, V1v, ...) from raw NSD
nsd_synthetic        NSD-synthetic ROI betas (colour-without-semantics control)
synthetic_targets    hue / colour targets from NSD-synthetic stimulus pixels
synthetic_decode     voxel-matched hue decoding on NSD-synthetic across ROIs
view_synthetic       render NSD-synthetic stimuli to PNGs for eyeballing
reconstruct          retinotopic (pRF) back-projection reconstruction of stimuli
inspect_rois         list available ROI region names per subject
clip_targets         CLIP image/text embedding targets
bboxes               COCO bounding boxes in the NSD stimulus frame
visualize            draw bounding boxes on a stimulus image
color_targets        11-way basic-color distribution per image (HSV rules)
perceptual_color_targets  11-way color via learned color-names (van de Weijer) [scaffold]
luminance_targets    11-bin brightness distribution per image
semantic_targets     COCO object-category presence per image (colour-free) [scaffold]
semantic_residual    residualise colour against semantics (chromatic-vs-semantic) [scaffold]
fg_bg_color_targets  foreground (in-box) vs background colour, to test the attention confound
color_decode         decode a target from an ROI + evaluate (single subject)
compare_rois         voxel-matched ROI comparison + plot (one subject)
replicate_subjects   voxel-matched comparison across subjects (any target)
stats                permutation tests + bootstrap CIs + FDR on ROI differences
reliability          split-half reliability of per-participant region decoding
ncsnr_quality        per-region noise-ceiling SNR vs decoding (signal-quality control)
alignment_check      falsification test of the betas<->image alignment (permutation null)
color_shared_subject shared-subject V4 model (per-subject projection, torch)
"""

__version__ = "0.1.0"

from brain2vision.roi import (
    ROI_SETS,
    load_roi_masks,
    load_roi_betas,
    load_roi_set,
)

__all__ = [
    "ROI_SETS",
    "load_roi_masks",
    "load_roi_betas",
    "load_roi_set",
]
