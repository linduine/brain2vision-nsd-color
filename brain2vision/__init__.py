"""
brain2vision: NSD brain-to-vision utilities.

ROI-selective fMRI loading, CLIP + color + bounding-box targets, and color
decoding experiments on the Natural Scenes Dataset (NSD).

Modules
-------
roi                  ROI-selective betas from the MindEye2 preprocessed release
raw_nsd              fine-grained ROIs (FFA, PPA, V1v, ...) from raw NSD
inspect_rois         list available ROI region names per subject
clip_targets         CLIP image/text embedding targets
bboxes               COCO bounding boxes in the NSD stimulus frame
visualize            draw bounding boxes on a stimulus image
color_targets        11-way basic-color distribution per image
color_decode         single-subject V4 -> color decoder
compare_rois         color decoding across ROI sets + plot
color_shared_subject shared-subject V4 -> color decoder (torch)
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
