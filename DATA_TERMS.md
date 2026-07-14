# Data terms

This repository contains **code only**. It does not include, and you must not
commit or redistribute, the underlying data.

## Natural Scenes Dataset (NSD)
NSD is released under its own Terms of Use. You must read and agree to them
before downloading:
- https://naturalscenesdataset.org/
- Citation: Allen et al. (2022), *A massive 7T fMRI dataset to bridge cognitive
  neuroscience and artificial intelligence*, Nature Neuroscience.

The preprocessed betas / masks used here come from the MindEye2 release
(`pscotti/mindeyev2` on Hugging Face), derived from NSD.

## COCO
NSD images are drawn from the COCO dataset. COCO annotations are CC-BY 4.0; the
images themselves are subject to Flickr terms. See https://cocodataset.org/.

## What this means in practice
- Do **not** commit `*.hdf5`, `*.nii.gz`, betas, COCO images, or arrays derived
  directly from them. `.gitignore` is set up to prevent this.
- Ship the download scripts instead; each user accepts the terms and downloads
  their own copy.
- If you publish results, cite NSD, COCO, and MindEye2.
