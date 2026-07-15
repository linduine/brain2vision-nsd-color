"""
clip_targets.py
==========================
Compute CLIP embeddings for the NSD images -- the targets your fMRI decoder
predicts (MindEye-style fMRI -> CLIP).

Two embedding types, either or both:
  * IMAGE embeddings  (default): run a CLIP image encoder on the stimulus images.
  * TEXT  embeddings  (--captions): encode the COCO captions of each image.

Model choice
------------
Uses open_clip, so any supported model works via --model / --pretrained:
  * default  ViT-L-14   / openai   -> 768-dim, light, good for first tests
  * MindEye2 ViT-bigG-14 / laion2b_s39b_b160k -> 1664-dim, SOTA target, heavy
Pass --tokens to keep the full patch-token sequence (needed for MindEye-style
priors); default returns the pooled global embedding.

Image source
------------
Reads images from an HDF5 array indexed by NSD id (0-based, 73k order):
  * MindEye2:  coco_images_224_float16.hdf5   (from pscotti/mindeyev2)
  * or NSD:    nsd_stimuli.hdf5                (425x425; will be resized)
The first dataset in the file is used; images are assumed (N, 3, H, W) or
(N, H, W, 3) in 0..1 or 0..255 and are normalized for CLIP automatically.

Usage
-----
    pip install open_clip_torch torch h5py numpy pillow
    # image embeddings, default model, subset for a smoke test
    python -m brain2vision.clip_targets --images coco_images_224_float16.hdf5 \
        --nsd-ids 0 1 2 3 --out clip_vitL_img.npy
    # MindEye2 target
    python -m brain2vision.clip_targets --images coco_images_224_float16.hdf5 \
        --model ViT-bigG-14 --pretrained laion2b_s39b_b160k --tokens \
        --out clip_bigG_tokens.npy
    # caption/text embeddings (needs COCO captions json, see build_coco_bboxes)
    python -m brain2vision.clip_targets --captions captions_by_nsdid.json \
        --out clip_vitL_text.npy
"""

import os
import json
import argparse
import numpy as np


def _load_model(model, pretrained, device):
    import torch
    import open_clip
    net, _, preprocess = open_clip.create_model_and_transforms(
        model, pretrained=pretrained)
    net = net.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model)
    return net, preprocess, tokenizer, torch


def _iter_image_batches(h5path, nsd_ids, batch):
    import h5py
    with h5py.File(h5path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        dset = f[key]
        n = dset.shape[0]
        ids = list(range(n)) if nsd_ids is None else list(nsd_ids)
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            arr = dset[sorted(chunk)]  # h5py needs increasing index
            yield np.asarray(chunk), np.asarray(arr)


def _to_pil_list(arr):
    """Normalize an image batch array to a list of PIL RGB images."""
    from PIL import Image
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[1] == 3:      # (N,3,H,W) -> (N,H,W,3)
        a = a.transpose(0, 2, 3, 1)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.001:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return [Image.fromarray(x).convert("RGB") for x in a]


def embed_images(h5path, nsd_ids, model, pretrained, out,
                 tokens=False, batch=64, device=None):
    if tokens:
        print("WARNING: --tokens not yet implemented; returning POOLED "
              "embeddings. See note in embed_images().")
    net, preprocess, _, torch = _load_model(
        model, pretrained, device or ("cuda" if _cuda() else "cpu"))
    dev = next(net.parameters()).device
    all_ids, all_vecs = [], []
    for ids, arr in _iter_image_batches(h5path, nsd_ids, batch):
        pil = _to_pil_list(arr)
        x = torch.stack([preprocess(im) for im in pil]).to(dev)
        with torch.no_grad():
            feats = net.encode_image(x)
            # NOTE: this returns the POOLED global embedding. MindEye-style
            # diffusion priors want the full patch-token sequence, which
            # requires hooking net.visual (e.g. open_clip's
            # forward_intermediates / a transformer forward hook). Left as a
            # follow-up so the pooled path stays simple and correct.
        all_ids.append(ids)
        all_vecs.append(feats.float().cpu().numpy())
        print(f"  embedded {sum(len(v) for v in all_vecs)} images")
    ids = np.concatenate(all_ids)
    vecs = np.concatenate(all_vecs, 0)
    order = np.argsort(ids)
    np.save(out, vecs[order])
    np.save(out.replace(".npy", "_ids.npy"), ids[order])
    print(f"Saved {out}  shape={vecs.shape} (rows aligned to sorted NSD ids)")


def embed_captions(caption_json, model, pretrained, out, batch=256, device=None):
    """caption_json: {nsd_id: "a caption string"} (one caption per image)."""
    net, _, tokenizer, torch = _load_model(
        model, pretrained, device or ("cuda" if _cuda() else "cpu"))
    dev = next(net.parameters()).device
    with open(caption_json) as f:
        caps = json.load(f)
    ids = sorted(int(k) for k in caps.keys())
    vecs = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        toks = tokenizer([caps[str(k)] for k in chunk]).to(dev)
        with torch.no_grad():
            feats = net.encode_text(toks)
        vecs.append(feats.float().cpu().numpy())
        print(f"  embedded {sum(len(v) for v in vecs)} captions")
    vecs = np.concatenate(vecs, 0)
    np.save(out, vecs)
    np.save(out.replace(".npy", "_ids.npy"), np.asarray(ids))
    print(f"Saved {out}  shape={vecs.shape}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", help="HDF5 of NSD images (image-embedding mode)")
    p.add_argument("--captions", help="JSON {nsd_id: caption} (text mode)")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--tokens", action="store_true",
                   help="Keep full patch-token sequence (image mode)")
    p.add_argument("--nsd-ids", nargs="+", type=int)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not args.images and not args.captions:
        p.error("Provide --images (image embeddings) and/or --captions (text).")
    if args.images:
        embed_images(args.images, args.nsd_ids, args.model, args.pretrained,
                     args.out, tokens=args.tokens, batch=args.batch)
    if args.captions:
        out = args.out if not args.images else args.out.replace(".npy", "_text.npy")
        embed_captions(args.captions, args.model, args.pretrained, out,
                       batch=max(args.batch, 128))


if __name__ == "__main__":
    main()
