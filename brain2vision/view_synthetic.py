"""
view_synthetic.py
=================
Render the NSD-synthetic stimuli to PNGs for eyeballing -- a plain visual sanity
check of what the decoder was actually given.

Produces:
  * a chromatic grid: the 64 chromatic images laid out as 16 hues (rows) x
    4 shared noise patterns (columns) -- the structure verified in the analysis
    (each column is one spatial pattern; hue shifts down the rows);
  * an achromatic sample: a grid sampling the 220 achromatic images.

The chromatic images are isoluminant and modestly saturated, so on screen they
look like faint tinted noise -- useful intuition for why hue is easy to *perceive*
but hard to *linearly decode* from single-trial betas.

Usage
-----
    pip install h5py numpy matplotlib
    python -m brain2vision.view_synthetic \
        --color nsdsynthetic_colorstimuli_subj01.hdf5 \
        --gray  nsdsynthetic_stimuli.hdf5 \
        --huedeg data/synth_huedeg.npy --out-dir figures
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h5py


def load_first_dataset(path):
    """Return the first hdf5 dataset in `path` as a numpy array."""
    with h5py.File(path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        return f[key][()]


def norm_img(im):
    """Coerce an image to (H,W[,3]) float in [0,1], handling channel order."""
    im = np.asarray(im)
    if im.ndim == 3 and im.shape[0] in (1, 3):      # (C,H,W) -> (H,W,C)
        im = im.transpose(1, 2, 0)
    if im.ndim == 3 and im.shape[-1] == 1:
        im = im[..., 0]
    im = im.astype(np.float32)
    if im.max() > 1.001:
        im = im / 255.0
    return np.clip(im, 0, 1)


def chromatic_grid(color_path, out, huedeg_path=None, step=4):
    """16 hues (rows) x 4 patterns (cols) grid of the 64 chromatic images."""
    cs = load_first_dataset(color_path)
    n = cs.shape[0]
    hd = (np.round(np.load(huedeg_path)).astype(int)
          if huedeg_path and os.path.exists(huedeg_path) else None)
    rows = (n + 3) // 4
    fig, ax = plt.subplots(rows, 4, figsize=(8, 1.6 * rows))
    ax = np.atleast_2d(ax)
    for k in range(rows * 4):
        a = ax[k // 4, k % 4]; a.set_xticks([]); a.set_yticks([])
        if k < n:
            a.imshow(norm_img(cs[k])[::step, ::step])
            if k % 4 == 0 and hd is not None:
                a.set_ylabel(f"{hd[k]}°", rotation=0, labelpad=16,
                             va="center", fontsize=8)
            if k // 4 == 0:
                a.set_title(f"pattern {k % 4}", fontsize=8)
        else:
            a.axis("off")
    fig.suptitle("NSD-synthetic chromatic: 16 hues (rows) x 4 patterns (cols)")
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"saved {out}  ({n} chromatic images)")


def achromatic_sample(gray_path, out, n_sample=40, step=4, cols=8):
    """Grid of the achromatic image set (each titled with its image number).
    n_sample<0 or >= N renders ALL images (a labelled contact sheet)."""
    ac = load_first_dataset(gray_path)
    if n_sample < 0 or n_sample >= len(ac):
        idx = np.arange(len(ac))
    else:
        idx = np.linspace(0, len(ac) - 1, n_sample).astype(int)
    rows = (len(idx) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(2 * cols, 1.8 * rows))
    for a, gi in zip(np.atleast_1d(ax).ravel(), idx):
        a.imshow(norm_img(ac[gi])[::step, ::step], cmap="gray")
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"img {gi + 1}", fontsize=7)
    for a in np.atleast_1d(ax).ravel()[len(idx):]:
        a.axis("off")
    fig.suptitle(f"NSD-synthetic achromatic (sample of {len(idx)} of {len(ac)})")
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"saved {out}  (sampled {len(idx)} of {len(ac)})")


def main():
    p = argparse.ArgumentParser(description="Render NSD-synthetic stimuli to PNGs.")
    p.add_argument("--color", help="nsdsynthetic_colorstimuli_subjXX.hdf5")
    p.add_argument("--gray", help="nsdsynthetic_stimuli.hdf5 (achromatic set)")
    p.add_argument("--huedeg", default="data/synth_huedeg.npy",
                   help="per-image hue (deg) from synthetic_targets, for labels")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--step", type=int, default=4, help="pixel downsample factor")
    p.add_argument("--n-sample", type=int, default=40,
                   help="achromatic images to show; -1 or >=N = ALL (contact sheet)")
    p.add_argument("--cols", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.color:
        chromatic_grid(args.color, os.path.join(args.out_dir, "synth_chromatic_grid.png"),
                       huedeg_path=args.huedeg, step=args.step)
    if args.gray:
        name = ("synth_achromatic_all.png" if args.n_sample < 0
                else "synth_achromatic_sample.png")
        achromatic_sample(args.gray, os.path.join(args.out_dir, name),
                          n_sample=args.n_sample, step=args.step, cols=args.cols)
    if not (args.color or args.gray):
        p.error("give --color and/or --gray")


if __name__ == "__main__":
    main()
