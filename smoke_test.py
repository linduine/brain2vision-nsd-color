"""
smoke_test.py
=============
Offline end-to-end check of the color pipeline WITHOUT downloading any NSD data.
It verifies that:
  1. the color-target extractor classifies colors correctly, and
  2. the decoder (train_eval) trains and recovers signal from voxels that
     linearly encode color.

If this passes, your install and the color code path work; only the real data
download remains. Run:  python smoke_test.py

Requires the color extras:  pip install -e ".[color]"
"""

import numpy as np
from brain2vision.color_targets import image_color_hist, COLOR_NAMES
from brain2vision.color_decode import train_eval


def main():
    rng = np.random.default_rng(0)

    # --- 1) color-target sanity: 11 solid swatches classify as themselves ----
    palette = {
        "red": (220, 30, 30), "green": (30, 180, 60), "blue": (40, 60, 220),
        "yellow": (240, 220, 40), "orange": (240, 150, 30),
        "purple": (150, 40, 200), "pink": (240, 180, 200),
        "brown": (130, 80, 30), "black": (15, 15, 15),
        "white": (240, 240, 240), "gray": (128, 128, 128),
    }
    ok = 0
    for name, rgb in palette.items():
        v = image_color_hist(np.array(rgb, np.float32).reshape(1, 1, 3) / 255.0)
        ok += COLOR_NAMES[int(v.argmax())] == name
    print(f"[1] color classifier: {ok}/{len(palette)} swatches correct")
    assert ok == len(palette), "color classifier regressed"

    # --- 2) decoder recovers signal from synthetic color-encoding voxels ------
    N, nvox = 1500, 120
    pal = np.array(list(palette.values()), np.float32)
    imgs = pal[rng.integers(0, len(pal), N)]
    imgs = (imgs + rng.normal(0, 12, imgs.shape)).clip(0, 255) / 255.0
    y = np.stack([image_color_hist(im.reshape(1, 1, 3)) for im in imgs])

    W = rng.normal(0, 1, (y.shape[1], nvox))       # color -> voxels
    X = (y @ W + rng.normal(0, 0.3, (N, nvox))).astype(np.float32)
    is_test = rng.random(N) < 0.2

    res = train_eval(X, y.astype(np.float32), is_test, model="ridge", alpha=100.0)
    print(f"[2] decoder: overall R^2={res['overall_r2']:.3f} "
          f"top1={res['top1']:.3f} (chance {1/len(COLOR_NAMES):.3f})")
    assert res["overall_r2"] > 0.5 and res["top1"] > 0.5, "decoder failed to learn"

    print("\nSMOKE TEST PASSED — install + color pipeline work.")


if __name__ == "__main__":
    main()
