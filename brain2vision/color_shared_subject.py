"""
train_v4_color_multisubject.py
=============================
Shared-subject V4 -> color decoder. V4 is small per subject (~600-800 voxels),
so we pool subjects: each subject gets its OWN linear projection from its V4
voxels into a common latent space, and a SINGLE shared readout predicts the
11-way color distribution. Trained jointly on all subjects' trials -- this is
the MindEye2 "shared-subject" idea, and it genuinely adds training data rather
than just averaging predictions.

    subj s:  V4 betas (n_vox_s)  --Linear_s-->  latent (k)  --\
                                                               shared MLP -> 11 colors
    subj t:  V4 betas (n_vox_t)  --Linear_t-->  latent (k)  --/

Only the per-subject Linear_s differs; the readout is shared, so the readout
must learn a subject-invariant mapping from the common latent to colors.

Subjects: 1, 2, 5, 7 by default (the four with all 40 sessions).
Split: per subject, test = shared-1000 images, train = the rest.

Reuses build_xy() from train_v4_color.py for per-subject aligned (X, y, is_test).

Usage
-----
    pip install torch scikit-learn h5py numpy huggingface_hub
    python train_v4_color_multisubject.py --color-targets color_targets.npy \
        --subjects 1 2 5 7 --latent 512 --epochs 30
"""

import argparse
import numpy as np

from brain2vision.color_decode import build_xy
from brain2vision.color_targets import COLOR_NAMES


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def load_all_subjects(subjects, color_targets):
    """Return dict subj -> (Xtr, ytr, Xte, yte) with per-subject z-scoring."""
    data = {}
    for s in subjects:
        print(f"\n---- loading subject {s} ----")
        X, y, is_test = build_xy(s, color_targets, rois=("V4",))
        Xtr, Xte = _standardize(X[~is_test], X[is_test])
        data[s] = (Xtr.astype(np.float32), y[~is_test].astype(np.float32),
                   Xte.astype(np.float32), y[is_test].astype(np.float32))
        print(f"  subj{s}: train {Xtr.shape}  test {Xte.shape}")
    return data


def build_model(vox_dims, latent, hidden, n_out, torch):
    """vox_dims: dict subj -> n_voxels. Returns an nn.Module."""
    import torch.nn as nn

    class SharedSubject(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleDict(
                {str(s): nn.Linear(d, latent) for s, d in vox_dims.items()})
            self.readout = nn.Sequential(
                nn.LayerNorm(latent), nn.GELU(),
                nn.Linear(latent, hidden), nn.GELU(),
                nn.Linear(hidden, n_out))

        def forward(self, x, subj):
            return self.readout(self.proj[str(subj)](x))

    return SharedSubject()


def r2_np(y, p):
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0) + 1e-9
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color-targets", required=True)
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 5, 7])
    ap.add_argument("--latent", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--out", default="v4_color_shared.pt")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_all_subjects(args.subjects, args.color_targets)
    vox_dims = {s: data[s][0].shape[1] for s in args.subjects}

    model = build_model(vox_dims, args.latent, args.hidden,
                        len(COLOR_NAMES), torch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = torch.nn.MSELoss()

    # pre-move tensors
    tens = {s: (torch.tensor(data[s][0]).to(dev), torch.tensor(data[s][1]).to(dev))
            for s in args.subjects}

    for epoch in range(args.epochs):
        model.train()
        # interleave subject minibatches so the shared readout sees all subjects
        batches = []
        for s in args.subjects:
            n = tens[s][0].shape[0]
            perm = torch.randperm(n)
            for i in range(0, n, args.batch):
                batches.append((s, perm[i:i + args.batch]))
        np.random.shuffle(batches)

        total = 0.0
        for s, idx in batches:
            xb, yb = tens[s][0][idx], tens[s][1][idx]
            opt.zero_grad()
            loss = lossf(model(xb, s), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        denom = sum(tens[s][0].shape[0] for s in args.subjects)
        print(f"epoch {epoch+1:02d}/{args.epochs}  train MSE {total/denom:.5f}")

    # ---- evaluate per subject on their shared-1000 test set ----
    model.eval()
    print("\n=== shared-subject V4 -> color: per-subject test ===")
    per_color = []
    for s in args.subjects:
        Xte, yte = torch.tensor(data[s][2]).to(dev), data[s][3]
        with torch.no_grad():
            pred = model(Xte, s).cpu().numpy()
        r2 = r2_np(yte, pred)
        overall = r2_np(yte, pred).mean()
        top1 = (pred.argmax(1) == yte.argmax(1)).mean()
        per_color.append(r2)
        print(f"subj{s}: mean R²={r2.mean():+.3f}  top1={top1:.3f} "
              f"(chance {1/len(COLOR_NAMES):.3f})")

    mean_r2 = np.mean(per_color, 0)
    print("\naveraged per-color R² across subjects:")
    for name, val in zip(COLOR_NAMES, mean_r2):
        print(f"   {name:8s} {val:+.3f}")

    torch.save({"state_dict": model.state_dict(),
                "vox_dims": vox_dims, "subjects": args.subjects,
                "latent": args.latent, "hidden": args.hidden}, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
