#!/usr/bin/env bash
#
# rerun_after_split_fix.sh
# ========================
# Re-run every analysis that reads fMRI trials through read_behav_alignment,
# after the fix that (a) reads only the current webdataset shards (train +
# new_test) instead of every shard including the legacy "test" ones, and
# (b) takes held-out status from the documented shared1000 behav column.
#
# WHAT DOES *NOT* NEED RE-RUNNING
#   * the image-derived targets (colour, perceptual colour, luminance, semantic,
#     residual, fg/bg). They are computed from images only and never touch the
#     betas or the trial alignment.
#   * the NSD-synthetic isoluminant control. It uses a separate raw-NSD loader
#     (nsd_synthetic.py) and does not call read_behav_alignment at all.
#
# ALREADY-FINISHED RUNS ARE SKIPPED. A run counts as finished when its
# _summary.npy exists AND contains all 8 participants (replicate_subjects
# checkpoints after every participant, so a partial file is not enough).
# Set FORCE=1 to redo everything regardless.
#
# Usage:
#     bash rerun_after_split_fix.sh              # whatever is still missing
#     bash rerun_after_split_fix.sh main         # headline runs only
#     bash rerun_after_split_fix.sh decoders     # the four alternative decoders
#     bash rerun_after_split_fix.sh controls     # reliability, ncsnr, alignment
#     bash rerun_after_split_fix.sh stats        # statistics only (fast)
#     FORCE=1 bash rerun_after_split_fix.sh main # redo even if complete
#
set -euo pipefail

S=(--subjects 1 2 3 4 5 6 7 8 --r2-weighting variance)
LUM_LABELS="L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10"
WHAT="${1:-all}"
FORCE="${FORCE:-0}"
NSUBJ=8

want () {  # want <step>
  [ "$WHAT" = "all" ] || [ "$WHAT" = "$1" ]
}

complete () {  # complete <summary.npy> -> 0 if it holds all NSUBJ participants
  [ "$FORCE" = "1" ] && return 1
  [ -f "$1" ] || return 1
  python - "$1" "$NSUBJ" <<'PY'
import sys, numpy as np
try:
    d = np.load(sys.argv[1], allow_pickle=True).item()
except Exception:
    sys.exit(1)
n = int(sys.argv[2])
subs = d.get("subjects")
by = d.get("by_subj")
have = len(by) if isinstance(by, dict) else (len(subs) if subs is not None else 0)
sys.exit(0 if have >= n else 1)
PY
}

maybe () {  # maybe <expected_summary.npy> <command...>
  local out="$1"; shift
  if complete "$out"; then
    echo "  [skip] $out already complete"
  else
    echo; echo "### $*"; "$@"
  fi
}

# ---------------------------------------------------------------- 0. sanity
echo "=================================================================="
echo "STEP 0 — confirm the alignment is clean before spending hours on it"
echo "=================================================================="
python - <<'PY'
import sys
from brain2vision.color_decode import read_behav_alignment
bad = False
for s in (3, 4):
    rows, ids, test = read_behav_alignment(s)
    n_dup = len(rows) - len({(int(r), int(i)) for r, i in zip(rows, ids)})
    n_test, n_train = int(test.sum()), int((~test).sum())
    ok = n_dup == 0 and n_train > 0 and n_test > 0
    bad |= not ok
    print(f"  subj{s:02d}: {len(rows):,} trials = {n_train:,} train + {n_test:,} "
          f"held out, {n_dup} duplicates -> {'OK' if ok else 'STILL BROKEN'}")
if bad:
    print("\nStopping: fix the alignment before re-running the analyses.")
    sys.exit(1)
PY

# ---------------------------------------------------------------- 1. main
if want main; then
echo
echo "=================================================================="
echo "STEP 1 — headline decoding runs (ridge, variance-weighted)"
echo "=================================================================="
maybe roi_color_vw_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" \
    --target data/color_targets.npy --out roi_color_vw_v2_8subj.png
maybe roi_luminance_vw_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" \
    --target data/luminance_targets.npy --labels "$LUM_LABELS" \
    --out roi_luminance_vw_v2_8subj.png
maybe roi_colorresid_vw_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" \
    --target data/color_targets_residual.npy --out roi_colorresid_vw_v2_8subj.png
maybe roi_pcolor_vw_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" \
    --target data/color_targets_perceptual.npy --out roi_pcolor_vw_v2_8subj.png
maybe roi_pcolorresid_vw_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" \
    --target data/color_targets_perceptual_residual.npy \
    --out roi_pcolorresid_vw_v2_8subj.png
maybe roi_fgcolor_cleanvw_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" \
    --target data/fgbg_clean_fg.npy --out roi_fgcolor_cleanvw_v2_8subj.png
maybe roi_bgcolor_cleanvw_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" \
    --target data/fgbg_clean_bg.npy --out roi_bgcolor_cleanvw_v2_8subj.png
fi

# ---------------------------------------------------------------- 2. decoders
if want decoders; then
echo
echo "=================================================================="
echo "STEP 2 — decoder robustness (slow; 3 draws for the nonlinear ones)"
echo "=================================================================="
for M in elasticnet svr kernel; do
  maybe "roi_color_${M}_v2_8subj_summary.npy" \
    python -m brain2vision.replicate_subjects "${S[@]}" --target data/color_targets.npy \
      --model "$M" --n-draws 3 --out "roi_color_${M}_v2_8subj.png"
done
maybe roi_color_mlp2_v2_8subj_summary.npy \
  python -m brain2vision.replicate_subjects "${S[@]}" --target data/color_targets.npy \
    --model mlp --n-draws 3 --out roi_color_mlp2_v2_8subj.png
fi

# ---------------------------------------------------------------- 3. controls
if want controls; then
echo
echo "=================================================================="
echo "STEP 3 — controls"
echo "=================================================================="
maybe reliability_v2_8subj_summary.npy \
  python -m brain2vision.reliability "${S[@]}" \
    --target data/color_targets.npy --out reliability_v2_8subj.png
python -m brain2vision.ncsnr_quality --subjects 1 2 3 4 5 6 7 8 \
  --decode-summary roi_color_vw_v2_8subj_summary.npy --out ncsnr_quality_v2.png
python -m brain2vision.alignment_check --subj 1 || \
  echo "  (alignment_check flags/args may differ — run it manually)"
fi

# ---------------------------------------------------------------- 4. stats
if want stats; then
echo
echo "=================================================================="
echo "STEP 4 — statistics (fast; safe to re-run)"
echo "=================================================================="
python -m brain2vision.stats \
  --color roi_color_vw_v2_8subj_summary.npy \
  --luminance roi_luminance_vw_v2_8subj_summary.npy
python -m brain2vision.stats \
  --raw roi_color_vw_v2_8subj_summary.npy \
  --residual roi_colorresid_vw_v2_8subj_summary.npy
python -m brain2vision.stats \
  --raw roi_pcolor_vw_v2_8subj_summary.npy \
  --residual roi_pcolorresid_vw_v2_8subj_summary.npy
python -m brain2vision.stats \
  --fg roi_fgcolor_cleanvw_v2_8subj_summary.npy \
  --bg roi_bgcolor_cleanvw_v2_8subj_summary.npy
fi

echo
echo "=================================================================="
echo "DONE. Compare old vs new:    python compare_runs.py"
echo "=================================================================="
