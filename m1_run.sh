#!/bin/bash
# =============================================================================
# m1_run.sh -- MODEL 1 (KAT-UNet incremental upgrade), BCSS 5-class
#
#   nohup bash m1_run.sh > logs/m1_run.out 2>&1 &        (your Quadro machine)
#   sbatch m1_run.sh                                      (SLURM clusters)
#
#   Run a subset of stages:   STAGES="2" bash m1_run.sh   (only the step-0 TTA test)
# =============================================================================
#SBATCH --job-name=kat_m1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=logs/m1_%j.out
#SBATCH --error=logs/m1_%j.err

# ------------------------------- EDIT THESE ---------------------------------
ROOT=${ROOT:-/DATA/VIJAY/dekaht_Unet}
DATA=${DATA:-$ROOT/data/BCSS5_patches}
SWIN=${SWIN:-$ROOT/swin_tiny_patch4_window7_224.pth}
LEGACY=${LEGACY:-$ROOT/out_bcss5_run1/best_model.pth}   # your existing checkpoint
OUT=${OUT:-$ROOT/out_m1}
STAGES=${STAGES:-"0 1 2 3 4"}
# ------------------------- TRAINING BUDGET (one place) -----------------------
# Keep these IDENTICAL for Model 1 and Model 2 so the M1-vs-M2 comparison is fair.
# Defaults = effective batch 24, 30,000 optimizer steps, 720k samples (6.9 passes):
# the same sample budget as the previous A100 run, whose val mDice peaked after only
# 2.1 passes. Every step-denominated knob is rescaled so its SAMPLE budget is unchanged.
BATCH=${BATCH:-24}          # micro-batch (fits the 16 GB Quadro in fp16)
ACCUM=${ACCUM:-1}           # effective batch = BATCH*ACCUM (no BatchNorm -> exact)
LR=${LR:-3.7e-4}            # 3e-4 @ Beff 16 * sqrt(24/16)  (AdamW square-root rule)
ITERS=${ITERS:-1500}        # micro-batches per "epoch" = validation cadence
EPOCHS=${EPOCHS:-20}        # 20 x 1500 x 24 = 720k samples
WARMUP=${WARMUP:-1000}      # 1000 x 24 = 24k samples (was 1500 x 16 = 24k)
EMA=${EMA:-0.9997}          # horizon 1/(1-d) = 3333 steps x 24 = 80k samples (unchanged)
SWA_LAST=${SWA_LAST:-4}     # last 20% of epochs (was 6 of 30)
NPROC=$(nproc 2>/dev/null || echo 8)
WORKERS=${WORKERS:-$(( NPROC > 14 ? 12 : (NPROC > 3 ? NPROC - 2 : 2) ))}
TTA_CANDS="1.0;1.0,1.25;1.0,1.25,1.5;1.0,1.25,1.5,1.75"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

set -u
cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
mkdir -p logs "$OUT"

# ---- environment: venv/conda prefix inside the project, else whatever is active
if [ -f "$ROOT/mtp_py.310/bin/activate" ]; then
    source "$ROOT/mtp_py.310/bin/activate"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda activate "$ROOT/mtp_py.310" 2>/dev/null || conda activate mtp_py.310 2>/dev/null || true
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1

echo "================ ENVIRONMENT ================"
echo "budget: batch $BATCH x accum $ACCUM | lr $LR | $EPOCHS epochs x $ITERS it | warmup $WARMUP | EMA $EMA | SWA last $SWA_LAST | workers $WORKERS"
python - <<'PY'
import sys, torch
try:
    import albumentations as A; av = A.__version__
except Exception as e:
    av = f"MISSING ({e})"
print("python", sys.version.split()[0], "| torch", torch.__version__, "| albumentations", av)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name} | {p.total_memory / 1e9:.1f} GB | compute capability {p.major}.{p.minor}")
else:
    print("NO CUDA DEVICE")
PY
has() { [[ " $STAGES " == *" $1 "* ]]; }
set -e

# =============================================================================
# STAGE 0 -- data hygiene (from diagnostics_report.json)
# =============================================================================
if has 0; then
  echo ""; echo ">>> [0] data layout checks"
  # The diagnostic found 1 image sitting directly in train/ instead of train/images/.
  shopt -s nullglob
  for f in "$DATA"/train/*.png; do
      b=$(basename "$f")
      if [ ! -e "$DATA/train/images/$b" ]; then
          mv "$f" "$DATA/train/images/$b" && echo "  moved stray $b -> train/images/"
      else
          echo "  stray duplicate $b already in train/images/ (left in place)"
      fi
  done
  shopt -u nullglob
  for s in train val test; do
      if [ -d "$DATA/$s/images" ]; then
          echo "  $s: $(ls "$DATA/$s/images" | wc -l) images, $(ls "$DATA/$s/masks" | wc -l) masks"
      else
          echo "  $s: MISSING"
      fi
  done
  if [ ! -d "$DATA/test/images" ]; then
      echo "  !!! No test split. Copy BCSS5_patches/test/ (institutions OL LL E2 EW GM S3,"
      echo "      42,157 patches on the previous cluster) here -- without it there is no"
      echo "      benchmark number. Training and validation still run."
  fi
  if [ ! -f "$SWIN" ]; then
      echo "  Swin-T checkpoint missing -> downloading"
      wget -q -O "$SWIN" https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth \
        || { echo "  download failed: copy swin_tiny_patch4_window7_224.pth to $SWIN"; exit 1; }
  fi
fi

# =============================================================================
# STAGE 1 -- self-test (CPU, ~5 min). 67 checks. Nothing trains if this fails.
# =============================================================================
if has 1; then
  echo ""; echo ">>> [1] self-test"
  python m1_train.py --self-test
fi

# =============================================================================
# STAGE 2 -- STEP 0 of the plan: zero-training TTA gain on your EXISTING checkpoint.
# The legacy bcss5 checkpoint loads into Model 1 code bit-exactly (verified).
# Candidates are in-distribution scales (>= 1.0): the context test showed this
# model LOSES Dice at every scale below 1.0 (it was trained with zoom-in only).
# =============================================================================
if has 2; then
  echo ""; echo ">>> [2] step 0: TTA selection on the existing checkpoint"
  if [ -f "$LEGACY" ]; then
    python m1_train.py --data-root "$DATA" --output-dir "$OUT/step0" \
      --eval-only "$LEGACY" --eval-weights ema --eval-on val \
      --tta-select --tta-candidates "$TTA_CANDS" \
      --workers "$WORKERS" --batch-size 8
  else
    echo "  no legacy checkpoint at $LEGACY -- skipped"
  fi
fi

# =============================================================================
# STAGE 3 -- train Model 1
#   A1 --skip-norm --skip-bridge kaca        A2 --uper-levels 4 --fpn-smooth
#   A4 boundary CE / boundary head / Tversky off (defaults)
#   fp16 is chosen automatically on this Turing GPU (cc 7.5).
#   Per-epoch validation runs WITHOUT TTA (4x faster); TTA is selected in stage 4.
#   If batch 8 does not fit in 16 GB, add --grad-ckpt (identical results, verified).
# =============================================================================
if has 3; then
  echo ""; echo ">>> [3] training Model 1"
  python m1_train.py --data-root "$DATA" --output-dir "$OUT" \
    --encoder-pretrained "$SWIN" --min-pretrained-coverage 0.90 \
    --model-variant swkat-tiny --num-classes 5 --img-size 224 --eval-size 224 \
    --epochs "$EPOCHS" --iters-per-epoch "$ITERS" --batch-size "$BATCH" --accum-steps "$ACCUM" \
    --workers "$WORKERS" --lr "$LR" --encoder-lr-scale 0.4 --layer-decay 0.90 \
    --weight-decay 0.05 --warmup-iters "$WARMUP" --min-lr-ratio 0.01 \
    --grad-clip-mode calibrated --grad-clip-ceiling 25.0 --clip-calib-start -1 \
    --ema-decay "$EMA" --swa-last "$SWA_LAST" --select-smooth 3 \
    --skip-norm --skip-bridge kaca --uper-levels 4 \
    --ce-weight 1.0 --dice-weight 1.0 --lovasz-weight 0.5 --lovasz-start-frac 0.4 \
    --tversky-weight 0.0 --boundary-weight 0.0 --boundary-head-weight 0.0 \
    --label-smoothing 0.05 --class-weights auto \
    --aux-budget 0.30 --aux-decay 0.6 --ds-loss-size 56 \
    --aug-mode rrc --rrc-min-scale 0.35 --hed-sigma 0.05 --clahe-p 0.20 --elastic-p 0.05 \
    --amp-dtype auto --no-tta --eval-interval 1 --log-interval 200 --seed 1234 --skip-test
fi

# =============================================================================
# STAGE 4 -- pick TTA on val, then score val and (if present) the test split once
# =============================================================================
if has 4; then
  echo ""; echo ">>> [4] final evaluation"
  BEST="$OUT/best_model.pth"
  python m1_train.py --data-root "$DATA" --output-dir "$OUT" \
    --eval-only "$BEST" --eval-weights ema --eval-on val \
    --tta-select --tta-candidates "$TTA_CANDS" \
    --workers "$WORKERS" --batch-size 8
  if [ -d "$DATA/test/images" ]; then
    python m1_train.py --data-root "$DATA" --output-dir "$OUT" \
      --eval-only "$BEST" --eval-weights ema --eval-on test \
      --tta-config "$OUT/tta_selection.json" --workers "$WORKERS" --batch-size 8
  else
    echo "  test split absent -- copy it and re-run with STAGES=\"4\""
  fi
fi
echo ""; echo "DONE $(date)"
