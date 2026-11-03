#!/bin/bash
#SBATCH --job-name=dekaht_bcss_UC5
#SBATCH --account=iitp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=submit_bcss_UC5_%j.out
#SBATCH --error=submit_bcss_UC5_%j.err

set +euo pipefail
mkdir -p logs

echo "=========================================="
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-na}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-na}"
echo "SLURM_CPUS_ON_NODE = ${SLURM_CPUS_ON_NODE:-na}"
echo "=========================================="

module load MLDL/miniconda3
export QT_XCB_GL_INTEGRATION=""
export CONDA_BACKUP_QT_XCB_GL_INTEGRATION=""
eval "$(conda shell.bash hook)"
conda activate mtp_py.310

set -euo pipefail
module load cuda 2>/dev/null || true

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA=./data/BCSS_formatted
CKPT=./swin_tiny_patch4_window7_224.pth
OUT=./out_bcss_UC5_run1

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

# ============================================================================
# STEP 1 — self-test (123 assertions, CPU, ~8 min)
# ============================================================================
echo ">>> [1/3] self-test"
python train_dekaht_unet_UC5.py --self-test

# ============================================================================
# STEP 2 — training
# ============================================================================
# WHAT CHANGED vs UC4 AND WHY (read the run log to confirm each one fires):
#
#   --grad-clip auto        UC4 used a FIXED clip of 1.0 while the mean grad norm
#                           was 10-22. ~100% of steps were scaled down ~20x, so the
#                           effective head LR was 1.7e-05 (not 3e-04) and the
#                           effective encoder LR was 9e-07 -- the encoder was
#                           frozen in all but name. Watch the "clip ...@N%" field:
#                           it must settle well under 20%.
#   --encoder-lr-scale 0.4  ImageNet -> H&E is a large domain shift; 0.1 was too low
#                           even before the clipping throttle.
#   --drop-path-rate 0.1    UC4 train fg-Dice 0.702 vs val 0.700 == ZERO
#   --head-dropout 0.05     generalisation gap. The model was UNDERfitting, so
#                           regularisation comes DOWN, not up.
#   --epochs 80             UC4 gained +0.0003 mDice over its last 99 epochs.
#   --lovasz-start-frac 0.4 Lovasz is a fine-tuning loss; running it from step 0
#                           fights CE and inflates the gradient norm.
#   --ohem-thresh 0.7       mDice(img) 0.717 vs mDice(agg) 0.825 says small regions
#                           are being missed. OHEM puts the gradient there.
#   --boundary-head-weight  explicit boundary stream (Gated-SCNN family).
#   --swa-last 8            average the tail; typically +0.3-0.8 mDice for free.
echo ">>> [2/3] training"
python train_dekaht_unet_UC5.py \
  --data-root "$DATA" \
  --img-subdir images --mask-subdir masks \
  --output-dir "$OUT" \
  --num-classes 3 \
  --img-size 224 --eval-size 224 --patch-size 4 --window-size 7 \
  --model-variant swkat-tiny \
  --decoder-depths 2,2,2 --bottleneck-depth 2 \
  --ffn-type kan --gate-mode residual_tanh --groups 8 \
  --drop-path-rate 0.1 --head-dropout 0.05 \
  --encoder-pretrained "$CKPT" --min-pretrained-coverage 0.90 \
  --epochs 80 \
  --batch-size 16 --accum-steps 1 --workers 8 \
  --opt adamw --lr 3e-4 --encoder-lr-scale 0.4 --layer-decay 0.90 \
  --weight-decay 0.05 \
  --sched cosine --warmup-iters 1500 --min-lr-ratio 0.01 \
  --grad-clip auto --clip-factor 3.0 --clip-warmup 200 \
  --ema-decay 0.9998 \
  --ce-weight 1.0 --dice-weight 1.0 --lovasz-weight 0.5 --lovasz-start-frac 0.4 \
  --label-smoothing 0.05 --boundary-weight 1.0 --focal-gamma 0.0 \
  --ohem-thresh 0.7 --ohem-min-frac 0.25 \
  --boundary-head-weight 0.5 \
  --swa-last 8 \
  --scale-min 0.6 --scale-max 1.6 --hed-sigma 0.05 \
  --amp-dtype bf16 \
  --tta --tta-scales 1.0 \
  --metric-ignore-bg \
  --eval-interval 1 --log-interval 100 \
  --resume "$OUT/last_model.pth" \
  --seed 1234

# ============================================================================
# STEP 3 — final score with multi-scale TTA
# ============================================================================
echo ">>> [3/3] final evaluation"
python train_dekaht_unet_UC5.py \
  --data-root "$DATA" --output-dir "$OUT" --num-classes 3 \
  --img-size 224 --eval-size 224 --workers 8 --batch-size 16 \
  --eval-only "$OUT/best_model.pth" --eval-weights ema \
  --tta --tta-scales 0.75,1.0,1.25 \
  --amp-dtype bf16 --metric-ignore-bg

echo "done."

# ============================================================================
# FOLLOW-UP RUNS (submit separately, in this order of expected payoff)
# ============================================================================
#
# B) CAPACITY. UC4 was underfitting, so a bigger encoder is the most direct win.
#    wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
#    ... --model-variant swkat-small --encoder-pretrained ./swin_small_patch4_window7_224.pth \
#        --batch-size 12 --output-dir ./out_bcss_UC5_small
#    (the remapper is variant-agnostic; it will report 100% coverage for Swin-S too)
#
# C) CLASS 0. Read the [class0-audit] line. If it says label 0 hugs the borders,
#    it is BCSS 'outside_roi' and you are wasting a third of the softmax on padding:
#    ... --bg-as-ignore --num-classes 2 --metric-include-bg
#
# D) HONEST SPLIT. The log will still warn that 151 slide ids straddle train/ and
#    val/. Your 0.825 is optimistically biased. Delete val/ and let the script build
#    a grouped split, then re-run:  rm -rf $DATA/val   (it will use --slide-id-regex)
#    Report BOTH numbers in the paper: the leaky one is not defensible alone.
#
# E) RESOLUTION. Only if [size-audit] says your patches are larger than 256:
#    ... --img-size 320 --eval-size 320 --batch-size 8 --accum-steps 2
#
# ABLATIONS (one flag each):
#   --grad-clip 1.0        reproduce the UC4 throttle (expect ~0.825)
#   --no-fpn               remove multi-level fusion
#   --no-boundary --boundary-head-weight 0   remove the boundary stream
#   --ohem-thresh 0        remove hard-pixel mining
#   --lovasz-start-frac 0  Lovasz from step 0 (UC4 behaviour)
#   --swa-last 0           no weight averaging
#   --ffn-type mlp         the KAN ablation -- the most important one for the paper
