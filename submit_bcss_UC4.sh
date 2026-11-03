#!/bin/bash
#SBATCH --job-name=dekaht_bcss_UC4
#SBATCH --account=iitp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8        # was 8; 2 workers could not feed an A100
#SBATCH --time=48:00:00
#SBATCH --output=submit_bcss_UC4%j.out
#SBATCH --error=submit_bcss_UC4_%j.err

set +euo pipefail
mkdir -p logs

echo "=========================================="
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-na}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-na}"
echo "SLURM_CPUS_ON_NODE = ${SLURM_CPUS_ON_NODE:-na}"
echo "=========================================="

# ---- environment -----------------------------------------------------------
module load MLDL/miniconda3

export QT_XCB_GL_INTEGRATION=""
export CONDA_BACKUP_QT_XCB_GL_INTEGRATION=""
eval "$(conda shell.bash hook)"
source activate mtp_py.310

set -euo pipefail
module load cuda 2>/dev/null || true

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# deterministic-ish cuBLAS without the perf hit of full determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA=./data/BCSS_formatted
CKPT=./swin_tiny_patch4_window7_224.pth
OUT=./out_bcss_UC4_run8

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

# ---- STEP 1: code self-test (~3 min, CPU) ----------------------------------
# 90 assertions covering every fix. If this fails, nothing else runs.
echo ">>> [1/3] self-test"
python train_dekaht_unet_UC4.py --self-test

# ---- STEP 2: training ------------------------------------------------------
# The script aborts before epoch 0 if:
#   * the pretrained encoder covers < 90% of transferable parameters
#   * any augmented image comes back constant
#   * labels fall outside [0, num_classes-1] and are not ignore_index
echo ">>> [2/3] training"
python train_dekaht_unet_UC4.py \
  --data-root "$DATA" \
  --img-subdir images --mask-subdir masks \
  --output-dir "$OUT" \
  --num-classes 3 \
  --img-size 224 --eval-size 224 --patch-size 4 --window-size 7 \
  --model-variant swkat-tiny \
  --decoder-depths 2,2,2 --bottleneck-depth 2 \
  --ffn-type kan --gate-mode residual_tanh --groups 8 \
  --drop-path-rate 0.2 --head-dropout 0.1 \
  --encoder-pretrained "$CKPT" \
  --min-pretrained-coverage 0.90 \
  --epochs 120 \
  --batch-size 16 --accum-steps 1 \
  --workers 8 \
  --opt adamw --lr 3e-4 --encoder-lr-scale 0.1 --layer-decay 0.85 \
  --weight-decay 0.05 \
  --sched cosine --warmup-iters 1500 --min-lr-ratio 0.01 \
  --grad-clip 1.0 \
  --ema-decay 0.9998 \
  --ce-weight 1.0 --dice-weight 1.0 --lovasz-weight 0.5 \
  --label-smoothing 0.05 --boundary-weight 1.0 --focal-gamma 0.0 \
  --scale-min 0.6 --scale-max 1.6 --hed-sigma 0.05 \
  --amp-dtype bf16 \
  --tta --tta-scales 1.0 \
  --metric-ignore-bg \
  --eval-interval 1 --log-interval 100 \
  --seed 1234

# ---- STEP 3: final score with full multi-scale TTA --------------------------
echo ">>> [3/3] final evaluation (multi-scale TTA)"
python train_dekaht_unet_UC4.py \
  --data-root "$DATA" --output-dir "$OUT" --num-classes 3 \
  --img-size 224 --eval-size 224 --workers 8 --batch-size 16 \
  --eval-only "$OUT/best_model.pth" --eval-weights ema \
  --tta --tta-scales 0.75,1.0,1.25 \
  --amp-dtype bf16 --metric-ignore-bg

echo "done."

# =============================================================================
# NOTES / KNOBS
# =============================================================================
#
# 1) IS CLASS 0 A REAL CLASS?
#    In the released BCSS label map, 0 = "outside_roi" (a don't-care region), and
#    the metric you have been reporting already excludes it. If that is what your
#    BCSS_formatted class 0 is, you are currently spending model capacity, loss
#    gradient, and softmax probability mass on a padding label. Add:
#        --bg-as-ignore --num-classes 2 --metric-include-bg
#    (--bg-as-ignore maps 0 -> ignore and shifts 1,2 -> 0,1, so after the shift
#     both remaining classes are real and should be scored.)
#    Read the [label-audit] line the script prints before deciding. If class 0 is
#    a genuine "other tissue" class, leave the flags as they are above.
#
# 2) IF THE LABEL AUDIT REPORTS VALUES >= 3
#    Your formatting step did not actually collapse BCSS's 21 raw classes. Supply
#    an explicit mapping instead of letting them all become "ignore", e.g.
#        --label-map '{"0":255,"1":0,"2":1,"3":2,"4":2,"5":2}'
#    (keys are RAW label values, values are training labels or 255 for ignore).
#
# 3) SLIDE-LEVEL SPLIT
#    If the [split] line warns that slide ids appear in both train/ and val/,
#    your validation Dice is inflated. Re-partition BCSS by TCGA slide id, or
#    delete val/ and let the script build a grouped split itself (it will use
#    --slide-id-regex, which defaults to the TCGA-XX-YYYY pattern).
#
# 4) MEMORY
#    batch 24 @ 224 with bf16 fits comfortably on a 40 GB A100. If you hit OOM,
#    use --batch-size 16 --accum-steps 2 (same effective batch, no LR change).
#
# 5) ABLATIONS (for the paper)
#    --ffn-type mlp | mlp_dw | mlp_dw_gate | kan     FFN ladder
#    --no-ppm                                        remove pyramid pooling
#    --no-aux                                        remove deep supervision
#    --lovasz-weight 0                               remove the IoU surrogate
#    --hed-sigma 0                                   remove stain augmentation
#    --boundary-weight 0                             remove boundary weighting
#    Each is a single flag, so the ablation table is one sbatch array away.
