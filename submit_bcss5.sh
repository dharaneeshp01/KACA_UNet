#!/bin/bash
# =============================================================================
# submit_bcss5.sh — KAT-UNet on the OFFICIAL 5-class BCSS benchmark
# =============================================================================
#SBATCH --job-name=submit_bcss5
#SBATCH --account=iitp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8   # run 103167 was GRANTED 8 despite requesting 16
#SBATCH --time=12:00:00
#SBATCH --output=logs/out_bcss5_%j.out
#SBATCH --error=logs/out_bcss5_%j.err

set +euo pipefail
mkdir -p logs

echo "=============================================="
echo "JOB      = ${SLURM_JOB_ID:-na}"
echo "NODE     = ${SLURM_JOB_NODELIST:-na}"
echo "CPUS     = ${SLURM_CPUS_ON_NODE:-na}"
echo "STARTED  = $(date)"
echo "=============================================="

module load MLDL/miniconda3
export QT_XCB_GL_INTEGRATION=""
export CONDA_BACKUP_QT_XCB_GL_INTEGRATION=""
eval "$(conda shell.bash hook)"
source activate mtp_py.310

set -euo pipefail
module load cuda 2>/dev/null || true

# 8 CPUs -> 7 dataloader workers. Run 103167 used 6 of 8 and was data-starved.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT=/home/dharaneeshp_iitp/dekaht_Unet
DATA=$ROOT/data/BCSS5_patches          # train/{images,masks}, val/..., test/...
CKPT=$ROOT/swin_tiny_patch4_window7_224.pth
OUT=$ROOT/out_bcss5_run1
STAIN_REF=""                           # set to a clean tumour patch to enable Macenko

cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

# ---------------------------------------------------------------------------
# STEP 1 — self-test: 130 assertions, CPU, ~5 min. Nothing runs if this fails.
# ---------------------------------------------------------------------------
echo ""
echo ">>> [1/4] SELF-TEST"
# python train_bcss5.py --self-test

# ---------------------------------------------------------------------------
# STEP 2 — training
# ---------------------------------------------------------------------------
# WHAT KILLED RUN 103167, AND WHAT CHANGED
#
# 1) CRASH  RuntimeError: main=nan aux=nan at step 6359.
#    FocalTversky computes (1-tv)**0.75. For a patch containing ONE class that the
#    model predicts near-perfectly, tv == 1.0 exactly, so the base is 0. The FORWARD
#    pass returns a clean 0.0 -- but d/dx[x**0.75] = 0.75*x**(-0.25) is +INF at x=0.
#    Reproduced exactly: grad_norm=nan. Fixed by clamping the base to [1e-6, 1].
#    The loss now also names the offending TERM instead of "main=nan".
#
# 2) THROTTLE  The clipper calibrated over steps 0-300, i.e. DURING warmup, when the
#    LR was 2e-05..6e-05 and gnorm was 1.06..3.36. It froze 6.70 while the
#    steady-state gnorm was 8.2-10.0, so 50-61% of steps were clipped.
#    Calibration now starts AFTER warmup (--clip-calib-start -1) and the ceiling is
#    25 rather than 10 so the p90 is not itself clamped. Replayed against the logged
#    trajectory: 100% clipped -> 9.2% clipped.
#
# 3) WALLTIME  1.46 s/it at batch 8 x 13,054 it = 5.29 h/EPOCH. 35 epochs = 185 h
#    against a 36 h grant: the job dies around epoch 6 even without the NaN.
#    Fixes: batch 24 (an A100 80GB was running batch 8), 7 workers not 6, aux heads
#    scored at 56x56 instead of 224x224 (measured 13.8x cheaper, 259 ms/step saved),
#    float32 HED jitter (3.76 -> 2.16 ms/patch), elastic 0.15 -> 0.05.
#
# 4) CADENCE  --iters-per-epoch 3000 decouples the validation/checkpoint cycle from
#    a full 104k-patch pass. A crash now costs minutes, not 5 hours, and you see a
#    val curve early enough to kill a bad run. The sampler reshuffles every epoch,
#    so all data is still seen; 30 x 3000 x 24 = 2.16M sample-views.
#
# MEASURE BEFORE YOU COMMIT: watch the s/it in the first 200 iterations. If it is
# not near ~0.45 s/it at batch 24, you are data-bound, not compute-bound -- drop
# --clahe-p and --elastic-p to 0 and re-check before touching anything else.
#
# WHY THESE FLAGS (each maps to a post-mortem item or a measured failure):
#   --aux-budget 0.30        TOTAL over all 4 DS heads, decayed towards shallow.
#                            UC6 used 0.3 PER HEAD = 1.2 > main. [P1]
#   --grad-clip-mode calibrated --grad-clip-ceiling 10
#                            Threshold is measured for 300 steps then FROZEN, and
#                            can never exceed 10. UC4 used a fixed 1.0 against
#                            grad norms of 10-22 (~100% of steps throttled 20x);
#                            UC6 used a rolling median that grew with the
#                            explosion and never bound. [P3]
#   --max-aux-ratio 5.0      Hard abort if aux/main exceeds 5. UC6 reached 7e8
#                            and limped for 75 epochs. [P4]
#   --aug-mode rrc           Patches are natively 224. Scaling DOWN then padding
#                            fabricates black borders + ignore labels that never
#                            occur at validation.
#   --amp-dtype bf16         fp16 produced `gnorm inf` in ~60% of UC2 epochs on
#                            this A100. bf16 has the same exponent range as fp32.
#   --class-weights auto     Inverse-sqrt frequency, normalised to MEAN 1.
#                            Normalising to SUM 1 (as the guide does) scales the
#                            whole CE term down ~5x relative to Dice.
echo ""
echo ">>> [2/4] TRAINING"
python train_bcss5.py \
  --data-root "$DATA" --output-dir "$OUT" \
  --img-subdir images --mask-subdir masks \
  --train-split train --val-split-name val --test-split-name test \
  --num-classes 5 \
  --img-size 224 --eval-size 224 --patch-size 4 --window-size 7 \
  --model-variant swkat-tiny \
  --decoder-depths 2,2,2 --bottleneck-depth 2 \
  --ffn-type kan --gate-mode residual_tanh --groups 8 \
  --kernel-sizes 3,5,7 \
  --drop-path-rate 0.15 --head-dropout 0.05 \
  --encoder-pretrained "$CKPT" --min-pretrained-coverage 0.90 \
  --epochs 30 \
  --iters-per-epoch 3000 \
  --batch-size 8 --accum-steps 2 --workers 7 \
  --opt adamw --lr 3e-4 --encoder-lr-scale 0.4 --layer-decay 0.90 \
  --weight-decay 0.05 \
  --sched cosine --warmup-iters 1500 --min-lr-ratio 0.01 \
  --grad-clip-mode calibrated --grad-clip-ceiling 25.0 --grad-clip-floor 1.0 \
  --clip-calib-iters 300 --clip-percentile 90 --clip-calib-start -1 \
  --ema-decay 0.9998 \
  --aug-mode rrc --rrc-min-scale 0.35 \
  --hed-sigma 0.05 --clahe-p 0.20 --elastic-p 0.05 \
  --ce-weight 1.0 --dice-weight 1.0 \
  --tversky-weight 0.5 --tversky-alpha 0.3 --tversky-beta 0.7 --tversky-gamma 0.75 \
  --lovasz-weight 0.5 --lovasz-start-frac 0.4 \
  --label-smoothing 0.05 --boundary-weight 0.5 --boundary-head-weight 0.2 \
  --class-weights auto --class-weight-power 0.5 --class-weight-clip 4.0 \
  --aux-budget 0.30 --aux-decay 0.6 --max-aux-ratio 5.0 --ds-loss-size 56 \
  --swa-last 6 \
  --amp-dtype bf16 --tta --tta-scales 1.0 \
  --eval-interval 1 --log-interval 200 --seed 1234 \
  ${STAIN_REF:+--stain-norm "$STAIN_REF"}

# ---------------------------------------------------------------------------
# STEP 3 — VAL scores (best checkpoint and the SWA average), multi-scale TTA
# ---------------------------------------------------------------------------
echo ""
echo ">>> [3/4] VALIDATION SCORES"
for W in best_model swa_model; do
  [ -f "$OUT/$W.pth" ] || continue
  if [ "$W" = "swa_model" ]; then FLAG="--swa-only"; else FLAG="--eval-only"; fi
  python train_bcss5.py \
    --data-root "$DATA" --output-dir "$OUT" --num-classes 5 \
    --img-size 224 --eval-size 224 --workers 8 --batch-size 16 \
    --window-size 7 --patch-size 4 --kernel-sizes 3,5,7 \
    --decoder-depths 2,2,2 --bottleneck-depth 2 \
    $FLAG "$OUT/$W.pth" --eval-weights ema --eval-on val \
    --tta --tta-scales 0.75,1.0,1.25 --amp-dtype bf16
done

# ---------------------------------------------------------------------------
# STEP 4 — HELD-OUT TEST (the six official institutions). Run ONCE.
# ---------------------------------------------------------------------------
# This is the number that goes in the paper. Do not use it to select anything:
# every architecture and hyper-parameter decision must be made on val only.
echo ""
echo ">>> [4/4] OFFICIAL TEST SET"
python train_bcss5.py \
  --data-root "$DATA" --output-dir "$OUT" --num-classes 5 \
  --img-size 224 --eval-size 224 --workers 8 --batch-size 16 \
  --window-size 7 --patch-size 4 --kernel-sizes 3,5,7 \
  --decoder-depths 2,2,2 --bottleneck-depth 2 \
  --eval-only "$OUT/best_model.pth" --eval-weights ema --eval-on test \
  --tta --tta-scales 0.75,1.0,1.25 --amp-dtype bf16

echo ""
echo "FINISHED = $(date)"

# =============================================================================
# WHAT TO CHECK IN THE FIRST 10 MINUTES OF THE LOG
# =============================================================================
#  [split] verified: institution-disjoint across all splits
#          -> if it says LEAK, stop. The protocol is institution-level.
#  [label-audit] raw values ... 0(IGNORE), 1(Tumor) ... 5(Other)
#          -> values above 5 abort the job: masks are not remapped.
#  [pretrained] tensors 171/171 (100.0%) | params 27.52M/27.52M (100.0%)
#  [ds-guard] 3 deep-supervision heads verified ... classifiers zero-initialised
#  [aux] weights [0.1379, 0.0827, 0.0496, 0.0298] | TOTAL 0.3000
#  [sanity] PASSED
#  [label-audit] scanning 500 RANDOMLY sampled masks out of 104438
#          -> run 103167 scanned the first 500 (= one slide) and mis-reported the
#             class distribution as Inflammatory 0.07% / Necrosis 31.41% when the
#             true figures were 13.28% / 6.84%.
#  Train e0 ... a/m 0.2xx  clip ceiling 25.0 (warmup n/1500)
#          -> then "calibrating n/300", then a frozen threshold near 10.
#          -> a/m must stay well under 1.0; clip rate must settle UNDER ~20%.
#          -> s/it must be near 0.45 at batch 24. If it is 1.5, you are data-bound.
#
# =============================================================================
# FOLLOW-UP RUNS — submit all three, they decide the paper
# =============================================================================
#
# RUN B — CAPACITY. You have been underfitting since UC4 and have never tried a
#   larger encoder. This is the largest untried lever.
#     wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
#     ... --model-variant swkat-small --encoder-pretrained ./swin_small_patch4_window7_224.pth \
#         --batch-size 16 --accum-steps 2 --output-dir $ROOT/out_bcss5_small
#
# RUN C — THE KAN ABLATION. The paper's central claim.
#     ... --ffn-type mlp --output-dir $ROOT/out_bcss5_mlp
#   If GR-KAN does not beat a plain MLP FFN under an identical optimiser, you
#   need to know that BEFORE writing the abstract, not after.
#
# RUN D — NOISE ROBUSTNESS. BCSS masks are crowd-sourced; the hardest pixels are
#   disproportionately mislabelled.
#     ... --gce-q 0.7 --output-dir $ROOT/out_bcss5_gce
#
# ABLATION TABLE (one flag each, all on val):
#     --kernel-sizes 3              remove multi-kernel GR-KAN   (N1)
#     --no-deep-supervision         remove the DS heads
#     --no-fpn                      remove multi-level fusion
#     --no-boundary --boundary-head-weight 0
#     --tversky-weight 0            remove Focal Tversky (watch Necrosis)
#     --lovasz-weight 0             remove the IoU surrogate
#     --swa-last 0                  remove weight averaging
#     --aug-mode scale_pad          reproduce the UC5 padding bug
#     --aux-budget 0.35 --aux-decay 1.0   flat DS weighting
