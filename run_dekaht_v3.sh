#!/bin/bash
# =============================================================================
# run_dekaht_v2.sh - DeKAHT-UNet v2 training
#
#   sbatch run_dekaht_v2.sh check      # STEP 0: verify labels, exits (30 s)
#   sbatch run_dekaht_v2.sh overfit    # STEP 1: pipeline sanity, 32 imgs (~10 min)
#   sbatch run_dekaht_v2.sh voc        # STEP 2: Pascal VOC, your "90+ mDice" track
#   sbatch run_dekaht_v2.sh coco       # STEP 3: full COCO
#   sbatch run_dekaht_v2.sh eval       # TTA evaluation of a trained checkpoint
#
# Files needed in this directory:
#   dekaht_v2.py  train_dekaht_v2.py  swin_tiny_patch4_window7_224.pth
# =============================================================================
#SBATCH --job-name=dekaht_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16          # was 8; --workers 2 was starving the GPU
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=dekaht_v2_%j.log
#SBATCH --error=dekaht_v2_%j.err

set -euo pipefail
MODE="${1:-coco}"
PY="$HOME/.conda/envs/mtp_py.310/bin/python"
PRETRAINED="./swin_tiny_patch4_window7_224.pth"

# ---------------------------------------------------------------- COCO symlinks
setup_coco () {
  mkdir -p ./data/coco_formatted/{train,val}
  ln -sfn "$(pwd)/data/coco/train2017"       ./data/coco/train2017
  ln -sfn "$(pwd)/data/coco/train2017_masks" ./data/coco/train2017_masks
  ln -sfn "$(pwd)/data/coco/val2017"         ./data/coco/val2017
  ln -sfn "$(pwd)/data/coco/val2017_masks"   ./data/coco/val2017_masks
}

# =============================================================================
case "$MODE" in

# --- STEP 0 ------------------------------------------------------------------
# Reads the raw label plane and compares it against PIL convert('L') and
# cv2.IMREAD_UNCHANGED. Your old dataset used convert('L'), which maps a palette
# INDEX to the LUMINANCE of the palette COLOUR (class 3 -> 87). If this reports
# the palette bug, every number from the 100-epoch run is meaningless.
# Also reports out-of-range labels and whether ignore_index is present.
check)
  setup_coco
  $PY train_dekaht_v2.py --data-root ./data/coco_formatted \
      --num-classes 81 --ignore-index 255 --check-masks --output-dir ./out_check
  echo
  echo "Also confirm in your COCO->mask generator:"
  echo "  * iscrowd regions written as 255 (ignore), NOT background"
  echo "  * instances painted LARGEST-AREA-FIRST so small objects land on top"
  echo "    (painting in annotation order lets a sofa erase the remote on it,"
  echo "     and 'remote' scored 0.10 last run)"
  ;;

# --- STEP 1 ------------------------------------------------------------------
# Trains and validates on the SAME 32 images.
#
# READ THE "GT-present only" LINE, NOT THE MAIN mDice.
#   With 32 images only ~33 of 81 classes appear in the ground truth. The main
#   mDice also counts classes the model PREDICTED but which are absent from the
#   GT -- each scores 0 and is averaged in. That is correct on the full val set
#   (all 80 classes occur, false positives should hurt) but it deflates a
#   32-image subset badly: 0.766 over the learned classes became 0.588 reported.
#
# GATE:  "GT-present only: mDice" must exceed ~0.90.
#
# 600 epochs, not 60. 32 imgs / batch 8 = 4 optimiser steps per epoch, so 60
# epochs is only 240 steps -- 20 of them warmup. You cannot memorise 32 COCO
# images with 81 classes in 240 steps. 600 epochs = 2400 steps, ~40 min at the
# 3.6 s/epoch this node showed.
overfit)
  setup_coco
  $PY train_dekaht_v2.py --data-root ./data/coco_formatted \
      --num-classes 81 --ignore-index 255 \
      --overfit 32 --crop-size 448 --base-size 448 --val-size 448 \
      --scale-min 0.9 --scale-max 1.1 --cat-max-ratio 0 \
      --batch-size 8 --epochs 600 --eval-interval 100 \
      --lr 1e-3 --warmup-iters 50 --drop-path-rate 0.0 \
      --encoder-pretrained "$PRETRAINED" \
      --workers 8 --output-dir ./out_overfit
  ;;

# --- STEP 2 : the "90+ mDice" track ------------------------------------------
# 21 classes, 10,582 train images, ~40 min/run. Pascal VOC SOTA is ~90 mIoU,
# i.e. ~0.95 mDice. This is where your headline number lives -- NOT COCO, where
# even Mask2Former/OneFormer only reach ~45 mIoU (~0.62 mDice) on 172 classes.
voc)
  $PY train_dekaht_v2.py --data-root ./data/voc_formatted \
      --num-classes 21 --ignore-index 255 \
      --crop-size 448 --base-size 448 --val-size 448 \
      --scale-min 0.5 --scale-max 2.0 --cat-max-ratio 0.75 \
      --model-variant swkat-tiny --encoder-pretrained "$PRETRAINED" \
      --gate-mode depthwise --head-dim 256 \
      --epochs 60 --batch-size 16 --accum-steps 1 \
      --lr 2e-4 --layer-decay 0.7 --weight-decay 0.01 \
      --sched poly --poly-power 1.0 --warmup-iters 500 \
      --loss ohem --dice-weight 0.3 --aux-weight 0.4 \
      --drop-path-rate 0.2 --grad-clip 1.0 --amp-dtype bf16 \
      --workers 12 --save-viz --output-dir ./out_voc
  ;;

# --- STEP 3 : COCO -----------------------------------------------------------
# crop 448 makes every stage tile EXACTLY with window 7:
#   448/4 = 112 -> 112/7=16  56/7=8  28/7=4  14/7=2   (zero window padding)
# 45 epochs, not 100: mmseg's 160k iters @ bs16 is ~22 COCO epochs. Epochs were
# never the bottleneck -- the un-loaded pretrained encoder was.
coco)
  setup_coco
  $PY train_dekaht_v2.py --data-root ./data/coco_formatted \
      --num-classes 81 --ignore-index 255 \
      \
      --crop-size 448 --base-size 448 --val-size 448 \
      --scale-min 0.5 --scale-max 2.0 --cat-max-ratio 0.75 \
      \
      --model-variant swkat-tiny \
      --encoder-pretrained "$PRETRAINED"  `# RAISES if coverage < 90%` \
      --min-pretrained-coverage 0.90 \
      --gate-mode depthwise               `# full gate = 9.4M/block at dim 768` \
      --head-dim 256 --bottleneck-depth 2 --decoder-depths 2,2,2 \
      --drop-path-rate 0.2 \
      \
      --epochs 45 --batch-size 16 --accum-steps 1 \
      --lr 2e-4 --layer-decay 0.7         `# head 2e-4 ... patch_embed 3.4e-5` \
      --weight-decay 0.01                 `# and 0.0 on norms/biases/rel-pos` \
      --sched poly --poly-power 1.0 \
      --warmup-iters 1500                 `# was 10 epochs = 74,000 iters` \
      --grad-clip 1.0 --amp-dtype bf16 \
      \
      --loss ohem --ohem-thresh 0.7 --ohem-min-kept 100000 \
      --dice-weight 0.3                   `# presence-masked dice` \
      --aux-weight 0.4                    `# deep supervision` \
      \
      --workers 12 --eval-interval 1 --log-interval 100 --save-viz \
      --tta-scales 0.75,1.0,1.25,1.5 \
      --output-dir ./out_coco_v2
  ;;

# --- TTA evaluation of an existing checkpoint --------------------------------
eval)
  setup_coco
  $PY train_dekaht_v2.py --data-root ./data/coco_formatted \
      --num-classes 81 --ignore-index 255 --val-size 448 --batch-size 8 \
      --eval-only --tta --tta-scales 0.75,1.0,1.25,1.5 \
      --resume ./out_coco_v2/last_model.pth --workers 12 --output-dir ./out_coco_v2
  ;;

*) echo "usage: sbatch $0 {check|overfit|voc|coco|eval}"; exit 1 ;;
esac
