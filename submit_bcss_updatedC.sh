#!/bin/bash
#SBATCH --job-name=dekaht_bcss_run4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --output=submit_bcss_updatedC_run4_%j.log
#SBATCH --error=train_bcss_updatedC_run4_%j.err

echo "Setting up virtual folder structure..."
mkdir -p ./data/BCSS_formatted/train
mkdir -p ./data/BCSS_formatted/val

ln -sfn $(pwd)/data/BCSS/train       ./data/BCSS_formatted/train/images
ln -sfn $(pwd)/data/BCSS/train_mask  ./data/BCSS_formatted/train/masks
ln -sfn $(pwd)/data/BCSS/val         ./data/BCSS_formatted/val/images
ln -sfn $(pwd)/data/BCSS/val_mask    ./data/BCSS_formatted/val/masks

PYTHON_BIN="$HOME/.conda/envs/mtp_py.310/bin/python"
DATA_DIR="./data/BCSS_formatted"

echo "Starting training on A100..."
#    --encoder-pretrained ./swin_tiny_patch4_window7_224.pth \

$PYTHON_BIN train_dekaht_unet_updatedC.py \
    --data-root $DATA_DIR \
    --data-format folder \
    --img-subdir images \
    --mask-subdir masks \
    --img-size 224 \
    --in-chans 3 \
    --num-classes 3 \
    --batch-size 8 \
    --accum-steps 2 \
    --workers 8 \
    --epochs 250 \
    --warmup-epochs 15 \
    --lr 1e-4 \
    --encoder-lr-scale 0.1 \
    --weight-decay 0.05 \
    --min-lr-ratio 0.01 \
    --drop-path-rate 0.2 \
    --ema-decay 0.9998 \
    --grad-clip 1.0 \
    --ce-weight 0.4 \
    --dice-weight 0.6 \
    --focal-gamma 2.0 \
    --class-weights '1.000,0.824,1.035' \
    --tversky-weight 0.3 \
    --tversky-alpha 0.3 \
    --tversky-beta 0.7 \
    --tta \
    --model-variant swkat-tiny \
    --resume ./out_bcss_updatedC_run4_seg/last_model.pth \
    --output-dir ./out_bcss_updatedC_run4_seg
