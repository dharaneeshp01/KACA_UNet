#!/bin/bash
#SBATCH --job-name=dekaht_bcss_run7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --output=submit_bcss_UC2_run7_%j.log
#SBATCH --error=train_bcss_UC2_run7_%j.err
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
#    --resume ./out_bcss_updatedC_run4_seg/last_model.pth \

$PYTHON_BIN train_dekaht_unet_UC3.py \
    --data-root $DATA_DIR \
    --data-format folder \
    --img-subdir images --mask-subdir masks \
    --img-size 224 --in-chans 3 --num-classes 3 \
    --batch-size 16 --accum-steps 1 --workers 8 \
    --epochs 200 --warmup-epochs 10 \
    --lr 2e-4 --encoder-lr-scale 0.1 \
    --kan-l2 1e-6 --gate-bias 4.0 \
    --aug-strength light \
    --drop-path-rate 0.1 \
    --weight-decay 0.05 --min-lr-ratio 0.01 \
    --ema-decay 0.9998 --grad-clip 5.0 \
    --ce-weight 0.5 --dice-weight 0.5 --focal-gamma 1.0 \
    --class-weights '1.000,0.824,1.035' \
    --tversky-weight 0.2 --tversky-alpha 0.3 --tversky-beta 0.7 \
    --tta \
    --model-variant swkat-tiny --ffn-type kan --enforce-kan \
    --encoder-pretrained ./swin_tiny_patch4_window7_224.pth \
    --resume ./out_bcss_UC3_run7/last_model.pth \
    --output-dir ./out_bcss_UC3_run7
