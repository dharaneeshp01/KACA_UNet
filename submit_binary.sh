#!/bin/bash
#SBATCH --job-name=dekaht_train
#SBATCH --output=logs/optimized_%j_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

# Activate your environment
source activate mtp_py

cd /home/dharaneeshp_iitp/dekaht_Unet/
python train_dekaht_unet.py \
    --data-root ./data/BUSI_npz \
    --img-size 224 \
    --in-chans 3 \
    --batch-size 4 \
    --epochs 200 \
    --lr 1e-5 \
    --weight-decay 0.05 \
    --ce-weight 0.4 \
    --dice-weight 0.6 \
    --model-variant swkat-tiny \
    --encoder-pretrained ./wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    --output-dir ./out_BUSI_Binary_2_seg
