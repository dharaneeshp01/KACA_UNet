#!/bin/bash
#SBATCH --job-name=unet_14class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=training_log_%j.txt

module load MLDL/miniconda3
source activate mtp_py.310

cd /home/dharaneeshp_iitp/dekaht_Unet/

python train_dekaht_unet.py \
    --data-root ./data/my_dataset \
    --data-format npz \
    --num-classes 14 \
    --epochs 100 \
    --img-size 512 \
    --batch-size 4 \
    --ce-weight 0.2 \
    --dice-weight 0.8 \
    --lr 1e-4 \
    --workers 4 \
    --encoder-pretrained ./out_seg/pretrained_encoder.pth \
    --output-dir ./out_seg_14class
