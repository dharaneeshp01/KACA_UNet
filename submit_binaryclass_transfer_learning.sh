#!/bin/bash
#SBATCH --job-name=DeKAHT_binary_transfer
#SBATCH --output=logs/binary_%j_output.txt
#SBATCH --error=logs/binary_%j_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

echo "=========================================================="
echo "Starting Multi-Class Transfer Learning Job (1 GPU)"
echo "Image Resolution: 224x224 (Native Swin-Tiny mapping)"
echo "Effective Batch Size: 32 (Micro-batch 8 x 4 Accumulation)"
echo "Date: $(date)"
echo "Node: $HOSTNAME"
echo "=========================================================="

source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate mtp_py.310

pip install --user albumentations

python train_dekaht_unet.py \
  --data-root "./data/BUSI_npz" \
  --data-format "npz" \
  --output-dir "./out_transfer_binary" \
  --num-classes 2 \
  --img-size 224 \
  --batch-size 8 \
  --workers 8 \
  --epochs 150 \
  --opt "adamw" \
  --lr 2.5e-5 \
  --weight-decay 0.05 \
  --warmup-epochs 5 \
  --ce-weight 0.5 \
  --dice-weight 0.5 \
  --ignore-index -100 \
  --encoder-pretrained "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"

echo "=========================================================="
echo "Job Submitted successfully."
echo "=========================================================="
