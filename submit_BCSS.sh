#!/bin/bash
#SBATCH --job-name=dekaht_bcss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=train_bcss_%j.log
#SBATCH --error=train_bcss_%j.err

echo "=========================================================="
echo "Starting DeKAHT-UNet Training on BCSS Dataset"
echo "Date: $(date)"
echo "Node: $HOSTNAME"
echo "=========================================================="

# 1. Automate Folder Restructuring (Fixes the path mismatch)
# The PyTorch dataloader expects data_root/train/images and data_root/train/masks.
# This block safely converts your downloaded structure into the required format.
DATA_DIR="./data/BCSS/BCSS"

if [ -d "$DATA_DIR/train_mask" ]; then
    echo "Restructuring BCSS dataset folders for PyTorch Dataloader..."
    
    # Temporarily rename image folders to avoid moving conflicts
    mv $DATA_DIR/train $DATA_DIR/train_tmp
    mv $DATA_DIR/val $DATA_DIR/val_tmp
    
    # Create the standard PyTorch format directories
    mkdir -p $DATA_DIR/train/images $DATA_DIR/train/masks
    mkdir -p $DATA_DIR/val/images $DATA_DIR/val/masks
    
    # Move the data into the correct nested folders
    mv $DATA_DIR/train_tmp/* $DATA_DIR/train/images/
    mv $DATA_DIR/val_tmp/* $DATA_DIR/val/images/
    mv $DATA_DIR/train_mask/* $DATA_DIR/train/masks/
    mv $DATA_DIR/val_mask/* $DATA_DIR/val/masks/
    
    # Clean up the empty temporary folders
    rm -rf $DATA_DIR/train_tmp $DATA_DIR/val_tmp $DATA_DIR/train_mask $DATA_DIR/val_mask
    
    echo "Folder restructuring complete!"
else
    echo "Dataset already in standard format. Proceeding to training..."
fi

# 2. Initialize Conda Environment (Adjust path if your conda is elsewhere)
eval "$(/opt/apps/anaconda3/bin/conda shell.bash hook)"
conda activate mtp_py

# 3. Launch the stabilized, hyper-optimized training run
python train_dekaht_unet.py \
    --data-root $DATA_DIR \
    --data-format folder \
    --img-size 224 \
    --in-chans 3 \
    --num-classes 3 \
    --batch-size 4 \
    --epochs 200 \
    --lr 3e-5 \
    --warmup-epochs 10 \
    --weight-decay 0.05 \
    --drop-path-rate 0.2 \
    --ce-weight 0.4 \
    --dice-weight 0.6 \
    --model-variant swkat-tiny \
    --encoder-pretrained ./swin_tiny_patch4_window7_224.pth \
    --output-dir ./out_BCSS_run1_seg

echo "=========================================================="
echo "Training Completed: $(date)"
echo "=========================================================="
