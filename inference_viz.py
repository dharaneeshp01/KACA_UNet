import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import your unified architecture
try:
    from dekaht_unet_unified import dekaht_unet
except ImportError:
    from dekaht_unet import dekaht_unet

def main():
    # --- Configuration ---
    weights_path = "./out_bcss_updatedC_run4_seg/last_model.pth"
    val_img_dir = "./data/BCSS_formatted/val/images"
    val_mask_dir = "./data/BCSS_formatted/val/masks"
    output_dir = "./inference_results"
    
    img_size = 224
    num_classes = 3
    in_chans = 3
    num_samples = 5  # How many images to visualize
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # --- 1. Load the Model ---
    print("Loading model architecture and weights...")
    model = dekaht_unet(
        variant="swkat-tiny", img_size=img_size, patch_size=4,
        in_chans=in_chans, num_classes=num_classes, window_size=7,
        mlp_ratio=4.0, drop_path_rate=0.2, groups=8,
        decoder_depths=(2, 2, 2), bottleneck_depth=2
    )
    
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # --- 2. Get Validation Images ---
    # Find image paths and match them with masks
    img_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))[:num_samples]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # --- 3. Run Inference and Plot ---
    print(f"Generating {num_samples} visualization grids...")
    for idx, img_path in enumerate(img_paths):
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(val_mask_dir, base_name) # Assumes mask has the same name
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask for {base_name} not found. Skipping.")
            continue

        # Load and preprocess image
        pil_img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        img_norm = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        # Load ground truth mask
        pil_mask = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
        mask_np = np.asarray(pil_mask, dtype=np.uint8)
        mask_np = np.clip(mask_np, 0, num_classes - 1) # Ensure valid classes
        
        # Predict
        with torch.no_grad():
            logits = model(img_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Input Image\n({base_name})")
        axes[0].axis('off')
        
        # Use a qualitative colormap (like 'viridis' or 'Set1') to distinguish the 3 classes clearly
        axes[1].imshow(mask_np, cmap='viridis', vmin=0, vmax=num_classes-1)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap='viridis', vmin=0, vmax=num_classes-1)
        axes[2].set_title("Model Prediction")
        axes[2].axis('off')
        
        out_file = os.path.join(output_dir, f"result_{idx}.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close()
        
    print(f"Done! Check the '{output_dir}' folder for the images.")

if __name__ == "__main__":
    main()
