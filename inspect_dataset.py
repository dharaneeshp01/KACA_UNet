import os
import glob
import numpy as np

# Point this to your validation folder (or train folder)
data_dir = "./data/my_dataset/val" # Change to 'train' if you don't have a val folder
files = glob.glob(os.path.join(data_dir, "*.npz"))

if not files:
    print(f"No .npz files found in {data_dir}. Checking train folder instead...")
    data_dir = "./data/my_dataset/train"
    files = glob.glob(os.path.join(data_dir, "*.npz"))

print(f"Scanning {len(files)} files in {data_dir}...\n")

global_counts = {}
files_with_class_12 = []

for f in files:
    data = np.load(f)
    label = data["label"]
    
    # Find unique classes and their pixel counts in this file
    unique, counts = np.unique(label, return_counts=True)
    
    for u, c in zip(unique, counts):
        global_counts[u] = global_counts.get(u, 0) + c
        
    # Check if class 12 is in this specific image
    if 12 in unique:
        files_with_class_12.append(f)

print("=== TOTAL PIXEL COUNTS PER CLASS ===")
for k in sorted(global_counts.keys()):
    print(f"Class {k:2}: {global_counts[k]:12,} pixels")

print("\n=== CLASS 12 INSPECTION ===")
if len(files_with_class_12) == 0:
    print("CRITICAL: Class 12 DOES NOT EXIST in this folder!")
else:
    print(f"Class 12 found in {len(files_with_class_12)} files.")
    print("Here are the first 5 files containing Class 12:")
    for f in files_with_class_12[:5]:
        print(f" -> {f}")
