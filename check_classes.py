import os
import numpy as np

# Updated path based on your 'find' command
data_dir = './data/my_dataset/train'

all_unique_classes = set()

print(f"Searching for files in: {data_dir}")
files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
print(f"Found {len(files)} files. Scanning... (this might take a few seconds)")

for f in files:
    file_path = os.path.join(data_dir, f)
    data = np.load(file_path)
    
    # .npz files can hold multiple arrays. We check all of them.
    for key in data.files:
        arr = data[key]
        unique_vals = np.unique(arr)
        
        # Images have thousands of unique pixel values. 
        # Segmentation masks only have a handful of integer labels (like 0 to 8).
        # This safely isolates the mask data.
        if len(unique_vals) < 20: 
            all_unique_classes.update(unique_vals)

print("\n" + "="*40)
print("             RESULTS")
print("="*40)
print(f"The exact class labels are: {sorted(list(all_unique_classes))}")
print(f"TOTAL NUMBER OF CLASSES: {len(all_unique_classes)}")
print("="*40)
