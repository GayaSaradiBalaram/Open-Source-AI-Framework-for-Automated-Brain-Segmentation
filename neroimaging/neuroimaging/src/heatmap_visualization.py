import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the Segmented MRI
segmented_mri_path = r"C:\Users\saradhi\neroimaging\data\segmented_mri.nii.gz"

try:
    segmented_img = nib.load(segmented_mri_path)
    segmented_data = segmented_img.get_fdata()
    print(f"Segmented MRI loaded! Shape: {segmented_data.shape}")
except FileNotFoundError:
    print("Segmented MRI file not found! Ensure segmentation was completed.")
    exit()

# ✅ Select a Single Channel (First Channel) if necessary
if len(segmented_data.shape) == 4 and segmented_data.shape[0] == 3:
    segmented_data = segmented_data[0]  # Take the first channel

# ✅ Select a Middle Slice for Visualization
middle_slice = segmented_data.shape[2] // 2
slice_data = segmented_data[:, :, middle_slice]

# ✅ Ensure 2D Input for Seaborn
if len(slice_data.shape) != 2:
    print(f"Shape error: Expected 2D but got {slice_data.shape}. Selecting first slice.")
    slice_data = slice_data[:, :, 0]  # Select a single slice if needed

# ✅ Generate Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(slice_data, cmap="jet", cbar=True)

plt.title(f"Heatmap Overlay - Slice {middle_slice}")
plt.axis("off")
plt.show()
