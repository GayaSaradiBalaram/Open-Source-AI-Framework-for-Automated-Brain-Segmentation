import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# ✅ Load the segmented MRI scan
segmented_path = r"C:\Users\saradhi\neroimaging\data\segmented_mri.nii.gz"

try:
    segmented_img = nib.load(segmented_path)
    segmented_data = segmented_img.get_fdata()
    print(f"Segmented MRI loaded successfully! Shape: {segmented_data.shape}")
except FileNotFoundError:
    print("Segmented MRI file not found! Run segmentation first.")
    exit()

# ✅ Select a middle slice
middle_slice = segmented_data.shape[2] // 2

# 🔍 Debugging: Print shape before visualization
print(f"Shape of segmented data before visualization: {segmented_data.shape}")

# Handle 3-channel MRI case
if len(segmented_data.shape) == 4 and segmented_data.shape[0] == 3:
    print("Segmented MRI has 3 channels. Using only the first channel for visualization.")
    segmented_data = segmented_data[0]  # Take only the first channel

# ✅ Display the segmented MRI slice
plt.figure(figsize=(6, 6))
plt.imshow(segmented_data[:, :, middle_slice], cmap="jet")
plt.title(f"Segmented MRI - Slice {middle_slice}")
plt.axis("off")
plt.show()
