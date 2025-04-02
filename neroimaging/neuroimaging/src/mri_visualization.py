import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ✅ Path to the downloaded MRI scan
mri_path = r"C:\Users\saradhi\neroimaging\data\CT_AVM.nii.gz"

# ✅ Load the MRI scan
try:
    mri_img = nib.load(mri_path)
    mri_data = mri_img.get_fdata()
    print(f"MRI scan loaded! Shape: {mri_data.shape}")
except FileNotFoundError:
    print("MRI scan file not found! Check the file path.")
    exit()

# ✅ Get the middle slice from each plane
slice_x = mri_data[mri_data.shape[0] // 2, :, :]  # Sagittal (X-axis)
slice_y = mri_data[:, mri_data.shape[1] // 2, :]  # Coronal (Y-axis)
slice_z = mri_data[:, :, mri_data.shape[2] // 2]  # Axial (Z-axis)

# ✅ Plot slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(slice_x.T, cmap="gray", origin="lower")
axes[0].set_title("Sagittal Slice (X-axis)")

axes[1].imshow(slice_y.T, cmap="gray", origin="lower")
axes[1].set_title("Coronal Slice (Y-axis)")

axes[2].imshow(slice_z.T, cmap="gray", origin="lower")
axes[2].set_title("Axial Slice (Z-axis)")

plt.tight_layout()
plt.show()
