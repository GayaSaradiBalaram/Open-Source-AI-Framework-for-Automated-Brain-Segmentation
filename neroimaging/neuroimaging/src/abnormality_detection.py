import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure

# Load the segmented MRI scan
segmented_path = r"C:\Users\saradhi\neroimaging\data\segmented_mri.nii.gz"
try:
    segmented_img = nib.load(segmented_path)
    segmented_data = segmented_img.get_fdata()
    print(f"Segmented MRI loaded successfully! Shape: {segmented_data.shape}")
except FileNotFoundError:
    print("Segmented MRI file not found! Ensure 'segmented_mri.nii.gz' exists in the Scripts/ folder.")
    exit()

# Select a middle slice
middle_slice = segmented_data.shape[2] // 2
slice_data = segmented_data[:, :, middle_slice]

# Convert to grayscale if it has three channels
if slice_data.ndim == 3 and slice_data.shape[-1] == 3:
    print("⚠ MRI slice has 3 channels. Converting to grayscale.")
    slice_data = slice_data[..., 0]  # Use only the first channel

# Debugging: Print shape
print(f"Final slice shape for visualization: {slice_data.shape}")

# Apply thresholding to highlight potential abnormalities
threshold = filters.threshold_otsu(slice_data)
abnormal_regions = slice_data > threshold  # Binary mask

# Label connected abnormal regions
labeled_regions = measure.label(abnormal_regions)

# Display the MRI with abnormality overlay
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(slice_data, cmap="gray")
ax[0].set_title("Segmented MRI - Original")
ax[0].axis("off")

ax[1].imshow(slice_data, cmap="gray")
ax[1].contour(labeled_regions, colors="red", linewidths=1)
ax[1].set_title("Detected Abnormalities")
ax[1].axis("off")

plt.tight_layout()
plt.show()
