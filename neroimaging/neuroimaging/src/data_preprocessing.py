import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    ScaleIntensity, EnsureChannelFirst, ToTensor, Compose
)

# Load the MRI scan
input_path = r"C:\Users\saradhi\neroimaging\data\CT_AVM.nii.gz"
output_path = r"C:\Users\saradhi\neroimaging\data\preprocessed_CT_AVM.nii.gz"


try:
    mri_img = nib.load(input_path)
    mri_data = mri_img.get_fdata()
except FileNotFoundError:
    print("MRI file not found! Check the file path.")
    exit()

# Apply MONAI preprocessing
transform = Compose([
    ScaleIntensity(),  # Normalization
    EnsureChannelFirst(),  # Ensures channel-first format
    ToTensor()  # Convert to PyTorch tensor
])

# Normalize manually before saving
mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))

# Save preprocessed MRI
nib.save(nib.Nifti1Image(mri_data, affine=mri_img.affine), output_path)
print(f" Preprocessing completed! Preprocessed MRI saved at: {output_path}")

# Visualize the MRI scan
plt.imshow(mri_data[:, :, mri_data.shape[2] // 2], cmap="gray")
plt.title("Preprocessed MRI Slice")
plt.axis("off")
plt.show()
