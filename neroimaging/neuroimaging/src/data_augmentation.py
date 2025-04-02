import nibabel as nib 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
from monai.transforms import ( 
    LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, 
    RandAffine, RandAdjustContrast, Compose 
) 

# ✅ Define file paths 
input_mri_path = r"C:\Users\saradhi\neroimaging\data\CT_AVM.nii.gz" 
augmented_mri_path = r"C:\Users\saradhi\neroimaging\data\ augmented_CT_AVM.nii.gz" 

# ✅ Define Data Augmentation Pipeline 
augment_transforms = Compose([ 
    LoadImage(image_only=True), 
    EnsureChannelFirst(), 
    ScaleIntensity(), 
    Resize(spatial_size=(256, 240, 144), mode="trilinear"), 
    RandAffine(prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)), 
    RandAdjustContrast(prob=0.5, gamma=(0.7, 1.3)) 
]) 

# Apply Augmentation 
try: 
    augmented_data = augment_transforms(input_mri_path) 
    print(f" MRI Augmentation Completed! Shape: {augmented_data.shape}") 

except FileNotFoundError: 
    print(" MRI file not found! Ensure `mri_scan.nii.gz` exists.") 
    exit() 

# Save Augmented MRI 
augmented_data_np = augmented_data.squeeze(0).numpy() 
augmented_img = nib.Nifti1Image(augmented_data_np, affine=np.eye(4)) 
nib.save(augmented_img, augmented_mri_path) 
print(f"Augmented MRI saved at: {augmented_mri_path}") 

#  Show a Sample Slice 
middle_slice = augmented_data_np.shape[2] // 2 
plt.imshow(augmented_data_np[:, :, middle_slice], cmap="gray") 
plt.title(f"Augmented MRI - Slice {middle_slice}") 
plt.axis("off") 
plt.show()
