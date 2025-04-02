import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.metrics import DiceMetric

# ✅ Load trained model
model_path = r"C:\Users\saradhi\neroimaging\data\monai_unet.pth"
try:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # Segmentation classes: Gray Matter, White Matter, CSF
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found! Train the model first.")
    exit()

# ✅ Load test MRI scan and ground truth segmentation
test_image_path = r"C:\Users\saradhi\neroimaging\data\test_mri.nii.gz"
test_label_path = r"C:\Users\saradhi\neroimaging\data\test_labels.nii.gz"

try:
    test_img = nib.load(test_image_path)
    test_label = nib.load(test_label_path)
    test_data = test_img.get_fdata()
    test_label_data = test_label.get_fdata()
    print(f"Test MRI and ground truth loaded! Shape: {test_data.shape}")
except FileNotFoundError:
    print("Test MRI or ground truth file not found! Check file paths.")
    exit()

# ✅ Convert MRI and labels to tensors
test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [B, C, D, H, W]
test_label_data = torch.tensor(test_label_data, dtype=torch.long)

# ✅ Ensure labels match the number of classes
num_classes = 3
test_label_data = torch.clamp(test_label_data, min=0, max=num_classes - 1)

# ✅ Check shape before processing
print(f" Shape before processing: {test_data.shape}")

# ✅ Expand depth if MRI has only 3 slices
if test_data.shape[2] == 3:
    print("MRI scan has only 3 slices. Expanding depth to match full volume.")
    test_data = F.interpolate(test_data, size=(256, 240, 144), mode="trilinear", align_corners=False)
print(f" Shape after depth expansion: {test_data.shape}")

# ✅ Ensure correct shape before passing to model
print(f" Shape before passing to model: {test_data.shape}")

# ✅ Predict segmentation
with torch.no_grad():
    predicted_segmentation = model(test_data)  # Shape: [1, 3, D, H, W]
    predicted_segmentation = predicted_segmentation.argmax(dim=1).squeeze(0).cpu().numpy()  # Convert to numpy

# 🔍 Debugging: Check unique values
print(f" Unique values in predicted segmentation: {np.unique(predicted_segmentation)}")
print(f" Unique values in ground truth labels: {np.unique(test_label_data)}")

# ✅ Compute Dice Score
dice_metric = DiceMetric(include_background=False, reduction="mean")
dice_score = dice_metric(
    torch.tensor(predicted_segmentation, dtype=torch.float32).unsqueeze(0),
    test_label_data.unsqueeze(0).float()
)
print(f" Dice Score: {dice_score.mean().item()}")

# ✅ Display Segmented MRI Slice
middle_slice = predicted_segmentation.shape[0] // 2
plt.imshow(predicted_segmentation[middle_slice], cmap="jet")
plt.title(f"Segmented MRI - Slice {middle_slice}")
plt.axis("off")
plt.show()
