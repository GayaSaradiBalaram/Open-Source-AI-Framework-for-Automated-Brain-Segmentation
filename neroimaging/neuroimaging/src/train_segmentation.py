import torch 
import torch.nn as nn 
import torch.optim as optim 
import nibabel as nib 
import numpy as np 
from monai.networks.nets import UNet 
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn.functional as F 

# ✅ Load the CT dataset 
dataset_path = r"C:\Users\saradhi\neroimaging\data\preprocessed_CT_AVM.nii.gz" 

try: 
    img = nib.load(dataset_path) 
    data = img.get_fdata() 
except FileNotFoundError: 
    print(" Preprocessed CT file not found! Check the path.") 
    exit() 

# ✅ Convert to PyTorch tensor (Add batch & channel dimensions) 
data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  

# ✅ Ensure input shape is divisible by 16 
def make_divisible_by_16(shape): 
    return [(dim // 16) * 16 if dim % 16 != 0 else dim for dim in shape] 

new_shape = make_divisible_by_16(data.shape[2:]) 
data = F.interpolate(data, size=new_shape, mode='trilinear', align_corners=False) 

print(f" Resized input shape: {data.shape}")  

# ✅ Generate Random Labels (Replace with real ground truth in the future) 
labels = torch.randint(0, 3, data.shape[2:], dtype=torch.long).unsqueeze(0)  # Labels: [Batch, Depth, Height, Width]

# Ensure correct label shape
labels = labels.unsqueeze(1)  # Add channel dimension for compatibility

# ✅ Define MONAI U-Net Model 
model = UNet( 
    spatial_dims=3, 
    in_channels=1, 
    out_channels=3,  # Segmentation classes: Gray Matter, White Matter, CSF 
    channels=(16, 32, 64, 128), 
    strides=(2, 2, 2), 
    num_res_units=2 
) 

# ✅ Define Loss Function & Optimizer 
loss_function = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# ✅ Prepare Dataset & DataLoader 
dataset = TensorDataset(data, labels) 
train_loader = DataLoader(dataset, batch_size=1, shuffle=True) 

# ✅ Train Model 
num_epochs = 5  # Adjust as needed 
for epoch in range(num_epochs): 
    for batch in train_loader: 
        optimizer.zero_grad() 
        output = model(batch[0])  # Predicted segmentation 
        
        # Ensure labels (batch[1]) have the correct shape 
        loss = loss_function(output, batch[1].squeeze(1).long())  # Target shape: [Batch, Depth, Height, Width] 
        
        loss.backward() 
        optimizer.step() 
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}") 

# ✅ Save Trained Model 
model_path = r"C:\Users\saradhi\neroimaging\data\monai_unet.pth" 
torch.save(model.state_dict(), model_path) 
print(f" Model training complete! Model saved at: {model_path}") 
