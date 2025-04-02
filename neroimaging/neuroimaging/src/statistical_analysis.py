import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the file paths
segmented_path = r"C:/Users/saradhi/neroimaging/data/segmented_mri.nii.gz"
output_csv_path = r"C:/Users/saradhi/neroimaging/data/statistical_results.csv"
output_histogram_path = r"C:/Users/saradhi/neroimaging/data/mri_intensity_histogram.png"

try:
    # Load the segmented MRI
    img = nib.load(segmented_path)
    data = img.get_fdata()
    print("Segmented MRI loaded successfully!")
except FileNotFoundError:
    print("Segmented MRI file not found! Check the file path.")
    exit()

# Compute statistics
voxel_values = data.flatten()
mean_intensity = np.mean(voxel_values)
std_intensity = np.std(voxel_values)
max_intensity = np.max(voxel_values)
min_intensity = np.min(voxel_values)

# Save results to CSV
stats_df = pd.DataFrame({
    "Metric": ["Mean Intensity", "Standard Deviation", "Max Intensity", "Min Intensity"],
    "Value": [mean_intensity, std_intensity, max_intensity, min_intensity]
})
stats_df.to_csv(output_csv_path, index=False)

# Plot histogram of intensity values
plt.figure(figsize=(8, 5))
plt.hist(voxel_values, bins=50, color='blue', alpha=0.7)
plt.title("MRI Intensity Distribution")
plt.xlabel("Intensity Value")
plt.ylabel("Frequency")
plt.grid(True)

# Save and show the plot
plt.savefig(output_histogram_path)
plt.show()

print(f"Statistical analysis completed! Results saved in '{output_csv_path}'.")
print(f"Histogram saved as '{output_histogram_path}'.")
