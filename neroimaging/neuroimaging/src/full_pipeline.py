import os
import subprocess

# ✅ Define the Scripts to Run in Order
scripts = [
    "data_preprocessing.py",
    "data_augmentation.py",
    "train_segmentation.py",
    "mri_segmentation.py",
    "abnormality_detection.py",
    "statistical_analysis.py",
    "hypothesis_testing.py",
    "heatmap_visualization.py",
]

# ✅ Execute Each Script Sequentially
scripts_dir = r"C:\Users\saradhi\neroimaging\neuroimaging\src"

for script in scripts:
    script_path = os.path.join(scripts_dir, script)
    
    if os.path.exists(script_path):
        print(f"\n Running {script}...")
        subprocess.run(["python", script_path], check=True)
    else:
        print(f"Warning: {script} not found. Skipping...")

print("\n**Full MRI Segmentation Pipeline Execution Completed!** Report generated.")
