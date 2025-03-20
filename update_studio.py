import os
from lightning_sdk import Studio

# Initialize Lightning Studio
studio = Studio(name="promising-olive-vdc0g", teamspace="Vision-model", user="andretiagosilva77")

# Define the local directory containing images
total_root = "images"  # Make sure this is the correct path

# Walk through the directory and upload files
for root, _, files in os.walk(total_root):
    for f in files:
        # Define remote path (relative to `total_root`)
        remote_path = os.path.join(os.path.relpath(root, total_root), f)
        
        # Upload file to remote location
        studio.upload_file(os.path.join(root, f), remote_path)

print("✅ Upload complete!")