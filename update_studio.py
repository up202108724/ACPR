import os
from lightning_sdk import Studio

# Initialize Lightning Studio
studio = Studio(name="promising-olive-vdc0g", teamspace="Vision-model", user="andretiagosilva77")

# Define the local file you want to upload
local_file = "text2img.zip"  # Update this with the correct file path

# Define remote path (optional: specify a different filename if needed)
remote_path = os.path.basename(local_file)  # This keeps the same name in the remote location

# Upload the file
studio.upload_file(local_file, remote_path)

print("✅ Upload complete!")
