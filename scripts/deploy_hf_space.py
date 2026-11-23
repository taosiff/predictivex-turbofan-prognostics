"""
PredictiveX Automated Hugging Face Space Deployment Script.
Uses huggingface_hub to create the Space and upload the project files directly.
"""

import sys
from pathlib import Path
import huggingface_hub

api = huggingface_hub.HfApi()

try:
    user_info = api.whoami()
    username = user_info["name"]
    print(f"[OK] Authenticated as Hugging Face user: {username}")
except Exception as e:
    print(f"[ERROR] Could not authenticate with Hugging Face: {e}")
    sys.exit(1)

space_name = "predictivex-turbofan-prognostics"
repo_id = f"{username}/{space_name}"

print(f"\n[1/3] Initializing Hugging Face Space: {repo_id} (SDK: Gradio)...")
try:
    space_url = api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True
    )
    print(f"[SUCCESS] Space initialized at: {space_url}")
except Exception as e:
    print(f"[INFO] Space exists or initialized: {e}")

print("\n[2/3] Uploading local project repository to Space...")
ignore_list = [
    ".git",
    ".git/*",
    ".git/**/*",
    "catboost_info",
    "catboost_info/*",
    ".pytest_cache",
    ".pytest_cache/*",
    "__pycache__",
    "**/__pycache__/*",
    "*.pyc"
]

try:
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_list
    )
    print("\n[3/3] Upload complete!")
    print(f"\n=======================================================")
    print(f"SPACE DEPLOYED SUCCESSFULLY!")
    print(f"Live Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"=======================================================")
except Exception as e:
    print(f"[ERROR] Failed to upload folder: {e}")
    sys.exit(1)
