#!/usr/bin/env python3
"""
Upload the extracted Coconut model to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
import argparse

def upload_model_to_hf(model_dir, repo_name, username="agurung", private=False, revision=None):
    """
    Upload model directory to Hugging Face Hub
    
    Args:
        model_dir: Directory containing the model files
        repo_name: Name of the repository (e.g., "coconut-qwen2.5-7b")
        username: HF username
        private: Whether to make the repo private
        revision: Branch/revision name (e.g., "checkpoint_3", "epoch_2")
    """
    
    repo_id = f"{username}/{repo_name}"
    
    if revision:
        print(f"Uploading model to: https://huggingface.co/{repo_id}/tree/{revision}")
        print(f"Revision: {revision}")
    else:
        print(f"Uploading model to: https://huggingface.co/{repo_id}")
        
    print(f"Model directory: {model_dir}")
    
    # Initialize HF API
    api = HfApi()
    
    # Check if user is logged in
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face Hub!")
        print("Run: huggingface-cli login")
        return False
    
    # Create repository
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✓ Repository created: {repo_id}")
    except Exception as e:
        print(f"Repository creation error: {e}")
        return False
    
    # Create branch/revision if specified
    if revision:
        try:
            api.create_branch(repo_id, branch=revision, exist_ok=True)
            print(f"✓ Branch/revision created: {revision}")
        except Exception as e:
            print(f"Branch creation warning: {e}")
    
    # Upload all files in the model directory
    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            revision=revision
        )
        print(f"✓ Upload completed!")
        if revision:
            print(f"✓ Model available at: https://huggingface.co/{repo_id}/tree/{revision}")
        else:
            print(f"✓ Model available at: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Directory containing the extracted model")
    parser.add_argument("--repo_name", default="coconut-qwen2.5-7b", help="HF repo name")
    parser.add_argument("--username", default="agurung", help="HF username")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--revision", help="Branch/revision name (e.g., checkpoint_3)")
    
    args = parser.parse_args()
    
    success = upload_model_to_hf(args.model_dir, args.repo_name, args.username, args.private, args.revision)
    if success:
        print("\n🎉 Upload successful! Your model is now available on Hugging Face Hub.")
    else:
        print("\n❌ Upload failed. Please check the errors above.")