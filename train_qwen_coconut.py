#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("=== Checking Requirements ===")
    
    # Check if config file exists
    config_path = Path("args/qwen_coconut.yaml")
    if not config_path.exists():
        print("✗ Config file not found")
        return False
    print(f"✓ Config file: {config_path}")
    
    # Check if dataset files exist
    data_files = [
        "ncp_data/your_dataset_train.json",
        "ncp_data/your_dataset_val.json"
    ]
    
    for data_file in data_files:
        if not Path(data_file).exists():
            print(f"✗ Dataset file not found: {data_file}")
            return False
        print(f"✓ Dataset: {data_file}")
    
    # Check if main training script exists
    if not Path("run.py").exists():
        print("✗ run.py not found")
        return False
    print("✓ Training script: run.py")
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("\n=== Setting up Directories ===")
    
    checkpoint_dir = Path("/mnt/disk/coconut/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"✓ Checkpoint directory: {checkpoint_dir}")

def get_gpu_count():
    """Get number of available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_count = int(result.stdout.strip().split('\n')[0])
            print(f"✓ Detected {gpu_count} GPU(s)")
            return gpu_count
    except:
        pass
    
    print("⚠ Could not detect GPUs, defaulting to 1")
    return 1

def create_training_command(num_gpus):
    """Create the torchrun command"""
    config_path = "args/qwen_coconut.yaml"
    
    # Base torchrun command
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", str(num_gpus),
        "run.py",
        config_path
    ]
    
    return cmd

def main():
    print("=== Qwen Coconut Training Setup ===")
    
    # Check if we're in the right directory
    if not Path("coconut.py").exists():
        print("✗ Please run this script from the coconut directory")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("\n✗ Requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Get GPU count
    num_gpus = get_gpu_count()
    
    # Create command
    cmd = create_training_command(num_gpus)
    
    print(f"\n=== Training Configuration ===")
    print(f"Model: Qwen/Qwen2.5-7B-Instruct")
    print(f"Method: Coconut (skipping CoT stage)")
    print(f"Training examples: ~10,677")
    print(f"Validation examples: ~1,541") 
    print(f"GPUs: {num_gpus}")
    print(f"Effective batch size: {2 * num_gpus * 4} (batch_size * gpus * grad_accum)")
    
    print(f"\n=== Training Command ===")
    print(" ".join(cmd))
    
    # Ask for confirmation
    response = input("\nStart training? [y/N]: ").strip().lower()
    if response in ['y', 'yes']:
        print("\n=== Starting Training ===")
        try:
            # Run the training command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed with exit code {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")
            sys.exit(0)
    else:
        print("Training cancelled.")
        
        print(f"\n=== Manual Command ===")
        print("To start training manually, run:")
        print(" ".join(cmd))

if __name__ == "__main__":
    main()