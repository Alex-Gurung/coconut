#!/usr/bin/env python3

import json
from pathlib import Path

def check_transformation():
    """Check if the transformation worked correctly"""
    
    output_dir = Path("/mnt/disk/coconut/ncp_data")
    
    files_to_check = [
        "your_dataset_train.json",
        "your_dataset_val.json", 
        "your_dataset_test.json"
    ]
    
    print("=== Transformation Results Check ===")
    
    for filename in files_to_check:
        filepath = output_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print(f"\n{filename}:")
            print(f"  ✓ File exists")
            print(f"  ✓ Examples: {len(data)}")
            
            if data:
                sample = data[0]
                required_keys = ['question', 'answer', 'steps']
                
                for key in required_keys:
                    if key in sample:
                        print(f"  ✓ Has '{key}' field")
                    else:
                        print(f"  ✗ Missing '{key}' field")
                
                print(f"  ✓ Question length: {len(sample['question'])} chars")
                print(f"  ✓ Answer length: {len(sample['answer'])} chars")
                print(f"  ✓ Number of steps: {len(sample['steps'])}")
                print(f"  Sample step: {sample['steps'][0][:80]}...")
        else:
            print(f"\n{filename}: ✗ File not found")
    
    # Check a few more samples to verify consistency
    train_file = output_dir / "your_dataset_train.json"
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        print(f"\n=== Sample Data Validation ===")
        print(f"Checking first 5 training examples...")
        
        for i in range(min(5, len(train_data))):
            sample = train_data[i]
            has_xml = any('<citation>' in step or '<reasoning>' in step for step in sample['steps'])
            print(f"  Example {i+1}: {len(sample['steps'])} steps, XML preserved: {has_xml}")

if __name__ == "__main__":
    check_transformation()