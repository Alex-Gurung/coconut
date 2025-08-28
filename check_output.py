#!/usr/bin/env python3

import json

def check_output():
    """Check the test output format"""
    with open('/mnt/disk/coconut/test_output.json', 'r') as f:
        data = json.load(f)
    
    print("=== Output Format Check ===")
    print(f"Number of examples: {len(data)}")
    
    for i, example in enumerate(data):
        print(f"\n--- Example {i+1} ---")
        print(f"Keys: {list(example.keys())}")
        print(f"Question length: {len(example['question'])} chars")
        print(f"Answer length: {len(example['answer'])} chars") 
        print(f"Number of steps: {len(example['steps'])}")
        print(f"Question start: {example['question'][:80]}...")
        print(f"Answer start: {example['answer'][:80]}...")
        print("Steps preview:")
        for j, step in enumerate(example['steps'][:3]):
            print(f"  Step {j+1}: {step[:60]}...")

if __name__ == "__main__":
    check_output()