#!/usr/bin/env python3

import json

def check_coconut_format():
    """Check existing coconut data format"""
    print("=== Coconut Expected Format ===")
    print("Each data point should have:")
    print("- question (str)")
    print("- answer (str)")  
    print("- steps (list of str)")
    
    # Check if there are existing examples
    try:
        with open('/mnt/disk/coconut/data/prosqa_train.json', 'r') as f:
            prosqa_data = json.load(f)
            
        print(f"\n=== Example from prosqa_train.json ===")
        if prosqa_data:
            example = prosqa_data[0]
            print(f"Keys: {list(example.keys())}")
            print(f"Question: {example['question'][:100]}...")
            print(f"Answer: {example['answer']}")
            print(f"Steps count: {len(example['steps'])}")
            print(f"First step: {example['steps'][0][:100]}...")
            
    except FileNotFoundError:
        print("No existing prosqa data found")
    
    print(f"\n=== Your Dataset Analysis ===")
    print("- Instructions are in user message (should become 'question')")
    print("- Answer is after 'In summary:' in assistant response")
    print("- Steps are the reasoning trace before 'In summary:', need to be split")

if __name__ == "__main__":
    check_coconut_format()