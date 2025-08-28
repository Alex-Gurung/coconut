#!/usr/bin/env python3

import json

def explore_data():
    # Load first example from training data
    with open('/mnt/disk/coconut/unmasked_sft_data_qwen7B/train.jsonl', 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
    
    print("=== Dataset Structure ===")
    print(f"Top-level keys: {list(data.keys())}")
    print(f"Number of messages: {len(data['messages'])}")
    
    print("\n=== Message Roles ===")
    for i, msg in enumerate(data['messages']):
        print(f"Message {i}: {msg['role']}")
    
    print("\n=== Last Message (Assistant Response) ===")
    last_msg = data['messages'][-1]['content']
    print(f"Length: {len(last_msg)} characters")
    
    # Find "In summary:" section
    summary_idx = last_msg.find('In summary:')
    if summary_idx != -1:
        print(f"'In summary:' found at position {summary_idx}")
        summary_part = last_msg[summary_idx:]
        print(f"Summary section length: {len(summary_part)} characters")
        print("\n=== Summary Section (first 300 chars) ===")
        print(summary_part[:300])
    else:
        print("No 'In summary:' found")
    
    # Show structure before summary
    if summary_idx != -1:
        reasoning_part = last_msg[:summary_idx]
        print(f"\n=== Reasoning Section (last 300 chars) ===")
        print(reasoning_part[-300:])

if __name__ == "__main__":
    explore_data()