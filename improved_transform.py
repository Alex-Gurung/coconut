#!/usr/bin/env python3

import json
import spacy
import re
from pathlib import Path

def clean_step_text(text):
    """Clean a step by removing XML-like tags and extra whitespace"""
    # Remove XML-like tags like <citation>, <reasoning>, etc.
    text = re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    # Remove extra colons at the beginning
    text = re.sub(r'^:\s*', '', text)
    
    return text.strip()

def split_reasoning_into_steps(reasoning_text):
    """Split reasoning text using multiple patterns like in your code"""
    
    # Split patterns from your code
    split_patterns = [
        "In summary:",
        "In summary,", 
        "Detailed Plan:"
    ]
    
    # First, apply the split patterns to extract just the reasoning part
    working_text = reasoning_text
    for pattern in split_patterns:
        working_text = working_text.split(pattern)[-1].strip()
    
    # Now use spacy for sentence splitting on the cleaned text
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(working_text)
    raw_steps = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Clean each step
    steps = []
    for step in raw_steps:
        cleaned_step = clean_step_text(step)
        if cleaned_step and len(cleaned_step) > 10:  # Only keep meaningful steps
            steps.append(cleaned_step)
    
    return steps

def test_improved_transform():
    """Test the improved transformation"""
    
    input_path = "/mnt/disk/coconut/unmasked_sft_data_qwen7B/train.jsonl"
    processed_data = []
    
    print("Testing improved transformation on first 3 examples...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > 3:  # Only process first 3 examples
                break
                
            try:
                data = json.loads(line.strip())
                
                # Extract question from user message (instructions)
                user_msg = data['messages'][1]['content']
                question = user_msg.strip()
                
                # Extract assistant response
                assistant_response = data['messages'][-1]['content']
                
                # Find "In summary:" section
                summary_idx = assistant_response.find('In summary:')
                if summary_idx == -1:
                    # Try alternative pattern
                    summary_idx = assistant_response.find('In summary,')
                    if summary_idx == -1:
                        print(f"Warning: No 'In summary' found in line {line_num}, skipping")
                        continue
                    summary_start = 'In summary,'
                else:
                    summary_start = 'In summary:'
                
                # Split reasoning trace and answer
                reasoning_trace = assistant_response[:summary_idx].strip()
                answer = assistant_response[summary_idx + len(summary_start):].strip()
                
                # Use improved splitting function
                steps = split_reasoning_into_steps(reasoning_trace)
                
                coconut_example = {
                    "question": question,
                    "answer": answer,
                    "steps": steps
                }
                processed_data.append(coconut_example)
                print(f"Processed example {line_num}: {len(steps)} steps")
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Save test output
    output_path = "/mnt/disk/coconut/improved_test_output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nImproved test complete! Saved {len(processed_data)} examples to {output_path}")
    
    # Show first example
    if processed_data:
        example = processed_data[0]
        print(f"\n=== Sample Improved Transformation ===")
        print(f"Question (first 100 chars): {example['question'][:100]}...")
        print(f"Answer (first 100 chars): {example['answer'][:100]}...")
        print(f"Number of steps: {len(example['steps'])}")
        print(f"First 5 steps:")
        for i, step in enumerate(example['steps'][:5]):
            print(f"  {i+1}: {step}")

if __name__ == "__main__":
    test_improved_transform()