#!/usr/bin/env python3

import json
import spacy
import re
import os
from pathlib import Path

def clean_step_text(text):
    """Clean a step by removing XML-like tags and extra whitespace"""
    # Remove XML-like tags like <citation>, <reasoning>, etc.
    text = re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def process_jsonl_file(input_path, output_path, nlp):
    """Transform a single JSONL file to coconut format"""
    processed_data = []
    
    print(f"Processing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
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
                    print(f"Warning: No 'In summary:' found in line {line_num}, skipping")
                    continue
                
                # Split reasoning trace and answer
                reasoning_trace = assistant_response[:summary_idx].strip()
                answer = assistant_response[summary_idx + len('In summary:'):].strip()
                
                # Use spacy to split reasoning trace into sentences/steps
                doc = nlp(reasoning_trace)
                raw_steps = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                
                # Clean each step
                steps = []
                for step in raw_steps:
                    cleaned_step = clean_step_text(step)
                    if cleaned_step:  # Only keep non-empty steps
                        steps.append(cleaned_step)
                
                # Only keep examples with reasonable number of steps
                if len(steps) >= 2:
                    coconut_example = {
                        "question": question,
                        "answer": answer,
                        "steps": steps
                    }
                    processed_data.append(coconut_example)
                else:
                    print(f"Warning: Only {len(steps)} steps found in line {line_num}, skipping")
                    
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Save the processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(processed_data)} examples from {input_path}")
    return len(processed_data)

def main():
    print("=== Dataset Transformation Script ===")
    print("Transforming your dataset to coconut format...")
    
    # Load spacy model
    print("Loading spacy model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Define input and output paths
    input_dir = Path("/mnt/disk/coconut/unmasked_sft_data_qwen7B")
    output_dir = Path("/mnt/disk/coconut/data")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process each file
    files_to_process = [
        ("train.jsonl", "your_dataset_train.json"),
        ("val.jsonl", "your_dataset_val.json"), 
        ("test.jsonl", "your_dataset_test.json")
    ]
    
    total_processed = 0
    for input_file, output_file in files_to_process:
        input_path = input_dir / input_file
        output_path = output_dir / output_file
        
        if input_path.exists():
            count = process_jsonl_file(input_path, output_path, nlp)
            total_processed += count
        else:
            print(f"Warning: {input_path} not found, skipping")
    
    print(f"\n=== Transformation Complete ===")
    print(f"Total examples processed: {total_processed}")
    print(f"Output files saved in: {output_dir}")
    
    # Show a sample from the training data
    train_output = output_dir / "your_dataset_train.json"
    if train_output.exists():
        with open(train_output, 'r') as f:
            data = json.load(f)
            if data:
                print(f"\n=== Sample Transformed Example ===")
                sample = data[0]
                print(f"Question (first 150 chars): {sample['question'][:150]}...")
                print(f"Answer (first 100 chars): {sample['answer'][:100]}...")
                print(f"Number of steps: {len(sample['steps'])}")
                print(f"First step: {sample['steps'][0]}")

if __name__ == "__main__":
    main()