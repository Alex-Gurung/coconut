#!/usr/bin/env python3

import json
import spacy
import re
import os
from pathlib import Path
from tqdm import tqdm

def clean_step_text(text):
    """Clean a step by removing XML-like tags and extra whitespace"""
    # Remove XML-like tags like <citation>, <reasoning>, etc.
    text = re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def count_lines(filepath):
    """Count lines in a file for progress tracking"""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

def process_jsonl_file(input_path, output_path, nlp):
    """Transform a single JSONL file to coconut format"""
    processed_data = []
    
    print(f"Processing {input_path}...")
    
    # Count total lines for progress bar
    total_lines = count_lines(input_path)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc=f"Processing {input_path.name}"), 1):
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
                    continue  # Skip examples without summary
                
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
                    
            except json.JSONDecodeError:
                continue  # Skip invalid JSON lines
            except Exception as e:
                continue  # Skip problematic lines
    
    # Save the processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(processed_data)} examples to {output_path}")
    return len(processed_data)

def main():
    print("=== Full Dataset Transformation ===")
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
    results = {}
    
    for input_file, output_file in files_to_process:
        input_path = input_dir / input_file
        output_path = output_dir / output_file
        
        if input_path.exists():
            count = process_jsonl_file(input_path, output_path, nlp)
            total_processed += count
            results[input_file] = count
        else:
            print(f"Warning: {input_path} not found, skipping")
            results[input_file] = 0
    
    print(f"\n=== Transformation Complete ===")
    for filename, count in results.items():
        print(f"{filename}: {count} examples")
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
                print(f"Question length: {len(sample['question'])} chars")
                print(f"Answer length: {len(sample['answer'])} chars")
                print(f"Number of steps: {len(sample['steps'])}")
                print(f"First step: {sample['steps'][0][:100]}...")
                
    print(f"\n=== Next Steps ===")
    print(f"Your transformed dataset is ready! You can now use it with coconut by:")
    print(f"1. Updating your config yaml to point to the new dataset files")
    print(f"2. Running: torchrun --nnodes 1 --nproc_per_node N_GPUS run.py your_config.yaml")

if __name__ == "__main__":
    main()