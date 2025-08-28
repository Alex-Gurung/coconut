#!/usr/bin/env python3

import json
import spacy
import re

def clean_step_text(text):
    """Clean a step by removing XML-like tags and extra whitespace"""
    # Remove XML-like tags like <citation>, <reasoning>, etc.
    text = re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def test_transform():
    """Test transformation on first few examples"""
    print("Loading spacy model...")
    nlp = spacy.load("en_core_web_sm")
    
    input_path = "/mnt/disk/coconut/unmasked_sft_data_qwen7B/train.jsonl"
    processed_data = []
    
    print("Processing first 3 examples...")
    
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
    output_path = "/mnt/disk/coconut/test_output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTest complete! Saved {len(processed_data)} examples to {output_path}")
    
    # Show first example
    if processed_data:
        example = processed_data[0]
        print(f"\n=== Sample Transformed Example ===")
        print(f"Question (first 100 chars): {example['question'][:100]}...")
        print(f"Answer (first 100 chars): {example['answer'][:100]}...")
        print(f"Number of steps: {len(example['steps'])}")
        print(f"First 3 steps:")
        for i, step in enumerate(example['steps'][:3]):
            print(f"  {i+1}: {step}")

if __name__ == "__main__":
    test_transform()