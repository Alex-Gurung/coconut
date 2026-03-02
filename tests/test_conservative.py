#!/usr/bin/env python3

import json
import spacy
import re
from pathlib import Path

def minimal_clean_step(text):
    """Very conservative cleaning - only remove obvious artifacts"""
    # Just clean up whitespace and newlines
    text = ' '.join(text.split())
    return text.strip()

def test_conservative():
    """Test conservative transformation on first example"""
    
    nlp = spacy.load("en_core_web_sm")
    
    with open('/mnt/disk/coconut/unmasked_sft_data_qwen7B/train.jsonl', 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
    
    # Extract components
    user_msg = data['messages'][1]['content']
    question = user_msg.strip()
    
    assistant_response = data['messages'][-1]['content']
    summary_idx = assistant_response.find('In summary:')
    reasoning_trace = assistant_response[:summary_idx].strip()
    answer = assistant_response[summary_idx + len('In summary:'):].strip()
    
    # Use spacy to split
    doc = nlp(reasoning_trace)
    raw_steps = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Minimal cleaning
    steps = []
    for step in raw_steps:
        cleaned_step = minimal_clean_step(step)
        if cleaned_step and len(cleaned_step) > 5:
            steps.append(cleaned_step)
    
    print("=== Conservative Transformation Test ===")
    print(f"Number of steps: {len(steps)}")
    print("\nFirst 5 steps (with XML tags preserved):")
    for i, step in enumerate(steps[:5]):
        print(f"  {i+1}: {step}")
    
    # Save test output
    output_dir = Path("/mnt/disk/coconut/ncp_data")
    output_dir.mkdir(exist_ok=True)
    
    coconut_example = {
        "question": question,
        "answer": answer,
        "steps": steps
    }
    
    with open(output_dir / "conservative_test.json", 'w') as f:
        json.dump([coconut_example], f, indent=2, ensure_ascii=False)
    
    print(f"\nTest saved to {output_dir / 'conservative_test.json'}")

if __name__ == "__main__":
    test_conservative()