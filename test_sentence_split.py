#!/usr/bin/env python3

import json
import spacy

def test_sentence_split():
    # Load spacy model for sentence segmentation
    nlp = spacy.load("en_core_web_sm")
    
    # Load first example
    with open('/mnt/disk/coconut/unmasked_sft_data_qwen7B/train.jsonl', 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
    
    # Extract the reasoning trace (before "In summary:")
    last_msg = data['messages'][-1]['content']
    summary_idx = last_msg.find('In summary:')
    
    if summary_idx == -1:
        print("No 'In summary:' found")
        return
        
    reasoning_trace = last_msg[:summary_idx].strip()
    answer = last_msg[summary_idx + len('In summary:'):].strip()
    
    print("=== Original Reasoning Trace (first 500 chars) ===")
    print(reasoning_trace[:500])
    print("\n=== Answer ===")
    print(answer)
    
    # Use spacy to split into sentences
    doc = nlp(reasoning_trace)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    print(f"\n=== Spacy Split Results ===")
    print(f"Number of sentences: {len(sentences)}")
    print("\n=== First 5 Steps ===")
    for i, sent in enumerate(sentences[:5]):
        print(f"Step {i+1}: {sent}")
    
    # Check question extraction
    user_msg = data['messages'][1]['content']  # User message (instructions)
    print(f"\n=== Question (first 200 chars) ===")
    print(user_msg[:200])

if __name__ == "__main__":
    test_sentence_split()