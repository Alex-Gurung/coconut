#!/usr/bin/env python3

import json
import spacy

def debug_formatting():
    """Debug where the extra formatting comes from"""
    
    with open('/mnt/disk/coconut/unmasked_sft_data_qwen7B/train.jsonl', 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
    
    # Extract the reasoning trace
    assistant_response = data['messages'][-1]['content']
    summary_idx = assistant_response.find('In summary:')
    reasoning_trace = assistant_response[:summary_idx].strip()
    
    print("=== Raw Reasoning Trace (first 1000 chars) ===")
    print(repr(reasoning_trace[:1000]))
    
    print("\n=== Spacy Sentence Split Debug ===")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(reasoning_trace)
    
    for i, sent in enumerate(doc.sents):
        if i >= 5:  # Only show first 5
            break
        print(f"Sentence {i+1}: {repr(sent.text)}")
        print(f"  Clean version: {repr(sent.text.strip())}")
        
    print("\n=== Looking for XML-like patterns ===")
    import re
    xml_patterns = re.findall(r'<[^>]+>', reasoning_trace[:1000])
    print(f"Found XML patterns: {xml_patterns}")
    
    citation_patterns = re.findall(r'<citation>[^<]*</citation>', reasoning_trace[:1000])
    print(f"Found citation patterns: {citation_patterns}")
    
    reasoning_patterns = re.findall(r'<reasoning>[^<]*</reasoning>', reasoning_trace[:1000])
    print(f"Found reasoning patterns: {reasoning_patterns}")

if __name__ == "__main__":
    debug_formatting()