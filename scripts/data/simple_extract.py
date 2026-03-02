#!/usr/bin/env python3
"""
Simple extraction: Load Coconut checkpoint and save just the base model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def simple_extract(checkpoint_path, output_dir):
    """
    Simple approach: Load checkpoint, extract base_causallm, save it
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load tokenizer (use original Qwen)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Create fresh base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
    )
    
    # Extract only base_causallm weights
    base_weights = {}
    for key, value in checkpoint.items():
        if key.startswith("base_causallm."):
            new_key = key.replace("base_causallm.", "")
            base_weights[new_key] = value
    
    print(f"Extracted {len(base_weights)} base model weights")
    
    # Load weights (let it handle any size mismatches)
    result = base_model.load_state_dict(base_weights, strict=False)
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")
    
    # Save
    print(f"Saving to: {output_dir}")
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✓ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint path")  
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()
    
    simple_extract(args.checkpoint, args.output)