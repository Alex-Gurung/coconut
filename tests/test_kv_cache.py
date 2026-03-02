#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test if past_key_values works as expected in transformers 4.43.3
model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Testing KV cache with {model_id}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cpu",  # Use CPU for testing
)

# Create test input
test_text = "The quick brown fox"
input_ids = tokenizer.encode(test_text, return_tensors="pt")
print(f"Input: {test_text}")
print(f"Input IDs shape: {input_ids.shape}")

# Test 1: Normal forward pass
print("\n=== Test 1: Normal forward pass ===")
with torch.no_grad():
    outputs1 = model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
    print(f"Logits shape: {outputs1.logits.shape}")
    print(f"Hidden states shape: {outputs1.hidden_states[-1].shape}")
    print(f"Past key values type: {type(outputs1.past_key_values)}")
    if outputs1.past_key_values:
        print(f"Number of layers: {len(outputs1.past_key_values)}")
        print(f"First layer K/V shapes: {outputs1.past_key_values[0][0].shape}, {outputs1.past_key_values[0][1].shape}")

# Test 2: Split forward pass (like Coconut does)
print("\n=== Test 2: Split forward pass ===")
seq_len = input_ids.shape[1]
split_point = seq_len // 2

print(f"Full sequence length: {seq_len}, split at: {split_point}")

with torch.no_grad():
    # First part
    outputs_part1 = model(
        input_ids=input_ids[:, :split_point], 
        output_hidden_states=True, 
        use_cache=True
    )
    print(f"Part 1 - Logits: {outputs_part1.logits.shape}, Hidden: {outputs_part1.hidden_states[-1].shape}")
    
    # Second part with cache
    if outputs_part1.past_key_values:
        try:
            outputs_part2 = model(
                input_ids=input_ids[:, split_point:],
                past_key_values=outputs_part1.past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            print(f"Part 2 - Logits: {outputs_part2.logits.shape}, Hidden: {outputs_part2.hidden_states[-1].shape}")
            print("✓ Split forward pass with past_key_values WORKS")
        except Exception as e:
            print(f"✗ Split forward pass FAILED: {e}")
    
# Test 3: Coconut-style cache slicing
print("\n=== Test 3: Coconut-style cache manipulation ===")
with torch.no_grad():
    try:
        # Get full cache first
        full_outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        kv_cache = full_outputs.past_key_values
        
        # Try Coconut's cache slicing approach
        next_compute_range = (0, split_point)
        past_key_values = [
            (
                k[:, :, : next_compute_range[0], :],  # This should be empty when next_compute_range[0] = 0
                v[:, :, : next_compute_range[0], :],
            )
            for k, v in kv_cache
        ]
        
        print(f"Sliced cache shapes: K={past_key_values[0][0].shape}, V={past_key_values[0][1].shape}")
        
        # Try to use sliced cache
        outputs_with_sliced = model(
            input_ids=input_ids[:, split_point:],
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True
        )
        print(f"✓ Coconut-style cache slicing WORKS")
        print(f"Output hidden shape: {outputs_with_sliced.hidden_states[-1].shape}")
        
    except Exception as e:
        print(f"✗ Coconut-style cache manipulation FAILED: {e}")

print("\n=== Summary ===")
print("This test checks if transformers 4.43.3 supports the KV cache manipulation that Coconut uses.")