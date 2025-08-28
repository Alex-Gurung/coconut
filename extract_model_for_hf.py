#!/usr/bin/env python3
"""
Extract the base model from Coconut checkpoint and save it in Hugging Face format
for uploading to Hugging Face Hub.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut

def extract_coconut_model(checkpoint_path, output_dir, model_id="Qwen/Qwen2.5-7B-Instruct"):
    """
    Load a Coconut checkpoint and extract the base model in HF format.
    
    Args:
        checkpoint_path: Path to the Coconut checkpoint (e.g., checkpoints/qwen-coconut/checkpoint_2)
        output_dir: Directory to save the extracted model
        model_id: Original model ID used for tokenizer and config
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load original tokenizer for config
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load base model architecture
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # The checkpoint contains Coconut wrapper state_dict
    # We need to extract only the base_causallm weights
    base_model_state = {}
    
    for key, value in checkpoint.items():
        if key.startswith("base_causallm."):
            # Remove the "base_causallm." prefix to get the original model keys
            new_key = key[len("base_causallm."):]
            base_model_state[new_key] = value
        elif not key.startswith("embedding"):
            # Include other keys that might be part of the base model
            # but skip Coconut-specific embedding modifications
            base_model_state[key] = value
    
    print(f"Found {len(base_model_state)} base model parameters")
    print(f"Original model has {len(base_model.state_dict())} parameters")
    
    # Handle embedding size mismatch due to added special tokens
    if "model.embed_tokens.weight" in base_model_state:
        checkpoint_embed_size = base_model_state["model.embed_tokens.weight"].shape[0]
        model_embed_size = base_model.model.embed_tokens.weight.shape[0]
        
        if checkpoint_embed_size != model_embed_size:
            print(f"Embedding size mismatch: checkpoint has {checkpoint_embed_size}, model has {model_embed_size}")
            
            if checkpoint_embed_size < model_embed_size:
                print("Checkpoint has fewer embeddings - this is expected since Coconut adds special tokens during training")
                print("Using checkpoint embeddings and truncating base model to match...")
                # The checkpoint represents the trained state, so we should resize the base model to match
                base_model.resize_token_embeddings(checkpoint_embed_size)
            else:
                print("Truncating checkpoint embeddings to match base model...")
                # Truncate to original vocabulary size
                base_model_state["model.embed_tokens.weight"] = base_model_state["model.embed_tokens.weight"][:model_embed_size]
                if "lm_head.weight" in base_model_state:
                    base_model_state["lm_head.weight"] = base_model_state["lm_head.weight"][:model_embed_size]
    
    # Load the extracted weights
    missing_keys, unexpected_keys = base_model.load_state_dict(base_model_state, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
    
    print(f"Saving model to: {output_dir}")
    
    # Save in HF format
    base_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    # Create a README with model info
    readme_content = f"""# Coconut-Enhanced Qwen2.5-7B-Instruct

This model was trained using the [Coconut](https://github.com/meta-research/coconut) method for continuous latent space reasoning.

## Base Model
- **Base**: {model_id}
- **Method**: Coconut (Continuous Latent Space Reasoning)
- **Training**: Custom reasoning dataset with spacy-segmented reasoning steps

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("agurung/coconut-qwen2.5-7b")
tokenizer = AutoTokenizer.from_pretrained("agurung/coconut-qwen2.5-7b")

# Use like any other Qwen model
inputs = tokenizer("Your question here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details
- **Dataset**: Reasoning traces with "In summary:" endings
- **Method**: Progressive latent token replacement during training
- **Latent Tokens**: 2 per reasoning step (c_thought)
- **Max Reasoning Stages**: 2 (max_latent_stage)

Extracted from checkpoint: `{checkpoint_path.split('/')[-1]}`
"""
    
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    print("✓ Model extracted and saved successfully!")
    print(f"✓ Ready to upload to Hugging Face Hub from: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", help="Path to Coconut checkpoint")
    parser.add_argument("output_dir", help="Output directory for HF model")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct", help="Base model ID")
    
    args = parser.parse_args()
    extract_coconut_model(args.checkpoint_path, args.output_dir, args.model_id)