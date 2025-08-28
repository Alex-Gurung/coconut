#!/usr/bin/env python3
"""
Coconut-style inference for extracted HuggingFace models.

This script provides Coconut reasoning capabilities for standard HF models
by detecting latent tokens and using iterative generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

class CoconutInference:
    def __init__(self, model_path):
        """
        Initialize Coconut inference with a standard HuggingFace model.
        
        Args:
            model_path: Path to HF model (local or HF Hub ID like "agurung/coconut-qwen2.5-7b")
        """
        print(f"Loading model: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Get special token IDs (if they exist)
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start-latent|>") 
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        
        # Check if this model has Coconut tokens
        self.has_coconut_tokens = (
            self.latent_id != self.tokenizer.unk_token_id and
            self.start_id != self.tokenizer.unk_token_id and 
            self.end_id != self.tokenizer.unk_token_id
        )
        
        print(f"Coconut tokens available: {self.has_coconut_tokens}")
        if self.has_coconut_tokens:
            print(f"  <|latent|>: {self.latent_id}")
            print(f"  <|start-latent|>: {self.start_id}") 
            print(f"  <|end-latent|>: {self.end_id}")
    
    def detect_reasoning_mode(self, input_ids):
        """
        Check if input contains latent tokens requiring Coconut-style processing.
        
        Args:
            input_ids: torch.Tensor of token IDs
            
        Returns:
            bool: True if reasoning mode needed
        """
        if not self.has_coconut_tokens:
            return False
            
        return (input_ids == self.latent_id).any().item()
    
    def generate_standard(self, input_ids, **kwargs):
        """
        Standard HuggingFace generation (no special processing).
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        return outputs
    
    def generate_coconut_style(self, input_ids, max_new_tokens=512, **kwargs):
        """
        Coconut-style iterative generation with latent token replacement.
        
        This mimics the Coconut forward pass logic for inference:
        1. Find latent token positions
        2. For each latent position, do forward pass and replace with hidden state
        3. Continue until all latents replaced
        4. Generate remaining tokens normally
        """
        print("Using Coconut-style reasoning generation...")
        
        # Find latent token positions
        latent_positions = (input_ids == self.latent_id).nonzero(as_tuple=True)[1].tolist()
        if not latent_positions:
            return self.generate_standard(input_ids, max_new_tokens=max_new_tokens, **kwargs)
        
        print(f"Found {len(latent_positions)} latent tokens at positions: {latent_positions}")
        
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Progressive latent replacement (like Coconut training)
        for i, latent_pos in enumerate(latent_positions):
            print(f"Replacing latent token {i+1}/{len(latent_positions)} at position {latent_pos}")
            
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                    use_cache=False,
                )
                
                # Get hidden state from position before latent token
                if latent_pos > 0:
                    hidden_state = outputs.hidden_states[-1][0, latent_pos - 1, :]
                    # Replace latent embedding with reasoning representation  
                    inputs_embeds[0, latent_pos] = hidden_state
                    print(f"  Replaced with hidden state from position {latent_pos - 1}")
                else:
                    print(f"  Warning: Latent at position 0, cannot replace")
        
        # Now generate normally with the reasoning-enhanced embeddings
        print("Generating continuation with reasoning embeddings...")
        
        # Get the processed sequence length
        current_length = inputs_embeds.shape[1]
        
        # Generate tokens one by one (simpler approach)
        generated_tokens = []
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(inputs_embeds=inputs_embeds)
                
                # Get next token
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                
                # Stop at EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token_id)
                
                # Add new token embedding to sequence
                next_token_embed = self.model.get_input_embeddings()(
                    torch.tensor([[next_token_id]], device=inputs_embeds.device)
                )
                inputs_embeds = torch.cat([inputs_embeds, next_token_embed], dim=1)
        
        # Combine original + generated tokens
        original_tokens = input_ids[0].tolist()
        all_tokens = original_tokens + generated_tokens
        
        return torch.tensor([all_tokens], device=input_ids.device)
    
    def generate(self, prompt, max_new_tokens=512, **kwargs):
        """
        Main generation method that automatically detects mode.
        
        Args:
            prompt: Text prompt (may contain <|latent|> tokens)
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional generation arguments
            
        Returns:
            str: Generated text
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)
        
        # Detect mode and generate
        if self.detect_reasoning_mode(input_ids):
            print("🧠 Reasoning mode detected")
            output_ids = self.generate_coconut_style(input_ids, max_new_tokens=max_new_tokens, **kwargs)
        else:
            print("💬 Standard generation mode")
            output_ids = self.generate_standard(input_ids, max_new_tokens=max_new_tokens, **kwargs)
        
        # Decode and return
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

def main():
    parser = argparse.ArgumentParser(description="Coconut-style inference")
    parser.add_argument("model_path", help="Path to HF model")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max new tokens")
    
    args = parser.parse_args()
    
    # Initialize inference
    coconut_inference = CoconutInference(args.model_path)
    
    # Generate
    print(f"\nPrompt: {args.prompt}\n")
    result = coconut_inference.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(f"Generated: {result}\n")

if __name__ == "__main__":
    main()