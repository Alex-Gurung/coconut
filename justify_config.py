#!/usr/bin/env python3

import yaml
from pathlib import Path

def analyze_existing_configs():
    """Look at existing configs to understand typical values"""
    
    config_dir = Path("/mnt/disk/coconut/args")
    config_files = list(config_dir.glob("*.yaml"))
    
    print("=== Analyzing Existing Configs ===")
    
    for config_file in config_files:
        print(f"\n{config_file.name}:")
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            key_fields = [
                'c_thought', 'epochs_per_stage', 'max_latent_stage', 
                'batch_size_training', 'gradient_accumulation_steps',
                'lr', 'model_id'
            ]
            
            for field in key_fields:
                if field in config:
                    print(f"  {field}: {config[field]}")
                    
        except Exception as e:
            print(f"  Error reading config: {e}")

def justify_config_choices():
    """Provide justification for each config choice"""
    
    print("\n=== Config Field Justifications ===")
    
    justifications = {
        "c_thought: 8": [
            "🤔 QUESTIONABLE - This was copied from examples without justification",
            "📚 GSM8K example uses 8, ProntoQA uses 8", 
            "💭 Should be based on complexity of your reasoning steps",
            "📏 Your steps vary from 5-34, so maybe start smaller (4-6)?"
        ],
        
        "epochs_per_stage: 3": [
            "📚 Matches GSM8K config (3 epochs per stage)",
            "⚡ Conservative - prevents overfitting early",
            "🔄 Can increase if convergence is slow"
        ],
        
        "max_latent_stage: 3": [
            "📚 GSM8K uses 3, ProntoQA uses 2", 
            "🎯 3 stages = stage 0 (baseline) + 2 coconut stages",
            "📈 Gradual increase in reasoning complexity"
        ],
        
        "batch_size_training: 2": [
            "💾 Conservative for 7B model memory usage",
            "🔢 With grad_accum=4 and N_GPUs, effective batch = 2*N*4",
            "⚖️ Balance between memory and training stability"
        ],
        
        "gradient_accumulation_steps: 4": [
            "📚 Matches existing configs (GSM8K uses 8, but we're more conservative)",
            "🎯 Achieves reasonable effective batch size",
            "💾 Helps when GPU memory limits batch size"
        ],
        
        "lr: 1e-5": [
            "🤔 QUESTIONABLE - This is quite conservative",
            "📚 GSM8K uses 5e-6, ProntoQA uses 2e-5",
            "🔍 Should probably use 2e-5 like ProntoQA",
            "⚡ Fine-tuning usually needs higher LR than this"
        ],
        
        "model_id: Qwen/Qwen2.5-7B-Instruct": [
            "✅ CORRECT - This is your actual base model",
            "🎯 Matches the model that generated your reasoning traces",
            "🚀 Skip CoT since this model already reasons well"
        ]
    }
    
    for field, reasons in justifications.items():
        print(f"\n{field}:")
        for reason in reasons:
            print(f"  {reason}")

if __name__ == "__main__":
    analyze_existing_configs()
    justify_config_choices()