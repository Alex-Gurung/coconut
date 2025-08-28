# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

Coconut implements **continuous latent space reasoning** for large language models. The key architectural components:

### Multi-Stage Training System
- **Training progresses in stages**: `scheduled_stage = epoch // epochs_per_stage`
- Each stage increases reasoning capacity: `k = min(max_latent_stage, scheduled_stage) * c_thought` latent tokens
- Optimizer resets between stages if `reset_optimizer: True`
- **Critical**: Don't stop training mid-stage - each stage fundamentally changes the model's reasoning capacity

### Coconut Model (`coconut.py`)
- Wraps base language model with latent reasoning tokens
- Latent tokens (`<|latent|>`) are replaced with learnable continuous embeddings during forward pass  
- Special tokens: `<|start-latent|>` and `<|end-latent|>` mark reasoning boundaries
- Supports GPT2 and Llama-family models

### Dataset Processing (`dataset.py`)
- Converts reasoning traces into discrete steps
- Dynamically adjusts latent token count based on training stage
- Expected format: `{"question": str, "answer": str, "steps": [str, ...]}`

## Common Training Commands

```bash
# Main training command
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py args/config.yaml

# Debug mode (subset of data, no wandb/saving)
# Set debug: True in config

# Resume from specific epoch
# Set resume: EPOCH_NUM in config

# Evaluation only
# Set only_eval: True and load_model_path: "path/to/checkpoint"
```

## Key Training Parameters

- **`c_thought`**: Latent tokens per reasoning step (start with 1-2, not 8+)
- **`epochs_per_stage`**: Epochs before increasing reasoning capacity (typically 3-5)  
- **`max_latent_stage`**: Maximum reasoning stages (3 for most tasks)
- **`batch_size_training`**: Per-GPU batch size (reduce for OOM: try 1 with higher grad_accum)
- **`gradient_accumulation_steps`**: Multiply by num_GPUs for effective batch size
- **`load_model_path`**: Use "None" (string) to start from base model, not empty string

## Memory Optimization for OOM

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Reduce `batch_size_training` and increase `gradient_accumulation_steps` proportionally to maintain effective batch size.

## Dataset Transformation

Transform your data to coconut format using provided scripts:
1. Data should have reasoning traces that end with "In summary:" 
2. Split reasoning before "In summary:" into steps using spacy sentence segmentation
3. Preserve structured reasoning (like `<citation>`, `<reasoning>` tags)

## Training Convergence

- **Don't stop mid-stage**: Each stage changes model architecture with more latent tokens
- Validation accuracy computed at epoch end, not during training
- Loss convergence in early stage doesn't mean overall convergence
- Complete at least one full stage before evaluating stopping

## Model Configuration Notes

- Set `save_only_improve: False` for coconut training (saves all stage checkpoints)
- Use `bf16: True` for 7B+ models for memory efficiency  
- Coconut method can skip CoT stage if base model already does reasoning well
- `reset_optimizer: True` recommended for clean stage transitions