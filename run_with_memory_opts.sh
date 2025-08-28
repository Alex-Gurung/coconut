#!/bin/bash

# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Run training with memory optimizations
echo "=== Running Coconut Training with Memory Optimizations ==="
echo "Batch size per GPU: 1"
echo "Gradient accumulation: 8"
echo "Effective batch size: 1 * 4 GPUs * 8 grad_accum = 32"
echo "Memory optimization: expandable_segments=True"
echo ""

torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_coconut.yaml