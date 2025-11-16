#!/usr/bin/env python
"""
Run evaluation N times with different seeds for a given checkpoint or HF model.

Usage:
    python run_eval_n_times.py <checkpoint_or_model_id> <num_runs> [--config-base config.yaml] [--name run_name]

Example:
    python run_eval_n_times.py checkpoints/qwen-coconut-ff-v2/checkpoint_13 5
    python run_eval_n_times.py Qwen/Qwen2.5-7B-Instruct 3 --config-base args/qwen_coconut_ff_v1_eval.yaml
"""

import argparse
import yaml
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run evaluation N times with different seeds")
    parser.add_argument("checkpoint_or_model", help="Path to checkpoint or HuggingFace model ID")
    parser.add_argument("num_runs", type=int, help="Number of evaluation runs")
    parser.add_argument("--config-base", default="args/qwen_coconut_ff_v1_eval.yaml",
                        help="Base config file to use as template")
    parser.add_argument("--name", default=None, help="Name for the eval run (will be appended with _v0, _v1, etc)")
    parser.add_argument("--val-path", default=None, help="Path to validation/test data")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs to use (for torchrun)")

    args = parser.parse_args()

    # Load base config
    if not os.path.exists(args.config_base):
        print(f"Error: Config file {args.config_base} not found")
        sys.exit(1)

    with open(args.config_base, 'r') as f:
        base_config = yaml.safe_load(f)

    # Set common eval settings
    base_config['only_eval'] = True
    base_config['load_model_path'] = args.checkpoint_or_model

    if args.val_path:
        base_config['val_path'] = args.val_path

    # Determine run name
    if args.name:
        run_name_base = args.name
    else:
        # Use checkpoint name or model name
        if '/' in args.checkpoint_or_model:
            run_name_base = args.checkpoint_or_model.split('/')[-1]
        else:
            run_name_base = args.checkpoint_or_model

    print(f"Starting {args.num_runs} evaluation runs for {args.checkpoint_or_model}")
    print(f"Base config: {args.config_base}")
    print(f"Run name base: {run_name_base}")
    print()

    for run_idx in range(args.num_runs):
        # Create config for this run
        config = base_config.copy()

        # Use different seed for each run
        seed = 42 + run_idx * 1000  # 42, 1042, 2042, etc.
        config['seed'] = seed

        # Set name to indicate which run this is
        config['name'] = f"{run_name_base}_eval_v{run_idx}"

        # Create temp config file
        config_file = f"/mnt/disk/coconut/tmp/eval_run_{run_idx}_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        print(f"[Run {run_idx+1}/{args.num_runs}] Seed: {seed}, Name: {config['name']}")

        # Run evaluation
        cmd = [
            "torchrun",
            "--nnodes", "1",
            "--nproc_per_node", str(args.num_gpus),
            "run.py",
            config_file
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd="/mnt/disk/coconut")

        if result.returncode != 0:
            print(f"  Error: Run {run_idx} failed with return code {result.returncode}")
            sys.exit(1)

        print(f"  ✓ Run {run_idx} completed")

        # Clean up temp config
        os.remove(config_file)

    print()
    print(f"✓ All {args.num_runs} evaluation runs completed!")
    print(f"Outputs saved to: {base_config.get('save_path', 'checkpoints')}/{run_name_base}_eval_v*")
    print()
    print("To combine the results, run:")
    print(f"  python combine_evals.py checkpoints/{run_name_base}_eval_v* combined_eval.json")

if __name__ == "__main__":
    main()
