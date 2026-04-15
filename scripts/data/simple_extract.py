#!/usr/bin/env python3
"""
Compatibility wrapper around the faithful Coconut HF exporter.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.extract_model_for_hf import export_coconut_model


def simple_extract(checkpoint_path, output_dir, model_id="Qwen/Qwen2.5-7B-Instruct"):
    return export_coconut_model(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        model_id=model_id,
        include_latent_checkpoint=True,
        safe_serialization=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint path")
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model id used to reconstruct the architecture and tokenizer",
    )
    args = parser.parse_args()

    simple_extract(args.checkpoint, args.output, model_id=args.model_id)
