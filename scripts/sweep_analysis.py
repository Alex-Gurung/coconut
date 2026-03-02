#!/usr/bin/env python
"""
Analyze hyperparameter sweep results to find the best config and checkpoint.

Reads eval_outputs.json from sweep run directories and reports:
- Per-config best epoch and accuracy
- Overall best config + epoch
- Per-class accuracy (TPR/TNR) if applicable

Usage:
    python scripts/sweep_analysis.py /mnt/disk/coconut/checkpoints
    python scripts/sweep_analysis.py /mnt/disk/coconut/checkpoints --configs sweep-s3-c1 sweep-s10-c2
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


SWEEP_NAMES = ["sweep-s3-c1", "sweep-s3-c2", "sweep-s10-c1", "sweep-s10-c2"]


def parse_eval_dir_name(dirname):
    """Extract config name and epoch from eval directory name.

    Expected format: {config_name}-eval-ckpt_{epoch:03d}
    e.g. sweep-s3-c1-eval-ckpt_008
    """
    m = re.match(r"^(.+)-eval-ckpt_(\d+)$", dirname)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def compute_per_class_accuracy(outputs):
    """Compute TPR (true positive rate) and TNR (true negative rate)."""
    tp = fp = tn = fn = 0
    for o in outputs:
        gt = str(o.get("ground_truth_answer", "")).strip().lower()
        correct = o.get("answer_correct", False)

        is_positive = "yes" in gt and "no" not in gt
        if is_positive:
            if correct:
                tp += 1
            else:
                fn += 1
        else:
            if correct:
                tn += 1
            else:
                fp += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "tpr": tpr, "tnr": tnr,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "pos_total": tp + fn, "neg_total": tn + fp,
    }


def find_sweep_results(ckpt_base, config_names):
    """Discover eval_outputs.json files for sweep configs."""
    results = []  # list of (config_name, epoch, eval_data)

    ckpt_path = Path(ckpt_base)
    for entry in sorted(ckpt_path.iterdir()):
        if not entry.is_dir():
            continue
        config_name, epoch = parse_eval_dir_name(entry.name)
        if config_name is None or config_name not in config_names:
            continue
        eval_file = entry / "eval_outputs.json"
        if not eval_file.exists():
            continue
        with open(eval_file) as f:
            data = json.load(f)
        results.append((config_name, epoch, data))

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("ckpt_base", help="Checkpoint base directory")
    parser.add_argument(
        "--configs", nargs="+", default=SWEEP_NAMES,
        help="Config names to analyze (default: all sweep configs)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save analysis to JSON file",
    )
    args = parser.parse_args()

    results = find_sweep_results(args.ckpt_base, set(args.configs))

    if not results:
        print(f"No sweep eval results found in {args.ckpt_base}")
        print(f"Expected directories like: sweep-s3-c1-eval-ckpt_008/eval_outputs.json")
        return

    # Group by config
    by_config = defaultdict(list)
    for config_name, epoch, data in results:
        accuracy = data.get("accuracy", 0)
        total = data.get("total_samples", 0)
        correct = data.get("correct_answers", 0)
        per_class = compute_per_class_accuracy(data.get("outputs", []))
        by_config[config_name].append({
            "epoch": epoch,
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "per_class": per_class,
        })

    # Sort each config's results by epoch
    for config_name in by_config:
        by_config[config_name].sort(key=lambda x: x["epoch"])

    # Print results
    print("=" * 72)
    print("  HYPERPARAMETER SWEEP RESULTS")
    print("=" * 72)

    global_best = None

    for config_name in sorted(by_config.keys()):
        entries = by_config[config_name]

        # Extract hyperparams from config name
        m = re.match(r"sweep-s(\d+)-c(\d+)", config_name)
        if m:
            max_stage = int(m.group(1))
            c_thought = int(m.group(2))
            header = f"{config_name}  (max_latent_stage={max_stage}, c_thought={c_thought})"
        else:
            header = config_name

        print(f"\n{'─' * 72}")
        print(f"  {header}")
        print(f"  {'Epoch':>5}  {'Accuracy':>8}  {'Correct':>8}  {'TPR':>6}  {'TNR':>6}")
        print(f"  {'─' * 50}")

        best_entry = max(entries, key=lambda x: x["accuracy"])

        for entry in entries:
            marker = " *" if entry is best_entry else "  "
            pc = entry["per_class"]
            tpr_str = f"{pc['tpr']:.3f}" if pc["pos_total"] > 0 else "  n/a"
            tnr_str = f"{pc['tnr']:.3f}" if pc["neg_total"] > 0 else "  n/a"
            print(
                f"  {entry['epoch']:>5}  {entry['accuracy']:>8.4f}  "
                f"{entry['correct']:>3}/{entry['total']:<4}  "
                f"{tpr_str:>6}  {tnr_str:>6}{marker}"
            )

        print(f"  Best: epoch {best_entry['epoch']} → {best_entry['accuracy']:.4f}")

        if global_best is None or best_entry["accuracy"] > global_best["accuracy"]:
            global_best = {**best_entry, "config": config_name}

    # Overall winner
    print(f"\n{'=' * 72}")
    if global_best:
        m = re.match(r"sweep-s(\d+)-c(\d+)", global_best["config"])
        if m:
            print(f"  BEST: {global_best['config']}  epoch {global_best['epoch']}")
            print(f"         max_latent_stage={m.group(1)}, c_thought={m.group(2)}")
        else:
            print(f"  BEST: {global_best['config']}  epoch {global_best['epoch']}")
        print(f"         accuracy={global_best['accuracy']:.4f}  "
              f"({global_best['correct']}/{global_best['total']})")
        pc = global_best["per_class"]
        if pc["pos_total"] > 0 and pc["neg_total"] > 0:
            print(f"         TPR={pc['tpr']:.3f} ({pc['tp']}/{pc['pos_total']})  "
                  f"TNR={pc['tnr']:.3f} ({pc['tn']}/{pc['neg_total']})")
    print("=" * 72)

    # Save JSON summary
    if args.output:
        summary = {
            "best": global_best,
            "per_config": {
                name: {
                    "best_epoch": max(entries, key=lambda x: x["accuracy"])["epoch"],
                    "best_accuracy": max(entries, key=lambda x: x["accuracy"])["accuracy"],
                    "all_epochs": entries,
                }
                for name, entries in by_config.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
