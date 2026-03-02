#!/usr/bin/env python
"""
Run evaluation across checkpoints and/or seeds, parallelized across GPUs.

Modes:
  Directory mode — evaluate every checkpoint_* in a directory:
    python scripts/eval/run_eval_n_times.py checkpoints/my-run/ --config-base args/eval.yaml --num-gpus 4

  Single checkpoint, multiple seeds:
    python scripts/eval/run_eval_n_times.py checkpoints/my-run/checkpoint_14 5 --config-base args/eval.yaml --num-gpus 4

  Directory + multiple seeds per checkpoint:
    python scripts/eval/run_eval_n_times.py checkpoints/my-run/ 3 --config-base args/eval.yaml --num-gpus 4

  Quick benchmark (5 samples, 1 checkpoint):
    python scripts/eval/run_eval_n_times.py checkpoints/my-run/checkpoint_4 --max-samples 5 --num-gpus 1

The `resume` value is automatically set to the checkpoint epoch number so the
correct latent stage is used (critical for coconut models).
"""

import argparse
import yaml
import subprocess
import os
import sys
import time
import json
import threading
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed


def format_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def discover_checkpoints(directory):
    """Find all checkpoint_N entries in a directory, sorted by epoch number."""
    ckpts = []
    for name in os.listdir(directory):
        if name.startswith("checkpoint_"):
            try:
                epoch = int(name.split("_", 1)[1])
                ckpts.append((epoch, os.path.join(directory, name)))
            except ValueError:
                continue
    ckpts.sort(key=lambda x: x[0])
    return ckpts  # list of (epoch, full_path)


def read_progress_file(save_path, config_name):
    """Try to read a progress.json written by run.py. Returns dict or None."""
    progress_file = os.path.join(save_path, config_name, "progress.json")
    try:
        with open(progress_file, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation across checkpoints/seeds, parallelized across GPUs"
    )
    parser.add_argument(
        "checkpoint_or_dir",
        help="Path to a single checkpoint, checkpoint directory, or HuggingFace model ID",
    )
    parser.add_argument(
        "num_runs", type=int, nargs="?", default=1,
        help="Seeds per checkpoint (default: 1)",
    )
    parser.add_argument(
        "--config-base", default="args/qwen3_ff_coconut_strict_eval.yaml",
        help="Base config file to use as template",
    )
    parser.add_argument("--name", default=None, help="Name prefix for eval output dirs")
    parser.add_argument("--val-path", default=None, help="Path to validation/test data")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to parallelize across")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of eval samples (for quick benchmarking)")
    parser.add_argument("--every", type=int, default=1,
                        help="Evaluate every Nth checkpoint (default: 1 = all)")
    parser.add_argument("--base-port", type=int, default=29500,
                        help="Base master port for torchrun (default: 29500)")

    args = parser.parse_args()

    # Load base config
    if not os.path.exists(args.config_base):
        print(f"Error: Config file {args.config_base} not found")
        sys.exit(1)

    with open(args.config_base, "r") as f:
        base_config = yaml.safe_load(f)

    base_config["only_eval"] = True
    if args.val_path:
        base_config["val_path"] = args.val_path
    if args.max_samples:
        base_config["max_eval_samples"] = args.max_samples

    save_path = base_config.get("save_path", "checkpoints")

    # ── Build job list ──────────────────────────────────────────────
    target = args.checkpoint_or_dir
    checkpoints = []

    if os.path.isdir(target):
        checkpoints = discover_checkpoints(target)
        if not checkpoints:
            print(f"Error: No checkpoint_* entries found in {target}")
            sys.exit(1)
        if args.every > 1:
            checkpoints = checkpoints[::args.every]
        dir_name = os.path.basename(os.path.normpath(target))
        print(f"Directory mode: found {len(checkpoints)} checkpoints in {target}")
    else:
        basename = os.path.basename(target)
        if basename.startswith("checkpoint_"):
            try:
                epoch = int(basename.split("_", 1)[1])
                checkpoints = [(epoch, target)]
            except ValueError:
                checkpoints = [(0, target)]
        else:
            checkpoints = [(0, target)]
        dir_name = basename

    name_base = args.name or dir_name

    tmp_dir = Path("/mnt/disk/coconut/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for epoch, ckpt_path in checkpoints:
        for seed_idx in range(args.num_runs):
            config = base_config.copy()
            config["load_model_path"] = ckpt_path
            config["resume"] = epoch
            config["seed"] = 42 + seed_idx * 1000

            ckpt_tag = f"ckpt_{epoch:03d}"
            if args.num_runs > 1:
                run_tag = f"{ckpt_tag}_seed{seed_idx}"
            else:
                run_tag = ckpt_tag
            config["name"] = f"{name_base}-eval-{run_tag}"

            job_idx = len(jobs)
            config_file = str(tmp_dir / f"eval_job_{job_idx}_config.yaml")
            with open(config_file, "w") as f:
                yaml.dump(config, f)

            label = f"checkpoint_{epoch}"
            if args.num_runs > 1:
                label += f" seed={config['seed']}"

            jobs.append({
                "label": label,
                "config_file": config_file,
                "config_name": config["name"],
            })

    num_jobs = len(jobs)
    num_gpus = min(args.num_gpus, num_jobs)
    base_port = args.base_port

    samples_note = ""
    if args.max_samples:
        samples_note = f"  |  Max samples: {args.max_samples}"
    print(f"Total jobs: {num_jobs}  |  GPUs: {num_gpus}  |  Seeds per checkpoint: {args.num_runs}{samples_note}")
    print(f"Config base: {args.config_base}")
    print(f"Output prefix: {name_base}-eval-*")
    print()

    # ── Parallel execution ──────────────────────────────────────────
    gpu_status = {}  # gpu_id -> {"job_idx", "label", "config_name", "start"} or None
    status_lock = threading.Lock()
    completed = []  # (job_idx, returncode)
    start_time = time.time()

    gpu_queue = Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)
        gpu_status[i] = None

    def run_job(job_idx):
        job = jobs[job_idx]
        gpu_id = gpu_queue.get()
        with status_lock:
            gpu_status[gpu_id] = {
                "job_idx": job_idx,
                "label": job["label"],
                "config_name": job["config_name"],
                "start": time.time(),
            }
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            master_port = base_port + job_idx
            cmd = [
                "torchrun",
                "--nnodes", "1",
                "--nproc_per_node", "1",
                "--master_port", str(master_port),
                "run.py",
                job["config_file"],
            ]
            result = subprocess.run(
                cmd, cwd="/mnt/disk/coconut", env=env,
                capture_output=True, text=True,
            )
            with status_lock:
                completed.append((job_idx, result.returncode))
                gpu_status[gpu_id] = None
            if result.returncode != 0:
                print(f"\n[{job['label']}] FAILED on GPU {gpu_id} (exit {result.returncode})")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-10:]:
                        print(f"  {line}")
            return job_idx, result.returncode
        finally:
            gpu_queue.put(gpu_id)

    # Progress display
    stop_progress = threading.Event()

    def progress_loop():
        while not stop_progress.is_set():
            stop_progress.wait(timeout=10)
            if stop_progress.is_set():
                break
            with status_lock:
                elapsed = time.time() - start_time
                n_done = len(completed)
                n_running = sum(1 for v in gpu_status.values() if v is not None)
                n_pending = num_jobs - n_done - n_running

                lines = ["\n=== Eval Progress ==="]
                lines.append(
                    f"Elapsed: {format_elapsed(elapsed)} | "
                    f"Done: {n_done}/{num_jobs} | "
                    f"Running: {n_running} | Pending: {n_pending}"
                )
                if n_done > 0:
                    avg_per_run = elapsed / n_done
                    remaining_runs = num_jobs - n_done
                    est_remaining = (remaining_runs / max(n_running, 1)) * avg_per_run
                    lines.append(f"Estimated remaining: {format_elapsed(est_remaining)}")

                for gpu_id in sorted(gpu_status.keys()):
                    info = gpu_status[gpu_id]
                    if info is not None:
                        gpu_elapsed = time.time() - info["start"]
                        # Read per-sample progress from run.py
                        prog = read_progress_file(save_path, info["config_name"])
                        if prog:
                            done = prog["samples_done"]
                            tot = prog["samples_total"]
                            avg = prog["avg_seconds_per_sample"]
                            last_toks = prog.get("last_generated_tokens", "?")
                            est_job = avg * (tot - done)
                            lines.append(
                                f"  GPU {gpu_id}: {info['label']} | "
                                f"{done}/{tot} samples | "
                                f"{avg:.1f}s/sample | "
                                f"last={last_toks} toks | "
                                f"est {format_elapsed(est_job)} left"
                            )
                        else:
                            lines.append(
                                f"  GPU {gpu_id}: {info['label']} | "
                                f"loading model... | {format_elapsed(gpu_elapsed)}"
                            )
                    else:
                        lines.append(f"  GPU {gpu_id}: idle")
                print("\n".join(lines))
                sys.stdout.flush()

    progress_thread = threading.Thread(target=progress_loop, daemon=True)
    progress_thread.start()

    failed = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(run_job, i): i for i in range(num_jobs)}
        for future in as_completed(futures):
            job_idx, returncode = future.result()
            if returncode != 0:
                failed.append(job_idx)

    stop_progress.set()
    progress_thread.join(timeout=2)

    # Clean up temp configs
    for job in jobs:
        try:
            os.remove(job["config_file"])
        except OSError:
            pass

    elapsed = time.time() - start_time
    print(f"\n{'='*40}")
    print(f"All jobs finished in {format_elapsed(elapsed)}")
    print(f"  Succeeded: {num_jobs - len(failed)}/{num_jobs}")
    if failed:
        print(f"  Failed: {[jobs[i]['label'] for i in sorted(failed)]}")
        sys.exit(1)

    print(f"Outputs saved to: {save_path}/{name_base}-eval-*/eval_outputs.json")
    print(f"\nTo combine results:")
    print(f"  python scripts/eval/combine_evals.py {save_path}/{name_base}-eval-ckpt_*/eval_outputs.json")


if __name__ == "__main__":
    main()
