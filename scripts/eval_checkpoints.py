#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_id(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except Exception:
        return -1


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _find_checkpoints(save_dir: str) -> list[str]:
    if not os.path.isdir(save_dir):
        return []
    ckpts = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_")]
    ckpts = sorted(ckpts, key=_checkpoint_id)
    return ckpts


def _latent_stage(epoch: int, epochs_per_stage: int, max_latent_stage: int) -> int:
    return min(epoch // epochs_per_stage, max_latent_stage)


def _pretty_md(rows: list[dict]) -> str:
    header = "| checkpoint | stage | latents | accuracy | avg_gen_tokens | samples |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['stage']} | {row['num_latents']}"
            f" | {row['accuracy']:.4f}"
            f" | {row['avg_gen_tokens']:.1f} | {row['samples']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and summarize accuracy.")
    parser.add_argument("--train-config", required=True, help="Training config YAML to mirror.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids to use.")
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--eval-script", default=str(REPO_ROOT / "run.py"))
    parser.add_argument("--checkpoints", default=None,
                        help="Comma-separated checkpoint ids to eval (e.g. '8,16,24,32'). Default: all.")
    parser.add_argument("--val-path", default=None,
                        help="Override val_path in config (e.g. gsm_hard_data/qwen3_4b/val.json).")
    parser.add_argument("--eval-suffix", default=None,
                        help="Suffix appended to eval directory names (e.g. 'test' -> *-eval-ckpt_004-test).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip checkpoints that already have eval_outputs.json and collect their results.")
    parser.add_argument("--list-only", action="store_true",
                        help="Print the checkpoints that would be evaluated and exit without launching jobs.")
    parser.add_argument("--out", default=None, help="Output summary base path (no extension).")
    args = parser.parse_args()

    base_cfg = _load_yaml(args.train_config)
    save_path = base_cfg["save_path"]
    run_name = base_cfg["name"]
    save_dir = os.path.join(save_path, run_name)

    checkpoints = _find_checkpoints(save_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {save_dir}")

    if args.checkpoints:
        keep = {int(x.strip()) for x in args.checkpoints.split(",")}
        checkpoints = [c for c in checkpoints if _checkpoint_id(c) in keep]
        if not checkpoints:
            raise SystemExit(f"None of the requested checkpoint ids found in {save_dir}")

    print(f"Train config: {args.train_config}")
    print(f"Run directory: {save_dir}")
    print(
        "Checkpoint policy from config: "
        f"save_every={base_cfg.get('save_every', 'n/a')}, "
        f"save_only_improve={base_cfg.get('save_only_improve', 'n/a')}, "
        f"num_epochs={base_cfg.get('num_epochs', 'n/a')}"
    )
    print(f"Discovered checkpoints ({len(checkpoints)}): {', '.join(checkpoints)}")
    if args.list_only:
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise SystemExit("No GPUs specified.")

    tmp_dir = str(REPO_ROOT / "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Helper to collect results from an eval_outputs.json
    epochs_per_stage = base_cfg.get("epochs_per_stage", 1)
    max_latent_stage = base_cfg.get("max_latent_stage", 100)
    c_thought = base_cfg.get("c_thought", 1)

    def _collect_result(ckpt_name: str, ckpt_id: int, eval_dir: str) -> dict:
        stage = _latent_stage(ckpt_id, epochs_per_stage, max_latent_stage)
        num_latents = stage * c_thought
        summary_path = os.path.join(eval_dir, "eval_outputs.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                payload = json.load(f)
            outputs = payload.get("outputs", [])
            gen_counts = [e.get("gen_tokens", e.get("num_cot_tokens", 0)) for e in outputs if e]
            avg_gen = sum(gen_counts) / len(gen_counts) if gen_counts else 0.0
            return {
                "checkpoint": ckpt_name,
                "stage": stage,
                "num_latents": num_latents,
                "accuracy": payload.get("accuracy", 0.0),
                "avg_gen_tokens": avg_gen,
                "samples": payload.get("total_samples", 0),
                "eval_dir": eval_dir,
            }
        return {
            "checkpoint": ckpt_name,
            "stage": stage,
            "num_latents": num_latents,
            "accuracy": 0.0,
            "avg_gen_tokens": 0.0,
            "samples": 0,
            "eval_dir": eval_dir,
        }

    # Prepare jobs
    jobs = []
    existing_results = []
    suffix = f"-{args.eval_suffix}" if args.eval_suffix else ""

    for ckpt in checkpoints:
        ckpt_id = _checkpoint_id(ckpt)
        eval_name = f"{run_name}-eval-ckpt_{ckpt_id:03d}{suffix}"
        eval_dir = os.path.join(save_path, eval_name)

        # Skip if results already exist
        if args.skip_existing and os.path.exists(os.path.join(eval_dir, "eval_outputs.json")):
            existing_results.append(_collect_result(ckpt, ckpt_id, eval_dir))
            print(f"Skipping {ckpt} (existing results in {eval_dir})")
            continue

        eval_cfg = dict(base_cfg)
        eval_cfg["only_eval"] = True
        eval_cfg["enable_gen_eval"] = True
        if args.val_path:
            eval_cfg["val_path"] = args.val_path
        eval_cfg["load_model_path"] = os.path.join(save_dir, ckpt)
        eval_cfg["resume"] = ckpt_id
        eval_cfg["name"] = eval_name
        # Ensure eval uses full validation set
        eval_cfg["debug"] = False
        eval_cfg["use_ddp"] = True

        cfg_path = os.path.join(tmp_dir, f"eval_{run_name}_ckpt_{ckpt_id:03d}.yaml")
        _write_yaml(cfg_path, eval_cfg)
        jobs.append(
            {
                "ckpt": ckpt,
                "ckpt_id": ckpt_id,
                "cfg_path": cfg_path,
                "eval_dir": eval_dir,
            }
        )

    if existing_results:
        print(f"Collected {len(existing_results)} existing results.")
    if not jobs:
        print("All checkpoints already evaluated, writing summary.")

    # Launch jobs across GPUs
    running = {}
    pending = list(jobs)
    results = []
    completed_durations = []
    env_base = os.environ.copy()

    base_port = 29500

    def launch(job, gpu_id):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        port = base_port + gpus.index(gpu_id)
        cmd = [
            args.torchrun,
            "--nnodes",
            "1",
            "--nproc_per_node",
            "1",
            "--master_port",
            str(port),
            args.eval_script,
            job["cfg_path"],
        ]
        proc = subprocess.Popen(cmd, env=env)
        running[gpu_id] = (proc, job, time.time())

    # initial fill
    for gpu_id in gpus:
        if not pending:
            break
        job = pending.pop(0)
        launch(job, gpu_id)

    last_status = 0.0
    status_interval = 10.0
    start_time = time.time()

    def fmt_seconds(sec: float) -> str:
        if sec < 0:
            sec = 0
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    while running:
        time.sleep(2)
        for gpu_id in list(running.keys()):
            proc, job, job_start = running[gpu_id]
            if proc.poll() is None:
                continue
            if proc.returncode != 0:
                print(f"Eval failed for {job['ckpt']} on GPU {gpu_id} (exit {proc.returncode}).")
            duration = time.time() - job_start
            completed_durations.append(duration)
            results.append(_collect_result(job["ckpt"], job["ckpt_id"], job["eval_dir"]))
            del running[gpu_id]
            if pending:
                launch(pending.pop(0), gpu_id)

        now = time.time()
        if now - last_status >= status_interval:
            last_status = now
            avg = sum(completed_durations) / len(completed_durations) if completed_durations else None
            elapsed = now - start_time
            remaining_total = len(pending) + len(running)
            if avg is not None:
                eta_total = remaining_total * avg / max(1, len(gpus))
            else:
                eta_total = None

            print("\n=== Eval Progress ===")
            print(f"Elapsed: {fmt_seconds(elapsed)} | Done: {len(results)} | Running: {len(running)} | Pending: {len(pending)}")
            if eta_total is not None:
                print(f"Estimated remaining (overall): {fmt_seconds(eta_total)}")
            for gpu_id in gpus:
                if gpu_id in running:
                    proc, job, job_start = running[gpu_id]
                    job_elapsed = now - job_start
                    if avg is not None:
                        job_eta = avg - job_elapsed
                        print(f"GPU {gpu_id}: {job['ckpt']} | elapsed {fmt_seconds(job_elapsed)} | est remaining {fmt_seconds(job_eta)}")
                    else:
                        print(f"GPU {gpu_id}: {job['ckpt']} | elapsed {fmt_seconds(job_elapsed)} | est remaining unknown")
                else:
                    print(f"GPU {gpu_id}: idle")

    # Merge existing + newly evaluated, sort by checkpoint order
    results = existing_results + results
    results = sorted(results, key=lambda r: _checkpoint_id(r["checkpoint"]))
    summary_name = f"eval_summary{suffix}" if suffix else "eval_summary"
    out_base = args.out or os.path.join(save_dir, summary_name)
    with open(out_base + ".json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_base + ".md", "w") as f:
        f.write(_pretty_md(results))

    print("Evaluation summary written to:")
    print(out_base + ".json")
    print(out_base + ".md")
    if results:
        best = max(results, key=lambda r: r["accuracy"])
        print(
            f"Best checkpoint: {best['checkpoint']} (accuracy {best['accuracy']:.4f}) -> {best['eval_dir']}"
        )


if __name__ == "__main__":
    main()
