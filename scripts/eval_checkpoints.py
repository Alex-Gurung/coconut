#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import yaml


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


def _pretty_md(rows: list[dict]) -> str:
    header = "| checkpoint | accuracy | cot_em | samples | eval_dir |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['accuracy']:.4f} | {row['cot_em']:.4f} | {row['samples']} | {row['eval_dir']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and summarize accuracy.")
    parser.add_argument("--train-config", required=True, help="Training config YAML to mirror.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids to use.")
    parser.add_argument("--torchrun", default="/mnt/disk/coconut/new4/bin/torchrun")
    parser.add_argument("--eval-script", default="/mnt/disk/coconut/run.py")
    parser.add_argument("--out", default=None, help="Output summary base path (no extension).")
    args = parser.parse_args()

    base_cfg = _load_yaml(args.train_config)
    save_path = base_cfg["save_path"]
    run_name = base_cfg["name"]
    save_dir = os.path.join(save_path, run_name)

    checkpoints = _find_checkpoints(save_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {save_dir}")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise SystemExit("No GPUs specified.")

    tmp_dir = "/mnt/disk/coconut/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare jobs
    jobs = []
    for ckpt in checkpoints:
        ckpt_id = _checkpoint_id(ckpt)
        eval_cfg = dict(base_cfg)
        eval_cfg["only_eval"] = True
        eval_cfg["load_model_path"] = os.path.join(save_dir, ckpt)
        eval_cfg["resume"] = ckpt_id
        eval_cfg["name"] = f"{run_name}-eval-ckpt_{ckpt_id:03d}"
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
                "eval_dir": os.path.join(save_path, eval_cfg["name"]),
            }
        )

    # Launch jobs across GPUs
    running = {}
    pending = list(jobs)
    results = []
    completed_durations = []
    env_base = os.environ.copy()

    def launch(job, gpu_id):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            args.torchrun,
            "--nnodes",
            "1",
            "--nproc_per_node",
            "1",
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
            # Collect results if present
            summary_path = os.path.join(job["eval_dir"], "eval_outputs.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    payload = json.load(f)
                results.append(
                    {
                        "checkpoint": job["ckpt"],
                        "accuracy": payload.get("accuracy", 0.0),
                        "cot_em": payload.get("cot_exact_match", 0.0),
                        "samples": payload.get("total_samples", 0),
                        "eval_dir": job["eval_dir"],
                    }
                )
            else:
                results.append(
                    {
                        "checkpoint": job["ckpt"],
                        "accuracy": 0.0,
                        "cot_em": 0.0,
                        "samples": 0,
                        "eval_dir": job["eval_dir"],
                    }
                )
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

    # Sort and write summary
    results = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    out_base = args.out or os.path.join(save_dir, "eval_summary")
    with open(out_base + ".json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_base + ".md", "w") as f:
        f.write(_pretty_md(results))

    print("Evaluation summary written to:")
    print(out_base + ".json")
    print(out_base + ".md")
    if results:
        best = results[0]
        print(
            f"Best checkpoint: {best['checkpoint']} (accuracy {best['accuracy']:.4f}) -> {best['eval_dir']}"
        )


if __name__ == "__main__":
    main()
