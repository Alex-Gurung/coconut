#!/usr/bin/env python3

"""Shared helpers for checkpoint eval summaries."""

import json
import re
from pathlib import Path
from typing import Optional


def checkpoint_id(name: str) -> int:
    try:
        return int(str(name).split("_")[-1])
    except Exception:
        return -1


def latent_stage(epoch: int, epochs_per_stage: int, max_latent_stage: int) -> int:
    return min(epoch // epochs_per_stage, max_latent_stage)


def format_eval_suffix(eval_suffix: Optional[str]) -> str:
    return f"-{eval_suffix}" if eval_suffix else ""


def summary_basename(eval_suffix: Optional[str]) -> str:
    suffix = format_eval_suffix(eval_suffix)
    return f"eval_summary{suffix}" if suffix else "eval_summary"


def pretty_md(rows: list[dict]) -> str:
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


def collect_result(
    eval_dir: str | Path,
    ckpt_name: str,
    ckpt_id: int,
    epochs_per_stage: Optional[int] = None,
    max_latent_stage: Optional[int] = None,
    c_thought: Optional[int] = None,
) -> dict:
    eval_dir = Path(eval_dir)
    summary_path = eval_dir / "eval_outputs.json"
    payload = {}
    outputs = []
    if summary_path.exists():
        with open(summary_path, "r") as f:
            payload = json.load(f)
        outputs = payload.get("outputs", [])

    # Fall back to the saved eval config when the caller did not provide the
    # training schedule parameters explicitly.
    cfg = payload.get("config", {}) if payload else {}
    epochs_per_stage = (
        epochs_per_stage if epochs_per_stage is not None else cfg.get("epochs_per_stage", 1)
    )
    max_latent_stage = (
        max_latent_stage if max_latent_stage is not None else cfg.get("max_latent_stage", 100)
    )
    c_thought = c_thought if c_thought is not None else cfg.get("c_thought", 1)

    stage = latent_stage(ckpt_id, int(epochs_per_stage), int(max_latent_stage))
    num_latents = stage * int(c_thought)
    gen_counts = [e.get("gen_tokens", e.get("num_cot_tokens", 0)) for e in outputs if e]
    avg_gen = sum(gen_counts) / len(gen_counts) if gen_counts else 0.0

    return {
        "checkpoint": ckpt_name,
        "stage": stage,
        "num_latents": num_latents,
        "accuracy": payload.get("accuracy", 0.0),
        "avg_gen_tokens": avg_gen,
        "samples": payload.get("total_samples", 0),
        "eval_dir": str(eval_dir),
    }


def discover_eval_results(
    save_path: str | Path,
    run_name: str,
    epochs_per_stage: Optional[int] = None,
    max_latent_stage: Optional[int] = None,
    c_thought: Optional[int] = None,
    eval_suffix: Optional[str] = None,
) -> list[dict]:
    save_path = Path(save_path)
    if not save_path.is_dir():
        return []

    pattern = re.compile(
        rf"^{re.escape(run_name)}-eval-ckpt_(?P<ckpt>\d+)(?:-(?P<suffix>.+))?$"
    )
    rows = []

    for child in save_path.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        suffix_part = match.group("suffix")
        if eval_suffix is None:
            if suffix_part is not None:
                continue
        elif suffix_part != eval_suffix:
            continue

        summary_path = child / "eval_outputs.json"
        if not summary_path.exists():
            continue

        ckpt_id = int(match.group("ckpt"))
        rows.append(
            collect_result(
                eval_dir=child,
                ckpt_name=f"checkpoint_{ckpt_id}",
                ckpt_id=ckpt_id,
                epochs_per_stage=epochs_per_stage,
                max_latent_stage=max_latent_stage,
                c_thought=c_thought,
            )
        )

    return sorted(rows, key=lambda row: checkpoint_id(row["checkpoint"]))
