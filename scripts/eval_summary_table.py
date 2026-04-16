#!/usr/bin/env python3

"""Print a terminal summary table for checkpoint eval results."""

import argparse
import json
from pathlib import Path

import yaml

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
except ImportError as exc:
    raise SystemExit(
        "The `rich` package is required for scripts/eval_summary_table.py. "
        "Install it with `pip install -r requirements.txt`."
    ) from exc

from eval_summary_utils import checkpoint_id, collect_result, discover_eval_results, summary_basename


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _rows_from_run_dir(run_dir, eval_suffix, refresh):
    summary_path = run_dir / f"{summary_basename(eval_suffix)}.json"
    if summary_path.exists() and not refresh:
        with open(summary_path, "r") as f:
            return json.load(f)

    # `--refresh` intentionally bypasses the cached summary so the table can be
    # rebuilt directly from the current eval_outputs.json files on disk.
    return discover_eval_results(
        save_path=run_dir.parent,
        run_name=run_dir.name,
        eval_suffix=eval_suffix,
    )


def _row_from_eval_dir(eval_dir):
    payload_path = eval_dir / "eval_outputs.json"
    with open(payload_path, "r") as f:
        payload = json.load(f)

    cfg = payload.get("config", {})
    ckpt_path = Path(str(payload.get("checkpoint", "checkpoint_unknown")))
    ckpt_name = ckpt_path.name
    ckpt_num = checkpoint_id(ckpt_name)
    if ckpt_num < 0:
        ckpt_num = int(cfg.get("resume", 0))
        ckpt_name = f"checkpoint_{ckpt_num}"

    return [
        collect_result(
            eval_dir=eval_dir,
            ckpt_name=ckpt_name,
            ckpt_id=ckpt_num,
            epochs_per_stage=cfg.get("epochs_per_stage"),
            max_latent_stage=cfg.get("max_latent_stage"),
            c_thought=cfg.get("c_thought"),
        )
    ]


def _rows_from_target(path, eval_suffix, refresh):
    # Accept the same path a user is likely to have on hand: a config, a run
    # directory, a single eval directory, or a saved summary JSON.
    if path.is_file() and path.suffix in {".yaml", ".yml"}:
        cfg = _load_yaml(str(path))
        run_dir = Path(cfg["save_path"]) / cfg["name"]
        return _rows_from_run_dir(run_dir, eval_suffix, refresh), str(run_dir)

    if path.is_file() and path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f), str(path)

    if path.is_dir() and (path / "eval_outputs.json").exists():
        return _row_from_eval_dir(path), str(path)

    if path.is_dir():
        return _rows_from_run_dir(path, eval_suffix, refresh), str(path)

    raise SystemExit(f"Unsupported target: {path}")


def _sort_rows(rows, sort_key, reverse):
    if sort_key == "accuracy":
        return sorted(rows, key=lambda row: (row["accuracy"], row["checkpoint"]), reverse=not reverse)

    return sorted(rows, key=lambda row: checkpoint_id(row["checkpoint"]), reverse=reverse)


def _print_table(rows, source_label, sort_key, reverse, limit):
    if not rows:
        raise SystemExit(f"No evaluation rows found for {source_label}")

    best = max(rows, key=lambda row: row["accuracy"])
    rows = _sort_rows(rows, sort_key, reverse)
    if limit is not None:
        rows = rows[:limit]

    table = Table(
        title=f"Eval Summary: {Path(source_label).name}",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Checkpoint", justify="right")
    table.add_column("Stage", justify="right")
    table.add_column("Latents", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Avg Gen", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Eval Dir")

    # Highlight the best checkpoint row so the terminal view immediately shows
    # the winner without requiring a second scan of the table.
    for row in rows:
        style = "bold green" if row["checkpoint"] == best["checkpoint"] else ""
        table.add_row(
            str(row["checkpoint"]),
            str(row["stage"]),
            str(row["num_latents"]),
            f"{row['accuracy']:.4f}",
            f"{row['avg_gen_tokens']:.1f}",
            str(row["samples"]),
            Path(row["eval_dir"]).name,
            style=style,
        )

    console = Console()
    console.print(table)
    console.print(
        f"Best checkpoint: [bold green]{best['checkpoint']}[/] "
        f"(accuracy {best['accuracy']:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Print an eval summary table for a run directory or summary JSON."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Run directory, eval directory, eval_summary.json, or train/eval YAML.",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="Training config YAML to resolve into a run directory.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Eval suffix to select (e.g. 'test' -> eval_summary-test.json).",
    )
    parser.add_argument(
        "--sort",
        choices=["checkpoint", "accuracy"],
        default="checkpoint",
        help="Sort rows by checkpoint order or accuracy.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the chosen sort order.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show only the first N rows after sorting.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild rows from eval_outputs.json directories instead of loading eval_summary.json.",
    )
    args = parser.parse_args()

    raw_target = args.train_config or args.target
    if not raw_target:
        raise SystemExit("Provide a target path or --train-config.")

    rows, source_label = _rows_from_target(Path(raw_target), args.suffix, args.refresh)
    _print_table(rows, source_label, args.sort, args.reverse, args.limit)


if __name__ == "__main__":
    main()
