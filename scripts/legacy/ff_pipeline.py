#!/usr/bin/env python3
"""
Flawed Fictions pipeline helper for Coconut strict-mode workflows.

This script replaces ad-hoc shell heredocs with explicit subcommands:
- prepare-data
- dataset-stats
- tokenizer-check
- make-smoke-config
- make-train-config
- make-eval-config
- show-eval
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml


DEFAULT_TRAIN_TRACES = [
    "/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/working/traces/ff_qwen3_4b_train_positive.json",
    "/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/working/traces/ff_gemma3_4b_train_positive.json",
]

DEFAULT_VAL_TRACES = [
    "/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/working/traces/ff_qwen3_4b_val_positive.json",
    "/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/working/traces/ff_gemma3_4b_val_positive.json",
]


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def clean_trace_records(paths: Sequence[Path]) -> List[Dict]:
    out: List[Dict] = []
    seen: set[Tuple[str, str, Tuple[str, ...]]] = set()

    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list.")

        for row in rows:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            steps = [str(s).strip() for s in row.get("steps", []) if str(s).strip()]
            if not question or not answer or not steps:
                continue
            key = (question, answer, tuple(steps))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "question": question,
                    "steps": steps,
                    "answer": answer,
                }
            )
    return out


def label_counts(rows: Iterable[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        answer = str(row.get("answer", ""))
        counts[answer] = counts.get(answer, 0) + 1
    return counts


def cmd_prepare_data(args: argparse.Namespace) -> None:
    train_paths = [Path(p) for p in (args.train_trace or DEFAULT_TRAIN_TRACES)]
    val_paths = [Path(p) for p in (args.val_trace or DEFAULT_VAL_TRACES)]

    for path in train_paths + val_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing trace file: {path}")

    train_rows = clean_trace_records(train_paths)
    val_rows = clean_trace_records(val_paths)

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    save_json(train_out, train_rows)
    save_json(val_out, val_rows)

    print(f"wrote {train_out} rows={len(train_rows)} labels={label_counts(train_rows)}")
    print(f"wrote {val_out} rows={len(val_rows)} labels={label_counts(val_rows)}")


def cmd_dataset_stats(args: argparse.Namespace) -> None:
    for split, path_str in (("train", args.train), ("val", args.val)):
        path = Path(path_str)
        rows = load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list.")
        print(f"{split}: rows={len(rows)} labels={label_counts(rows)}")


def cmd_tokenizer_check(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for tokenizer-check."
        ) from exc

    cfg = load_yaml(Path(args.config))
    model_id = cfg["model_id"]
    latent_init_token = cfg.get("latent_init_token", "<<")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.eos_token is None:
        raise RuntimeError("Tokenizer has no eos_token.")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.add_tokens("<|start-latent|>")
    tok.add_tokens("<|end-latent|>")
    tok.add_tokens("<|latent|>")

    unk = tok.unk_token_id
    for token in ("<|latent|>", "<|start-latent|>", "<|end-latent|>"):
        token_id = tok.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or (unk is not None and token_id == unk):
            raise RuntimeError(f"Special token registration failed: {token}")

    init_id = tok.convert_tokens_to_ids(latent_init_token)
    if init_id is None or init_id < 0 or (unk is not None and init_id == unk):
        raise RuntimeError(
            f"Initialization token is missing from vocab: {latent_init_token!r}"
        )

    print("tokenizer check: PASS")
    print(f"model_id: {model_id}")
    print(f"eos_token: {tok.eos_token} ({tok.eos_token_id})")
    print(f"pad_token: {tok.pad_token} ({tok.pad_token_id})")
    print(f"latent_init_token: {latent_init_token!r} ({init_id})")


def _with_generated_name(cfg: Dict, prefix: str, name_override: str | None) -> Dict:
    cfg = dict(cfg)
    cfg["name"] = name_override if name_override else f"{prefix}-{now_tag()}"
    return cfg


def cmd_make_smoke_config(args: argparse.Namespace) -> None:
    cfg = load_yaml(Path(args.base))
    cfg = _with_generated_name(cfg, args.name_prefix, args.name)
    cfg["debug"] = True
    cfg["num_epochs"] = args.num_epochs
    cfg["epochs_per_stage"] = args.epochs_per_stage
    cfg["max_latent_stage"] = args.max_latent_stage
    cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    cfg["batch_size_training"] = args.batch_size
    save_yaml(Path(args.out), cfg)
    print(f"wrote {args.out}")
    print(f"name={cfg['name']}")


def cmd_make_train_config(args: argparse.Namespace) -> None:
    cfg = load_yaml(Path(args.base))
    cfg = _with_generated_name(cfg, args.name_prefix, args.name)
    save_yaml(Path(args.out), cfg)
    print(f"wrote {args.out}")
    print(f"name={cfg['name']}")


def _latest_checkpoint(save_path: str, run_name: str) -> Tuple[Path, int]:
    pattern = f"{save_path}/{run_name}/checkpoint_*"
    candidates = sorted(
        glob.glob(pattern),
        key=lambda p: int(Path(p).name.split("_")[-1]),
    )
    if not candidates:
        raise RuntimeError(f"No checkpoints found for pattern: {pattern}")
    ckpt = Path(candidates[-1]).resolve()
    epoch = int(ckpt.name.split("_")[-1])
    return ckpt, epoch


def cmd_make_eval_config(args: argparse.Namespace) -> None:
    train_cfg = load_yaml(Path(args.train_config))
    eval_cfg = load_yaml(Path(args.base_eval))

    if args.checkpoint:
        ckpt = Path(args.checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
        epoch = args.resume if args.resume is not None else int(ckpt.name.split("_")[-1])
    else:
        ckpt, epoch = _latest_checkpoint(train_cfg["save_path"], train_cfg["name"])

    # Keep eval model strictly aligned with the train run config.
    eval_cfg["model_id"] = train_cfg["model_id"]
    eval_cfg["name"] = train_cfg["name"]
    eval_cfg["load_model_path"] = str(ckpt)
    eval_cfg["resume"] = int(epoch)
    if args.val_path:
        eval_cfg["val_path"] = args.val_path

    save_yaml(Path(args.out), eval_cfg)
    print(f"wrote {args.out}")
    print(f"checkpoint={ckpt}")
    print(f"resume={epoch}")
    print(f"name={eval_cfg['name']}")


LITEREASON_DATA_DIR = Path("/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/data")


def _convert_litereason_jsonl(src: Path) -> List[Dict]:
    """Convert a litereason_anon JSONL file to coconut eval format."""
    rows: List[Dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            answer_str = r"\boxed{Yes}" if rec["answer"] == "1" else r"\boxed{No}"
            rows.append({
                "question": rec["prompt"],
                "answer": answer_str,
                "steps": [],
            })
    return rows


def cmd_convert_litereason(args: argparse.Namespace) -> None:
    for split, out_path in [("val", Path(args.val_out)), ("test", Path(args.test_out))]:
        src = Path(args.data_dir) / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        rows = _convert_litereason_jsonl(src)
        save_json(out_path, rows)
        print(f"wrote {out_path} rows={len(rows)} labels={label_counts(rows)}")


DEFAULT_TRAIN_ALL_TRACE = "/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/working/traces/ff_qwen3_4b_train_all.json"


def load_ground_truth(gt_dir: Path) -> Dict[str, str]:
    """Read train.jsonl + val.jsonl from litereason_anon, return {prompt_text: '1'/'0'}."""
    gt: Dict[str, str] = {}
    for split in ("train", "val"):
        src = gt_dir / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing GT file: {src}")
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gt[rec["prompt"].strip()] = str(rec["answer"])
    return gt


def filter_correct_traces(
    traces: List[Dict], gt: Dict[str, str]
) -> Tuple[List[Dict], List[Dict]]:
    """Filter traces for non-empty answer+steps and correctness vs ground truth.

    Returns (correct_yes, correct_no) lists.
    """
    correct_yes: List[Dict] = []
    correct_no: List[Dict] = []
    for row in traces:
        answer = str(row.get("answer", "")).strip()
        steps = [str(s).strip() for s in row.get("steps", []) if str(s).strip()]
        if not answer or not steps:
            continue
        # Map boxed answer to GT label
        if answer == r"\boxed{Yes}":
            pred_label = "1"
        elif answer == r"\boxed{No}":
            pred_label = "0"
        else:
            continue
        question = str(row.get("question", "")).strip()
        gt_label = gt.get(question)
        if gt_label is None or gt_label != pred_label:
            continue
        rec = {"question": question, "steps": steps, "answer": answer}
        if pred_label == "1":
            correct_yes.append(rec)
        else:
            correct_no.append(rec)
    return correct_yes, correct_no


def cmd_prepare_balanced_data(args: argparse.Namespace) -> None:
    gt = load_ground_truth(Path(args.gt_dir))
    traces = load_json(Path(args.train_trace))
    if not isinstance(traces, list):
        raise ValueError(f"{args.train_trace} must contain a JSON list.")

    correct_yes, correct_no = filter_correct_traces(traces, gt)
    print(f"correct traces: Yes={len(correct_yes)} No={len(correct_no)}")

    # Dedup using same key pattern as clean_trace_records
    def dedup(rows: List[Dict]) -> List[Dict]:
        seen: set[Tuple[str, str, Tuple[str, ...]]] = set()
        out: List[Dict] = []
        for row in rows:
            key = (row["question"], row["answer"], tuple(row["steps"]))
            if key not in seen:
                seen.add(key)
                out.append(row)
        return out

    correct_yes = dedup(correct_yes)
    correct_no = dedup(correct_no)
    print(f"after dedup: Yes={len(correct_yes)} No={len(correct_no)}")

    if args.balance_strategy == "downsample":
        minority = min(len(correct_yes), len(correct_no))
        rng = random.Random(args.seed)
        if len(correct_yes) > minority:
            correct_yes = rng.sample(correct_yes, minority)
        if len(correct_no) > minority:
            correct_no = rng.sample(correct_no, minority)
        print(f"after downsample: Yes={len(correct_yes)} No={len(correct_no)}")

    combined = correct_yes + correct_no
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    out_path = Path(args.train_out)
    save_json(out_path, combined)
    print(f"wrote {out_path} rows={len(combined)} labels={label_counts(combined)}")


def cmd_show_eval(args: argparse.Namespace) -> None:
    if args.eval_output:
        out_path = Path(args.eval_output)
    else:
        cfg = load_yaml(Path(args.eval_config))
        out_path = Path(cfg["save_path"]) / cfg["name"] / "eval_outputs.json"

    if not out_path.exists():
        raise FileNotFoundError(f"Eval output not found: {out_path}")

    result = load_json(out_path)
    print(f"eval_outputs: {out_path}")
    print(f"samples: {result.get('total_samples')}")
    print(f"accuracy: {result.get('accuracy')}")
    print(f"cot_exact_match: {result.get('cot_exact_match')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Flawed Fictions strict pipeline helper.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Build ff_data/train.json and ff_data/val.json from traces.")
    p_prepare.add_argument("--train-trace", action="append", help="Trace file for train split. Repeatable.")
    p_prepare.add_argument("--val-trace", action="append", help="Trace file for val split. Repeatable.")
    p_prepare.add_argument("--train-out", default="ff_data/train.json")
    p_prepare.add_argument("--val-out", default="ff_data/val.json")
    p_prepare.set_defaults(func=cmd_prepare_data)

    p_stats = sub.add_parser("dataset-stats", help="Show counts for prepared Coconut JSON files.")
    p_stats.add_argument("--train", default="ff_data/train.json")
    p_stats.add_argument("--val", default="ff_data/val.json")
    p_stats.set_defaults(func=cmd_dataset_stats)

    p_tok = sub.add_parser("tokenizer-check", help="Run strict tokenizer checks against a training config.")
    p_tok.add_argument("--config", default="args/legacy/qwen3_ff_coconut_strict_train.yaml")
    p_tok.set_defaults(func=cmd_tokenizer_check)

    p_smoke = sub.add_parser("make-smoke-config", help="Create a debug/smoke training config from a base config.")
    p_smoke.add_argument("--base", default="args/legacy/qwen3_ff_coconut_strict_train.yaml")
    p_smoke.add_argument("--out", default="args/legacy/generated/tmp_smoke_train.yaml")
    p_smoke.add_argument("--name-prefix", default="qwen3-ff-coconut-smoke")
    p_smoke.add_argument("--name", default=None)
    p_smoke.add_argument("--num-epochs", type=int, default=1)
    p_smoke.add_argument("--epochs-per-stage", type=int, default=1)
    p_smoke.add_argument("--max-latent-stage", type=int, default=1)
    p_smoke.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p_smoke.add_argument("--batch-size", type=int, default=1)
    p_smoke.set_defaults(func=cmd_make_smoke_config)

    p_train = sub.add_parser("make-train-config", help="Create a timestamped training config from a base config.")
    p_train.add_argument("--base", default="args/legacy/qwen3_ff_coconut_strict_train.yaml")
    p_train.add_argument("--out", default="args/legacy/generated/qwen3_ff_coconut_run.yaml")
    p_train.add_argument("--name-prefix", default="qwen3-ff-coconut")
    p_train.add_argument("--name", default=None)
    p_train.set_defaults(func=cmd_make_train_config)

    p_eval = sub.add_parser("make-eval-config", help="Create eval config for latest (or explicit) checkpoint.")
    p_eval.add_argument("--train-config", default="args/legacy/generated/qwen3_ff_coconut_run.yaml")
    p_eval.add_argument("--base-eval", default="args/legacy/qwen3_ff_coconut_strict_eval.yaml")
    p_eval.add_argument("--out", default="args/legacy/generated/qwen3_ff_coconut_run_eval.yaml")
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--resume", type=int, default=None)
    p_eval.add_argument("--val-path", default=None)
    p_eval.set_defaults(func=cmd_make_eval_config)

    p_conv = sub.add_parser("convert-litereason", help="Convert litereason_anon JSONL splits to coconut eval format.")
    p_conv.add_argument("--data-dir", default=str(LITEREASON_DATA_DIR))
    p_conv.add_argument("--val-out", default="ff_data/val_litereason.json")
    p_conv.add_argument("--test-out", default="ff_data/test_litereason.json")
    p_conv.set_defaults(func=cmd_convert_litereason)

    p_bal = sub.add_parser("prepare-balanced-data", help="Build balanced Yes+No training data from all-prediction traces.")
    p_bal.add_argument("--train-trace", default=DEFAULT_TRAIN_ALL_TRACE)
    p_bal.add_argument("--gt-dir", default=str(LITEREASON_DATA_DIR))
    p_bal.add_argument("--balance-strategy", choices=["downsample", "all"], default="downsample")
    p_bal.add_argument("--seed", type=int, default=42)
    p_bal.add_argument("--train-out", default="ff_data/qwen_balanced_train.json")
    p_bal.set_defaults(func=cmd_prepare_balanced_data)

    p_show = sub.add_parser("show-eval", help="Show key metrics from eval_outputs.json.")
    p_show.add_argument("--eval-config", default="args/legacy/generated/qwen3_ff_coconut_run_eval.yaml")
    p_show.add_argument("--eval-output", default=None)
    p_show.set_defaults(func=cmd_show_eval)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
