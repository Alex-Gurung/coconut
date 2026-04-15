#!/usr/bin/env python3
"""Build reward-filtered Flawed Fictions train/val sets from trace JSON files.

This script reads one or more trace JSON arrays in the LiteReason FF format:
  [{"question": str, "steps": [str, ...], "answer": "\\boxed{Yes|No}"}, ...]

It uses the canonical Flawed Fictions reward function against the ground-truth
train/val JSONL files to keep only reward-passing examples, deduplicates them,
and writes:

- Coconut-style JSON arrays for train/val
- Optional CoLaR-style JSONL with a model-specific chat template applied
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _load_reward_module(path: Path):
    spec = importlib.util.spec_from_file_location("ff_reward_func", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reward module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trace_rows(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"{path} must contain a JSON list, got {type(data)}")
        rows.extend(data)
    return rows


def _load_gt_splits(gt_dir: Path) -> Dict[str, Dict[str, str]]:
    splits: Dict[str, Dict[str, str]] = {}
    for split in ("train", "val", "test"):
        split_path = gt_dir / f"{split}.jsonl"
        records: Dict[str, str] = {}
        with split_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records[str(obj["prompt"]).strip()] = str(obj["answer"]).strip()
        splits[split] = records
    return splits


def _normalize_trace_row(row: dict) -> Tuple[str, List[str], str]:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    raw_steps = row.get("steps", [])

    if isinstance(raw_steps, list):
        steps = [str(step).strip() for step in raw_steps if str(step).strip()]
    else:
        text = str(raw_steps).strip()
        steps = [text] if text else []

    return question, steps, answer


def _reconstruct_query(steps: Sequence[str], answer: str) -> str:
    reasoning = "\n".join(step for step in steps if step)
    if reasoning and answer:
        return f"{reasoning}\n{answer}"
    return reasoning or answer


def _dedupe_rows(rows: Iterable[dict]) -> List[dict]:
    seen: set[Tuple[str, str, Tuple[str, ...]]] = set()
    out: List[dict] = []
    for row in rows:
        key = (
            row["question"],
            row["answer"],
            tuple(row["steps"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _filter_rows(
    trace_rows: Sequence[dict],
    gt_splits: Dict[str, Dict[str, str]],
    reward_module,
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    candidates: Dict[str, List[Tuple[dict, str]]] = {"train": [], "val": []}
    stats = {
        "raw_rows": len(trace_rows),
        "dropped_missing_fields": 0,
        "dropped_unknown_split": 0,
        "reward_failed": 0,
        "reward_passed_before_dedup": 0,
        "train_final": 0,
        "val_final": 0,
    }

    split_for_question: Dict[str, str] = {}
    label_for_question: Dict[str, str] = {}
    for split in ("train", "val"):
        for question, label in gt_splits[split].items():
            split_for_question[question] = split
            label_for_question[question] = label

    for raw_row in trace_rows:
        question, steps, answer = _normalize_trace_row(raw_row)
        if not question or not steps or not answer:
            stats["dropped_missing_fields"] += 1
            continue
        split = split_for_question.get(question)
        if split not in ("train", "val"):
            stats["dropped_unknown_split"] += 1
            continue
        query = _reconstruct_query(steps, answer)
        row = {
            "question": question,
            "steps": steps,
            "answer": answer,
        }
        candidates[split].append((row, query))

    filtered: Dict[str, List[dict]] = {"train": [], "val": []}
    for split in ("train", "val"):
        rows = candidates[split]
        if not rows:
            continue
        queries = [query for _, query in rows]
        labels = [label_for_question[row["question"]] for row, _ in rows]
        rewards = reward_module.reward_func(
            queries=queries,
            prompts=[""] * len(rows),
            labels=labels,
        )["rewards"].tolist()

        for (row, _query), reward in zip(rows, rewards):
            if float(reward) == 1.0:
                filtered[split].append(row)
                stats["reward_passed_before_dedup"] += 1
            else:
                stats["reward_failed"] += 1

        filtered[split] = _dedupe_rows(filtered[split])
        stats[f"{split}_final"] = len(filtered[split])

    return filtered, stats


def _write_coconut_json(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def _apply_chat_template(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _write_colar_jsonl(out_dir: Path, split_rows: Dict[str, List[dict]], model_id: str) -> None:
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    for split in ("train", "val"):
        rows = split_rows[split]
        out_path = out_dir / f"{split}_colar_format.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(rows):
                rec = {
                    "idx": idx,
                    "question": _apply_chat_template(tokenizer, row["question"]),
                    "steps": "\n".join(row["steps"]),
                    "answer": row["answer"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reward-filtered FF train/val datasets.")
    ap.add_argument(
        "--trace",
        nargs="+",
        required=True,
        help="One or more trace JSON files to pool before split assignment.",
    )
    ap.add_argument(
        "--gt-dir",
        default="/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/data",
        help="Directory containing train.jsonl and val.jsonl ground-truth files.",
    )
    ap.add_argument(
        "--reward-func",
        default="/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/reward_func.py",
        help="Path to the canonical Flawed Fictions reward_func.py file.",
    )
    ap.add_argument("--train-out", required=True, help="Output Coconut-format train JSON path.")
    ap.add_argument("--val-out", required=True, help="Output Coconut-format val JSON path.")
    ap.add_argument(
        "--colar-out-dir",
        default=None,
        help="Optional output dir for CoLaR train_colar_format.jsonl / val_colar_format.jsonl.",
    )
    ap.add_argument(
        "--colar-model-id",
        default=None,
        help="HF model id for applying the CoLaR chat template when --colar-out-dir is used.",
    )
    args = ap.parse_args()

    if (args.colar_out_dir is None) != (args.colar_model_id is None):
        raise ValueError("--colar-out-dir and --colar-model-id must be provided together.")

    trace_paths = [Path(path) for path in args.trace]
    gt_dir = Path(args.gt_dir)
    reward_func_path = Path(args.reward_func)

    reward_module = _load_reward_module(reward_func_path)
    trace_rows = _load_trace_rows(trace_paths)
    gt_splits = _load_gt_splits(gt_dir)
    filtered, stats = _filter_rows(trace_rows, gt_splits, reward_module)

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    _write_coconut_json(train_out, filtered["train"])
    _write_coconut_json(val_out, filtered["val"])

    if args.colar_out_dir is not None and args.colar_model_id is not None:
        _write_colar_jsonl(Path(args.colar_out_dir), filtered, args.colar_model_id)

    print(json.dumps(stats, indent=2))
    print(f"wrote train -> {train_out}")
    print(f"wrote val   -> {val_out}")
    if args.colar_out_dir is not None:
        print(f"wrote colar -> {args.colar_out_dir}")


if __name__ == "__main__":
    main()
