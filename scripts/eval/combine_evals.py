import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from scipy import stats

def combine_and_analyze_evals(eval_dir: str, output_path: str = "combined_eval.json"):
    """
    Combine multiple eval outputs and compute SEM metrics grouped by question.
    Handles two patterns:
    - eval_outputs_v*.json in a single directory
    - eval_outputs.json in multiple subdirectories (checkpoint_13_eval_v0/, etc.)
    """

    eval_dir = Path(eval_dir)

    # Try pattern 1: eval_outputs_v*.json in single directory
    eval_files = sorted(eval_dir.glob("eval_outputs_v*.json"))

    # Try pattern 2: eval_outputs.json in subdirectories
    if not eval_files:
        eval_files = sorted(eval_dir.glob("*/eval_outputs.json"))

    print(f"Found {len(eval_files)} eval files")

    if not eval_files:
        print("No eval files found!")
        print(f"Searched in: {eval_dir}")
        print("Expected patterns: eval_outputs_v*.json or */eval_outputs.json")
        return

    # Group responses by question
    responses_by_question = defaultdict(list)
    first_config = None
    first_checkpoint = None
    all_responses = []

    for eval_file in eval_files:
        # Show relative path for clarity
        rel_path = eval_file.relative_to(eval_dir) if eval_file.parent != eval_dir else eval_file.name
        print(f"Loading {rel_path}...")
        with open(eval_file) as f:
            data = json.load(f)

            if first_config is None:
                first_config = data["config"]
                first_checkpoint = data["checkpoint"]

            for output in data["outputs"]:
                # Convert ground truth to string for comparison
                ground_truth = str(output.get("ground_truth_answer"))
                extracted = str(output.get("extracted_answer", ""))

                response_data = {
                    "idx": output["idx"],
                    "extracted_answer": extracted,
                    "ground_truth_answer": ground_truth,
                    "generated_output": output.get("generated_output", ""),
                    "boxed_extracted_answer": output.get("boxed_extracted_answer"),
                    "extracted_cot": output.get("extracted_cot"),
                    "ground_truth_cot": output.get("ground_truth_cot", ""),
                    "answer_correct": output.get("answer_correct"),
                    "num_cot_tokens": output.get("num_cot_tokens", 0),
                }

                all_responses.append(response_data)

                # Group by question
                question_key = output["question"]
                responses_by_question[question_key].append(response_data)

    print(f"Total responses: {len(all_responses)}")
    print(f"Grouped responses into {len(responses_by_question)} unique questions")

    # Compute metrics from all responses (not per-question)
    all_token_lengths = []
    correct_count = 0

    for response in all_responses:
        # Use pre-computed answer_correct if available, otherwise compute
        if response["answer_correct"] is not None:
            if response["answer_correct"]:
                correct_count += 1
        else:
            if response["extracted_answer"] == response["ground_truth_answer"]:
                correct_count += 1

        # Token count
        num_tokens = response.get("num_cot_tokens", 0)
        if num_tokens > 0:
            all_token_lengths.append(num_tokens)
        else:
            # Fallback to counting tokens in output
            output = response.get("generated_output", "")
            tokens = len(output.split())
            all_token_lengths.append(tokens)

    # Compute per-question metrics
    question_metrics = []
    per_question_accuracies = []

    for question, responses in responses_by_question.items():
        # Accuracy per question
        correct = sum(1 for r in responses if (r["answer_correct"] if r["answer_correct"] is not None else r["extracted_answer"] == r["ground_truth_answer"]))
        accuracy = correct / len(responses) if len(responses) > 0 else 0
        per_question_accuracies.append(accuracy)

        # Average tokens for this question
        question_tokens = [r.get("num_cot_tokens", 0) if r.get("num_cot_tokens", 0) > 0 else len(r.get("generated_output", "").split()) for r in responses]
        avg_tokens = np.mean(question_tokens) if question_tokens else 0

        question_metrics.append({
            "question": question[:100] + "..." if len(question) > 100 else question,
            "num_responses": len(responses),
            "accuracy": float(accuracy),
            "correct_answers": correct,
            "avg_token_length": float(avg_tokens),
        })

    # Compute overall statistics using all responses
    overall_accuracy = correct_count / len(all_responses) if len(all_responses) > 0 else 0

    # Run-level SEM: compute mean per run (eval file), then SEM across runs
    # This measures uncertainty due to model stochasticity
    n_runs = len(eval_files)
    n_questions = len(responses_by_question)

    # Compute accuracy and token mean for each run
    acc_per_run = []
    tok_per_run = []

    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)
            # Accuracy for this run
            correct = sum(1 for o in data["outputs"]
                         if (o.get("answer_correct") if o.get("answer_correct") is not None
                             else str(o.get("extracted_answer", "")) == str(o.get("ground_truth_answer"))))
            run_acc = correct / len(data["outputs"]) if data["outputs"] else 0
            acc_per_run.append(run_acc)

            # Mean tokens for this run
            tokens = [o.get("num_cot_tokens", 0) if o.get("num_cot_tokens", 0) > 0
                     else len(o.get("generated_output", "").split()) for o in data["outputs"]]
            run_tok = np.mean(tokens) if tokens else 0
            tok_per_run.append(run_tok)

    # Compute mean and SEM using run-level approach
    mean_accuracy = np.mean(acc_per_run) if acc_per_run else 0
    accuracy_sem = np.std(acc_per_run, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0

    overall_avg_tokens = np.mean(tok_per_run) if tok_per_run else 0
    tokens_sem = np.std(tok_per_run, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0

    # Build combined output
    combined_data = {
        "config": first_config,
        "checkpoint": first_checkpoint,
        "num_eval_runs": n_runs,
        "num_unique_questions": n_questions,
        "total_responses": len(all_responses),
        "run_level_statistics": {
            "mean_accuracy": float(mean_accuracy),
            "sem_accuracy": float(accuracy_sem),
            "mean_token_length": float(overall_avg_tokens),
            "sem_token_length": float(tokens_sem),
            "95_ci_accuracy": [
                float(max(0, mean_accuracy - 1.96 * accuracy_sem)),
                float(min(1, mean_accuracy + 1.96 * accuracy_sem))
            ],
            "95_ci_token_length": [
                float(overall_avg_tokens - 1.96 * tokens_sem),
                float(overall_avg_tokens + 1.96 * tokens_sem)
            ],
            "n_questions": n_questions,
            "n_runs": n_runs,
            "methodology_note": "SEM = std(run_means) / sqrt(R), measures uncertainty due to model stochasticity"
        },
        "per_question_metrics": question_metrics
    }

    # Save combined data
    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"\nCombined eval saved to {output_path}")
    print(f"\n=== Run-Level Statistics ===")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {accuracy_sem:.4f} (SEM)")
    print(f"Mean Tokens: {overall_avg_tokens:.1f} ± {tokens_sem:.1f} (SEM)")
    print(f"  N questions: {n_questions}, R runs: {n_runs}")
    print(f"  SEM = std(run_means) / sqrt({n_runs})")
    print(f"  (measures uncertainty due to model stochasticity)")
    print(f"\n=== Summary ===")
    print(f"Evaluation runs: {n_runs}")
    print(f"Questions: {n_questions}")
    print(f"Total responses: {len(all_responses)}")

    return combined_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine multiple eval outputs and compute SEM metrics")
    parser.add_argument("eval_dir", help="Directory containing eval_outputs_*.json files")
    parser.add_argument("--output", default="combined_eval_with_sem.json",
                        help="Output file path for combined results")

    args = parser.parse_args()

    combined_data = combine_and_analyze_evals(args.eval_dir, args.output)
