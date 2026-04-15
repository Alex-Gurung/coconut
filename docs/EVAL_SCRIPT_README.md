# Evaluation Scripts

Primary workflow note:

- For GSM-Hard with `Qwen/Qwen3-4B-Instruct-2507`, start with
  `experiments/gsm_hard_qwen3_4b/README.md`.
- This document covers the lower-level multi-run eval helpers.

Two scripts for managing multiple evaluation runs and computing statistical metrics:

## `scripts/eval/run_eval_n_times.py`

Run evaluation N times with different seeds for a checkpoint or HuggingFace model.

### Usage

```bash
python scripts/eval/run_eval_n_times.py <checkpoint_or_model_id> <num_runs> [options]
```

### Examples

**Evaluate a local checkpoint 5 times:**
```bash
python scripts/eval/run_eval_n_times.py checkpoints/qwen-coconut-ff-v2/checkpoint_13 5
```

**Evaluate a HuggingFace model 3 times with custom config:**
```bash
python scripts/eval/run_eval_n_times.py Qwen/Qwen3-4B-Instruct-2507 3 --config-base experiments/gsm_hard_qwen3_4b/eval.yaml
```

**Evaluate with custom validation data:**
```bash
python scripts/eval/run_eval_n_times.py checkpoints/my-model 4 --val-path data/my_test.json --name my_eval
```

### Options

- `--config-base`: Base config file to use as template (default: `experiments/gsm_hard_qwen3_4b/eval.yaml`)
- `--name`: Name for the eval run (default: derived from checkpoint/model name)
- `--val-path`: Path to validation/test data
- `--num-gpus`: Number of GPUs to use with torchrun (default: 4)

### Output

Each run creates a per-run eval directory containing `eval_outputs.json` with:
- Overall statistics (accuracy, token counts)
- Per-sample results (question, answer, extracted answer, CoT, tokens)
- Full config used for evaluation

---

## `scripts/eval/combine_evals.py`

Combine multiple eval outputs from different runs and compute SEM (Standard Error of the Mean) statistics.

### Usage

```bash
python scripts/eval/combine_evals.py <eval_dir> [--output output.json]
```

### Examples

**Combine all eval outputs under a run directory:**
```bash
python scripts/eval/combine_evals.py checkpoints/qwen3-4b-coconut-gsm-hard
```

**Combine with custom output path:**
```bash
python scripts/eval/combine_evals.py checkpoints/qwen3-4b-coconut-gsm-hard --output results/combined_eval.json
```

### Output Format

The combined JSON file includes:

```json
{
  "config": {...},                          // Config from first run
  "checkpoint": "path/to/checkpoint",
  "num_eval_runs": 5,
  "num_unique_questions": 62,
  "total_responses": 310,
  "overall_statistics": {
    "accuracy": 0.5935,                     // Overall accuracy across all responses
    "correct_count": 184,
    "total_samples": 310,
    "mean_token_length": 486.2,
    "std_token_length": 152.5,
    "sem_token_length": 8.7,                // Standard error of the mean
    "95_ci_token_length": [469.1, 503.3]   // 95% confidence interval
  },
  "per_question_statistics": {
    "mean_accuracy": 0.5935,                // Mean accuracy per question (across runs)
    "std_accuracy": 0.0412,
    "sem_accuracy": 0.0026,                 // SEM: std / sqrt(n_questions)
    "95_ci_accuracy": [0.5884, 0.5986]
  },
  "per_question_metrics": [
    {
      "question": "You are tasked with...",
      "num_responses": 5,                   // Number of times this question was evaluated
      "accuracy": 0.6,
      "correct_answers": 3,
      "avg_token_length": 512.4
    }
  ]
}
```

### Key Metrics

1. **Overall Accuracy**: Proportion of correct answers across all responses
   - Useful for: Overall model performance

2. **Per-Question Accuracy + SEM**: Mean accuracy per question with statistical uncertainty
   - Useful for: Robust estimate that accounts for question difficulty variance
   - SEM = std_accuracy / sqrt(num_unique_questions)

3. **Token Length Statistics**: Mean output length with 95% confidence interval
   - Useful for: Understanding computational costs and output verbosity

### Interpreting Results

- **SEM (Standard Error of Mean)**: Smaller values = more confident estimate
  - Use with 95% CI: 95% of the true mean lies in this range
  - For N questions, SEM decreases as sqrt(N) increases

- **Per-question accuracy**: Each question evaluated multiple times (one per run)
  - Accounts for randomness in model behavior
  - Better estimate than single-run accuracy

---

## Workflow Example

```bash
# 1. Run evaluation 5 times with different seeds
python scripts/eval/run_eval_n_times.py checkpoints/gsm_hard/qwen3-4b-coconut-gsm-hard/checkpoint_20 5 --name my_model_eval

# 2. Combine results and compute SEM metrics
python scripts/eval/combine_evals.py checkpoints/gsm_hard/qwen3-4b-coconut-gsm-hard --output results/my_model_eval_combined.json

# 3. View the results
cat results/my_model_eval_combined.json | python -m json.tool
```

This gives you:
- Robust accuracy estimates with confidence intervals
- Token cost estimates
- Per-question analysis for debugging
