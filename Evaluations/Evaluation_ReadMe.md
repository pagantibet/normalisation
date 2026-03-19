# Tibetan Text Normalisation - Evaluation Script Usage Guide

An evaluation script, which outputs character error rate (CER), precision, recall, F1-scores, correction precision (CP), and correction recall (CR) to assesses how well a model (or set of predictions) corrects errors in Tibetan script.

This was developed as part of [PaganTibet](https://www.pagantibet.com/)'s Normalisation workflow. For more information, see our [Normalisation README](https://github.com/pagantibet/normalisation/tree/main?tab=readme-ov-file). 

The seq2seq and KenLM models used in Meelen & Griffiths (2026) can be found on the [PaganTibet HuggingFace](https://huggingface.co/datasets/pagantibet/Tibetan-abbreviation-dictionary).

## Overview

The `evaluate_model.py` script supports two modes of evaluation:

1. **Model mode**: Load and run a trained neural model for inference and evaluation
2. **Predictions mode**: Evaluate pre-generated predictions from any inference method (for more on inference methods see the [Inference README](https://github.com/pagantibet/normalisation/blob/main/Inference/FlexibleInference_ReadMe.md))

This allows you to evaluate all different inference modes, including:
- Pure neural seq2seq
- Neural seq2seq + KenLM ranking
- Rule-based preprocessing/postprocessing
- Any combination of the above
- Purely rule-based approaches (no neural model)

## Evaluation Script Variants

Two versions of the evaluation script are available:

- `evaluate_model.py` — the standard script described in this README
- `evaluate-model-withCIs.py` — an extended version that additionally computes **95% bootstrap confidence intervals (CI)** for all metrics

### When to use `evaluate-model-withCIs.py`?

Use the CI version when you need statistically robust results, for example:

- Comparing two inference methods and wanting to know if the difference is significant
- Working with a smaller test set where point estimates alone may be misleading

It accepts all the same arguments as `evaluate_model.py`, plus one additional option:

- `--bootstrap_n`: Number of bootstrap resampling iterations (default: 1000; set to `0` to disable)

Output metrics will include confidence intervals, e.g.:
```bash
Character Error Rate (CER): 4.21%  (95% CI: 3.87–4.55%)
F1 Score: 93.10%  (95% CI: 92.64–93.56%)
```

The text report is saved as `_CIreport.txt` rather than `_report.txt`.

### When to use `evaluate_model.py`?

- Quick iterative runs during development where CIs aren't needed
- Large test sets where bootstrap resampling would add noticeable runtime
- Any situation where a point estimate is sufficient

## Running on an HPC Cluster (SLURM)

A [SLURM batch script](https://github.com/pagantibet/normalisation/blob/main/Evaluations/evaluate-model.sh) is provided for running evaluation on an HPC cluster. To submit the job:
```bash
sbatch evaluate_model.sh
```

The provided script requests an RTX 4090 GPU with 16 CPUs and a 6-hour time limit, and activates the `pagantibenv` conda environment before evaluation. 

It is pre-configured for predictions mode, but a commented-out example for model mode is also included. Note that predictions mode does not require a GPU. Adjust the `#SBATCH` directives, mode, and file paths as needed for your cluster and dataset.
  

## Key Features

**Automatic Output File Generation**: The script automatically:
- Creates the `evaluation-results/` directory if it doesn't exist
- Generates output filenames from your predictions filename
- Example: `predictions_neural.txt` → `evaluation-results/predictions_neural_eval.json` + `evaluation-results/predictions_neural_eval_report.txt`

**No `--output`** flag needed for most use cases.

## Usage Examples

### Mode 1: Evaluating a Neural Model Directly


**1. Simplest version** - outputs to `evaluation-results/evaluation_results.json` automatically:
```bash
python evaluate_model.py \
  --mode model \
  --model path/to/model.pth \
  --test_src test_source.txt \
  --test_tgt test_target.txt
```

**2. Or specify a custom output name** (still goes to `evaluation-results/`):
```bash
python evaluate_model.py \
  --mode model \
  --model path/to/model.pth \
  --test_src test_source.txt \
  --test_tgt test_target.txt \
  --output my_model_results.json
```

### Mode 2: Evaluating Pre-generated Predictions

**1. Minimal** - Neural Only (automatic output):

```bash
# Output automatically goes to evaluation-results/predictions_neural_eval.json
python evaluate_model.py \
  --mode predictions \
  --predictions predictions_neural.txt \
  --test_src test_source.txt \
  --test_tgt test_target.txt \
  --inference_method "neural"
```

**2. Neural + KenLM** (with model metadata):

```bash
# Output automatically goes to evaluation-results/predictions_neural_lm_eval.json
python evaluate_model.py \
  --mode predictions \
  --predictions predictions_neural_lm.txt \
  --test_src test_source.txt \
  --test_tgt test_target.txt \
  --inference_method "neural+lm" \
  --uses_neural_model \
  --uses_kenlm \
  --model path/to/model.pth \
  --kenlm_path path/to/kenlm.arpa \
  --description "Seq2seq model with KenLM n-gram ranking for candidate selection"
```

**3. Rules + Neural + LM** (full pipeline):

```bash
# Output automatically goes to evaluation-results/predictions_rules_neural_lm_eval.json
python evaluate_model.py \
  --mode predictions \
  --predictions predictions_rules_neural_lm.txt \
  --test_src test_source.txt \
  --test_tgt test_target.txt \
  --inference_method "rules+neural+lm" \
  --uses_neural_model \
  --uses_kenlm \
  --uses_preprocessing \
  --model path/to/model.pth \
  --kenlm_path path/to/kenlm.arpa \
  --description "Rule-based preprocessing, then seq2seq with KenLM ranking"
```

**4. Purely Rule-Based** (no neural model):

```bash
# Output automatically goes to evaluation-results/predictions_rules_eval.json
python evaluate_model.py \
  --mode predictions \
  --predictions predictions_rules.txt \
  --test_src test_source.txt \
  --test_tgt test_target.txt \
  --inference_method "rules" \
  --description "Purely rule-based normalization without neural model"
```

## Command-Line Arguments

### Required Arguments

- `--test_src`: Path to test source file (one line per example)
- `--test_tgt`: Path to test target file (one line per example)

### Mode Selection

- `--mode`: Choose between `model` or `predictions` (default: `model`)

### Model Mode Arguments

- `--model`: Path to saved model checkpoint (.pth file)
- `--batch_size`: Batch size for evaluation (default: 128)

### Predictions Mode Arguments

- `--predictions`: Path to predictions file (one prediction per line)

### Inference Method Metadata (Optional, for Predictions Mode)

These flags help document what methods were used:

- `--inference_method`: Short description (e.g., "seq2seq", "seq2seq+kenlm", "rules_only")
- `--uses_neural_model`: Flag to indicate neural model was used
- `--uses_kenlm`: Flag to indicate KenLM was used
- `--uses_preprocessing`: Flag to indicate rule-based preprocessing
- `--uses_postprocessing`: Flag to indicate rule-based postprocessing
- `--model`: Path to neural model (if used)
- `--kenlm_path`: Path to KenLM model (if used)
- `--description`: Free-text description of the approach

### Common Arguments

- `--max_samples`: Maximum number of samples to evaluate (default: all)
- `--output`: Output JSON file name (optional - auto-generated from predictions filename if not specified)
- `--output_dir`: Output directory for all results (default: `evaluation-results/`)

**Note**: If `--output` is not specified:
- In predictions mode: filename is auto-generated from predictions file
  - Example: `predictions_neural.txt` → `evaluation-results/predictions_neural_eval.json`
- In model mode: defaults to `evaluation-results/evaluation_results.json`

## Output Files

The script automatically generates two output files in the `evaluation-results/` directory:

1. **JSON file** (e.g., `predictions_neural_eval.json`): Machine-readable results including:
   - Evaluation mode and configuration
   - Model/inference method information
   - All metrics (CER, precision, recall, F1, correcting precision/recall)
   - Timing information

2. **Text report** (e.g., `predictions_neural_eval_report.txt`): Human-readable report with:
   - Evaluation information
   - Inference method details
   - Neural model parameters (if applicable)
   - All evaluation metrics
   - Error correction statistics
   - Timing information
   - 10 random example predictions

### Output File Naming

**Predictions mode:**
- Input: `predictions_neural.txt`
- Output: `evaluation-results/predictions_neural_eval.json` + `evaluation-results/predictions_neural_eval_report.txt`

**Model mode:**
- Default: `evaluation-results/evaluation_results.json` + `evaluation-results/evaluation_results_report.txt`

**Custom output directory:**
```bash
--output_dir my-results/
# Creates: my-results/predictions_neural_eval.json
```

## Metrics Calculated

The script calculates:

- **Character Error Rate (CER)**: Edit distance normalised by reference length
- **Precision**: Correct characters / predicted characters
- **Recall**: Correct characters / reference characters
- **F1 Score**: Harmonic mean of precision and recall
- **Correcting Recall (CR)**: Correctly corrected errors / total errors
- **Correcting Precision (CP)**: Correctly corrected errors / identified errors

## Tips for Comparing Results

- **Automatic organisation**: All evaluation results are automatically saved to `evaluation-results/` with descriptive names based on your predictions files
- **Name your predictions files clearly** for automatic organisation:
   ```
   predictions_neural.txt          → evaluation-results/predictions_neural_eval.json
   predictions_neural_lm.txt       → evaluation-results/predictions_neural_lm_eval.json
   predictions_rules_neural_lm.txt → evaluation-results/predictions_rules_neural_lm_eval.json
   predictions_rules.txt           → evaluation-results/predictions_rules_eval.json
   ```
- **Use metadata flags** to document what was used in each experiment (appears in reports)
- **The JSON output** can be easily parsed to create comparison tables or plots
- **The text reports** provide detailed information for documentation and analysis

## Notes

- The predictions file must have the same number of lines as the test source file
- Each line in the predictions file should contain the prediction for the corresponding line in the test source
- For methods without a neural model, simply omit the `--model` flag and don't use the `--uses_neural_model` flag
- The script will still extract and display neural model parameters if a model is provided, even in predictions mode

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
