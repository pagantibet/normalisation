# Evaluation Script Usage Guide

## Overview

The updated `evaluate_model.py` script now supports two modes of evaluation:

1. **Model mode**: Load and run a trained neural model for inference and evaluation
2. **Predictions mode**: Evaluate pre-generated predictions from any inference method

This allows you to evaluate all your different inference modes, including:
- Pure neural seq2seq
- Neural seq2seq + KenLM ranking
- Rule-based preprocessing/postprocessing
- Any combination of the above
- Purely rule-based approaches (no neural model)

## Key Features

✨ **Automatic Output File Generation**: The script now automatically:
- Creates the `evaluation-results/` directory if it doesn't exist
- Generates output filenames from your predictions filename
- Example: `predictions_neural.txt` → `evaluation-results/predictions_neural_eval.json` + `evaluation-results/predictions_neural_eval_report.txt`

💡 **No --output flag needed** for most use cases!

## Usage Examples

### Mode 1: Evaluating a Neural Model Directly

```bash
# Simplest version - outputs to evaluation-results/evaluation_results.json automatically
python evaluate_model.py \
  --mode model \
  --model path/to/model.pth \
  --test_src test_source.txt \
  --test_tgt test_target.txt

# Or specify a custom output name (still goes to evaluation-results/)
python evaluate_model.py \
  --mode model \
  --model path/to/model.pth \
  --test_src test_source.txt \
  --test_tgt test_target.txt \
  --output my_model_results.json
```

### Mode 2: Evaluating Pre-generated Predictions

#### Example 2a: Minimal - Neural Only (Automatic Output)

```bash
# Output automatically goes to evaluation-results/predictions_neural_eval.json
python evaluate_model.py \
  --mode predictions \
  --predictions predictions_neural.txt \
  --test_src test_source.txt \
  --test_tgt test_target.txt \
  --inference_method "neural"
```

#### Example 2b: Neural + KenLM (with model metadata)

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

#### Example 2c: Rules + Neural + LM (Full Pipeline)

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

#### Example 2d: Purely Rule-Based (No Neural Model)

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

## Supported Inference Methods

Based on your `tibetan-inference-flexible.py` script, these are the 6 supported modes:

1. **`rules`** - Rule-based only (dictionary + punctuation rules)
2. **`neural`** - Seq2seq only (neural model)
3. **`neural+lm`** - Seq2seq + KenLM (neural + language model)
4. **`neural+lm+rules`** - Neural + LM + Rules (neural + LM, then rules postprocessing)
5. **`rules+neural+lm`** - Rules + Neural + LM (rules preprocessing, then neural + LM)
6. **`rules+neural`** - Rules + Neural (rules preprocessing, then neural)

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

- **Character Error Rate (CER)**: Edit distance normalized by reference length
- **Precision**: Correct characters / predicted characters
- **Recall**: Correct characters / reference characters
- **F1 Score**: Harmonic mean of precision and recall
- **Correcting Recall (CR)**: Correctly corrected errors / total errors
- **Correcting Precision (CP)**: Correctly corrected errors / identified errors

## Tips for Comparing Results

1. **Automatic organization**: All evaluation results are automatically saved to `evaluation-results/` with descriptive names based on your predictions files

2. **Name your predictions files clearly** for automatic organization:
   ```
   predictions_neural.txt          → evaluation-results/predictions_neural_eval.json
   predictions_neural_lm.txt       → evaluation-results/predictions_neural_lm_eval.json
   predictions_rules_neural_lm.txt → evaluation-results/predictions_rules_neural_lm_eval.json
   predictions_rules.txt           → evaluation-results/predictions_rules_eval.json
   ```

3. **Use metadata flags** to document what was used in each experiment (appears in reports)

4. **The JSON output** can be easily parsed to create comparison tables or plots

5. **The text reports** provide detailed information for documentation and analysis

## Workflow Example

1. Train your seq2seq model (generates `model.pth`)

2. Run your inference script to generate predictions for different modes:
   ```bash
   # These create predictions_neural.txt, predictions_neural_lm.txt, etc.
   python tibetan-inference-flexible.py --mode neural --model_path model.pth \
     --input_file test_src.txt --output_file predictions_neural.txt
   
   python tibetan-inference-flexible.py --mode neural+lm --model_path model.pth \
     --kenlm_path lm.arpa --input_file test_src.txt --output_file predictions_neural_lm.txt
   
   python tibetan-inference-flexible.py --mode rules --rules_dict abbrev.txt \
     --input_file test_src.txt --output_file predictions_rules.txt
   ```

3. Evaluate each set of predictions (outputs automatically saved to `evaluation-results/`):
   ```bash
   # Evaluate neural only - creates evaluation-results/predictions_neural_eval.json
   python evaluate_model.py \
     --mode predictions \
     --predictions predictions_neural.txt \
     --test_src test_src.txt \
     --test_tgt test_tgt.txt \
     --inference_method "neural" \
     --uses_neural_model \
     --model model.pth
   
   # Evaluate neural + KenLM - creates evaluation-results/predictions_neural_lm_eval.json
   python evaluate_model.py \
     --mode predictions \
     --predictions predictions_neural_lm.txt \
     --test_src test_src.txt \
     --test_tgt test_tgt.txt \
     --inference_method "neural+lm" \
     --uses_neural_model \
     --uses_kenlm \
     --model model.pth \
     --kenlm_path lm.arpa
   
   # Evaluate rules only - creates evaluation-results/predictions_rules_eval.json
   python evaluate_model.py \
     --mode predictions \
     --predictions predictions_rules.txt \
     --test_src test_src.txt \
     --test_tgt test_tgt.txt \
     --inference_method "rules"
   ```

4. Compare the results: all JSON and text reports are in `evaluation-results/` folder

## Notes

- The predictions file must have the same number of lines as the test source file
- Each line in the predictions file should contain the prediction for the corresponding line in the test source
- For methods without a neural model, simply omit the `--model` flag and don't use the `--uses_neural_model` flag
- The script will still extract and display neural model parameters if a model is provided, even in predictions mode

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
