# Tibetan Text Normalisation — Seq2Seq Transformer Training

A training script for a character-level encoder-decoder Transformer that learns to normalise diplomatic Tibetan text into Standard Classical Tibetan. Unlike causal language models, this is a traditional seq2seq architecture that trains directly on aligned source–target sentence pairs.

This script was developed as part of [PaganTibet](https://www.pagantibet.com/)'s Normalisation workflow. For more information, see our [Normalisation README](https://github.com/pagantibet/normalisation/tree/main?tab=readme-ov-file).

## Overview

The model is trained on paired text (.txt) files where the source is noisy/diplomatic Tibetan and the target is the normalised form. The character-level vocabulary covers the full Tibetan Unicode block (U+0F00–U+0FFF), ensuring no `<unk>` tokens appear in Tibetan output.

Evaluation uses both standard metrics (CER, precision, recall, F1) and the SG-specific correction metrics described below.


## Requirements

- Python 3.6+
- [PyTorch](https://pytorch.org/)
- NumPy

```bash
pip install torch numpy
```

The script is designed to run in a conda environment (e.g. `pagantibenv`).


## Input Data Format

Two plain text files are required, one line per sentence, with source and target lines aligned:

```
train_source.txt    →  diplomatic / noisy Tibetan
train_target.txt    →  normalised Classical Tibetan
```

Both files must have exactly the same number of lines. The script will exit with a clear error if they do not match.

## Usage

```bash
python3 tibtrainencdecoder_witheval.py \
    --train_src train_source.txt \
    --train_tgt train_target.txt
```

To use the pre-built normalised Tibetan vocabulary (recommended):

```bash
python3 tibtrainencdecoder_witheval.py \
    --train_src train_source.txt \
    --train_tgt train_target.txt \
    --use_normalized_vocab
```

## Recommended GPU Configurations

### RTX 4060 (ultrafast)

```bash
python3 tibtrainencdecoder_witheval.py \
  --train_src augmented_1m_src.txt \
  --train_tgt augmented_1m_tgt.txt \
  --d_model 256 --num_layers 4 --nhead 8 \
  --batch_size 160 --gradient_accumulation_steps 2 \
  --lr 0.001 --dropout 0.1 --weight_decay 0.0001 \
  --early_stopping 3 --epochs 12 --save_every 4 \
  --test_split 0.005 --val_split 0.01 \
  --use_normalized_vocab --checkpoint_dir checkpoints
```

### RTX 6000

```bash
python3 tibtrainencdecoder_witheval.py \
  --train_src augmented_1m_src.txt \
  --train_tgt augmented_1m_tgt.txt \
  --d_model 512 --num_layers 6 --nhead 8 \
  --batch_size 512 --gradient_accumulation_steps 1 \
  --lr 0.001 --dropout 0.1 --weight_decay 0.0001 \
  --early_stopping 3 --epochs 12 --save_every 4 \
  --test_split 0.005 --val_split 0.01 \
  --use_normalized_vocab --checkpoint_dir checkpoints
```

Expected training time on an RTX 4090 with ~1M sentence pairs: approximately 3–4 hours.


## All Arguments

### Data

| Argument | Description | Default |
|---|---|---|
| `--train_src` | Training source file | *(required)* |
| `--train_tgt` | Training target file | *(required)* |
| `--val_src` | Validation source file | auto-split |
| `--val_tgt` | Validation target file | auto-split |
| `--test_src` | Test source file | auto-split |
| `--test_tgt` | Test target file | auto-split |
| `--val_split` | Fraction of data for validation | `0.15` |
| `--test_split` | Fraction of data for test | `0.15` |
| `--no_auto_split` | Disable automatic train/val/test split | off |
| `--use_normalized_vocab` | Use full normalised Tibetan Unicode vocab | off |

### Model Architecture

| Argument | Description | Default |
|---|---|---|
| `--d_model` | Transformer model dimension | `512` |
| `--nhead` | Number of attention heads | `8` |
| `--num_layers` | Number of encoder and decoder layers | `4` |
| `--dropout` | Dropout rate | `0.2` |

### Training

| Argument | Description | Default |
|---|---|---|
| `--epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size | `32` |
| `--lr` | Learning rate | `0.001` |
| `--weight_decay` | L2 regularisation | `0.0001` |
| `--gradient_accumulation_steps` | Accumulate gradients over N batches | `1` |
| `--early_stopping` | Stop after N epochs without improvement (0 = off) | `0` |
| `--beam_width` | Beam search width at evaluation | `5` |

### Checkpointing & Output

| Argument | Description | Default |
|---|---|---|
| `--save_model` | Path for best model checkpoint | `tibetan_model.pt` |
| `--checkpoint_dir` | Directory for periodic checkpoints | `checkpoints` |
| `--save_every` | Save a checkpoint every N epochs | `5` |
| `--results_file` | JSON results output path | `training_results.json` |
| `--report_file` | Plain-text report output path | `tibetan_report.txt` |


## Data Splits

If no separate validation or test files are provided, the script automatically splits the training data using the `--val_split` and `--test_split` ratios. To disable this and train on the full dataset without a validation set, pass `--no_auto_split`.


## Vocabulary

The vocabulary is built from the complete Tibetan Unicode block (U+0F00–U+0FFF), ASCII punctuation, and digits. This ensures no `<unk>` tokens appear for any Tibetan character in the output. With `--use_normalized_vocab`, the same comprehensive vocabulary is applied to both source and target. Without it, additional non-Tibetan characters found in the source data (e.g. Latin script) are also included.

Special tokens: `<pad>` (0), `<sos>` (1), `<eos>` (2), `<unk>` (3).

The vocabulary is sampled from up to 10,000 texts during building for speed on large datasets.


## Training Details

- **Loss:** Label smoothing (smoothing = 0.2) via KL divergence
- **Optimiser:** Adam (β₁ = 0.9, β₂ = 0.997)
- **Gradient clipping:** max norm 1.0
- **Weight initialisation:** Xavier uniform for all parameters with dim > 1
- **Decoding during training:** fast greedy decode; beam search used only for final test evaluation

## Evaluation Metrics

Final evaluation is run on the held-out test set using beam search on the best checkpoint (lowest validation loss).

### Standard metrics

| Metric | Description |
|---|---|
| CER | Character Error Rate (Levenshtein distance / reference length) |
| Precision | Proportion of predicted characters that are correct |
| Recall | Proportion of reference characters that are matched |
| F1 | Harmonic mean of precision and recall |

### SG correction metrics

These metrics measure how well the model corrects actual errors relative to the source, rather than just measuring similarity to the reference.

| Metric | Formula | Description |
|---|---|---|
| Correcting Recall (CR) | Ccorr / Etotal | Proportion of true errors in the source that were successfully fixed |
| Correcting Precision (CP) | Ccorr / Eident | Proportion of changes made by the model that were correct |

Where:
- **Ccorr** = positions where source ≠ target AND hypothesis = target (correctly fixed)
- **Etotal** = positions where source ≠ target (total real errors)
- **Eident** = positions where source ≠ hypothesis (total changes the model made)


## Output Files

| File | Contents |
|---|---|
| `tibetan_model.pt` (or `--save_model`) | Best model checkpoint (includes model weights, vocabularies, and training args) |
| `checkpoints/` | Periodic epoch checkpoints |
| `tibetan_report.txt` (or `--report_file`) | Plain-text training report with system info, configuration, metrics, and 10 random example predictions |
| `training_results.json` (or `--results_file`) | Full results in JSON including all metrics, training info, and model config |


## Notes

- GPU is used automatically if available; falls back to CPU.
- Source and target files must have the same number of lines — the script will exit with a clear error message if not.
- Sequences are truncated to a maximum length of 100 characters.
- The best model (lowest validation loss) is reloaded for final test evaluation.


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
