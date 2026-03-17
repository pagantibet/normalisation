# Tibetan NLP Data Augmentation — Word & Syllable Swapping

A Python script for augmenting Tibetan text datasets by randomly swapping adjacent words or syllables. Inspired by the [nlpaug](https://github.com/makcedward/nlpaug) library, it is designed to generate noisy, unnormalised text from clean Classical Tibetan — approximating the kind of variation found in diplomatic (manuscript) sources.

---

## Background

The script generates new sentence pairs from normalised Classical Tibetan source text, producing output that resembles less-standardised, "diplomatic" transcriptions. It does not reproduce abbreviations, but the swapping augmentation introduces realistic positional variation for use in NLP model training.

Two modes are supported depending on whether the input text is whitespace-segmented or unsegmented Tibetan:

| Mode | Unit swapped |
|---|---|
| `segmented` | Whole whitespace-separated words |
| `nonsegmented` | Individual syllables (split on the tsheg ་) |

---

## Requirements

- Python 3.6+
- No external dependencies

---

## Usage

```bash
python3 nlpaugtib.py --input <input_file.txt> --type <segmented|nonsegmented> [--aug_prob FLOAT]
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--input` | Path to the input `.txt` file | *(required)* |
| `--type` | Text type: `segmented` or `nonsegmented` | *(required)* |
| `--aug_prob` | Probability of swapping each adjacent pair | `0.05` |

### Examples

```bash
# Segmented (whitespace-tokenised) input
python3 nlpaugtib.py --input unsegACTib-tok_lines_ocr_noise_500k.txt --type segmented

# Unsegmented (raw Tibetan script) input
python3 nlpaugtib.py --input unsegACTib_lines_ocr_noise_500k.txt --type nonsegmented

# Custom swap probability
python3 nlpaugtib.py --input my_corpus.txt --type segmented --aug_prob 0.1
```

The output is written to the same directory as the input, with `_augmented` appended before the extension:

```
my_corpus.txt  →  my_corpus_augmented.txt
```

---

## How It Works

### Segmented mode
The line is split on whitespace into words. Adjacent word pairs are then considered one by one from left to right — each pair is swapped with probability `aug_prob`. When a swap occurs, the cursor skips forward by two positions to avoid overlapping swaps.

### Nonsegmented mode
The line is first split into syllables by treating each tsheg (་) as a syllable boundary. The same adjacent-swap logic is then applied to the syllable list, and the result is rejoined without spaces.

### `<utt>` tag handling
If a line contains the `<utt>` utterance boundary marker, it is stripped before augmentation and reattached at the end of the output line, so corpus structure is preserved.

### Empty lines
Empty lines are passed through unchanged.

---

## Notes

- Each run produces different results due to random sampling; call `random.seed()` before running if you need reproducible output.
- The output file is UTF-8 encoded.
- `aug_prob` of `0.05` means roughly 1 in 20 adjacent pairs will be swapped — a light perturbation. Increase it for more aggressive augmentation.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
