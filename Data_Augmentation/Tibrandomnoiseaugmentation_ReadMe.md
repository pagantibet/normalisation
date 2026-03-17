# Tibetan Text Data Augmentation — Random Noise Insertion

A Python script for augmenting Tibetan-script datasets by inserting random noise characters into text, useful for training more robust NLP and OCR models.

---

## How It Works

For each character in a given string, the script decides whether to insert a random Tibetan noise character immediately after it. The insertion probability is currently set as:

```
P(noise) = ratio / 100
```

This flat probability ensures that even short lines receive augmentation. (The original length-weighted formula `len(s) × (ratio / max_text_length) / 100` is preserved in the code as a comment if you wish to switch back to it.)

Noise characters are drawn from a built-in list of common Tibetan Unicode characters, including consonants, vowel signs, and punctuation marks.

---

## Requirements

- Python 3.6+
- No external dependencies

---

## Usage

### Command Line

```bash
python Tibrandomnoiseaugmentation.py <input_file.txt> [ratio] [max_text_length]
```

| Argument          | Description                                      | Default |
|-------------------|--------------------------------------------------|---------|
| `input_file`      | Path to the input `.txt` file (required)         | —       |
| `ratio`           | Noise intensity as a percentage (0–100)          | `10.0`  |
| `max_text_length` | Max text length in the dataset (legacy parameter)| `1000`  |

**Examples:**

```bash
# Basic usage with default ratio (10%)
python Tibrandomnoiseaugmentation.py my_corpus.txt

# Custom ratio of 15%
python Tibrandomnoiseaugmentation.py my_corpus.txt 15.0

# Custom ratio and max_text_length
python Tibrandomnoiseaugmentation.py my_corpus.txt 15.0 500
```

The output is written to a new file in the same directory, with `_noiseout` appended before the extension:

```
my_corpus.txt  →  my_corpus_noiseout.txt
```

---

### As a Python Module

```python
from Tibrandomnoiseaugmentation import TibetanAugmenter

augmenter = TibetanAugmenter(ratio=10.0, max_text_length=1000)

# Augment a single string
noisy_text = augmenter.insert_noise("བཀྲ་ཤིས་བདེ་ལེགས།")

# Augment a batch (with multiple augmentations per text)
texts = ["བཀྲ་ཤིས།", "བདེ་ལེགས།"]
augmented = augmenter.augment_batch(texts, num_augmentations=3)
```

---

## Parameters

### `TibetanAugmenter`

| Parameter         | Type    | Description                                               | Default |
|-------------------|---------|-----------------------------------------------------------|---------|
| `ratio`           | `float` | Percentage chance of inserting noise after each character | `10.0`  |
| `max_text_length` | `int`   | Used in the length-weighted formula (currently inactive)  | `1000`  |

### `insert_noise(text, noise_chars=None)`

| Parameter     | Type              | Description                                                      |
|---------------|-------------------|------------------------------------------------------------------|
| `text`        | `str`             | The input Tibetan string                                         |
| `noise_chars` | `List[str] \| None` | Custom list of noise characters; defaults to `TIBETAN_CHARACTERS` |

### `augment_batch(texts, num_augmentations=1)`

| Parameter           | Type        | Description                              |
|---------------------|-------------|------------------------------------------|
| `texts`             | `List[str]` | List of input strings                    |
| `num_augmentations` | `int`       | Number of noisy variants to produce per text |

---

## Noise Character Set

The default noise pool (`TIBETAN_CHARACTERS`) includes:

- **Consonants:** ཀ ཁ ག ང ཅ ཆ ཇ ཉ ཏ ཐ ད ན པ ཕ བ མ ཙ ཚ ཛ ཝ ཞ ཟ འ ཡ ར ལ ཤ ས ཧ ཨ
- **Punctuation:** ་ (tsheg) ། (shad) ༄ ༅
- **Vowel signs:** ི ུ ེ ོ ྀ ཱ
- **Sub-joined letters:** ྭ ྱ ྲ ླ

You can supply a custom character list via the `noise_chars` argument to `insert_noise()`.

---

## Notes

- Empty lines in the input file are preserved unchanged.
- The output file is UTF-8 encoded.
- Each run produces different results due to random sampling; set `random.seed()` before calling if you need reproducibility.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
