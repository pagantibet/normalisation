# Tibetan Text Data Augmentation - Rule-based Diplomatic Transformations

A Python script for performing rule-based data augmentation on Tibetan text (.txt) files, designed specifically for improving machine translation and NLP models through controlled noise injection.

This script was developed as part of [PaganTibet](https://www.pagantibet.com/)'s Normalisation workflow. For more information, see our [Normalisation README](https://github.com/pagantibet/normalisation/tree/main?tab=readme-ov-file).

## Key Features

### 1. Bidirectional Character Replacements

| Original | Replacement | 
|----------|-------------|
| འ ↔ བ | Vowel carrier / consonant confusion | 
| ས ↔ པ | Consonant confusion | 
|  ེ ↔  &#xFEFF;ི | Vowel marker confusion | 
| ལ ↔ ཡ | Semi-vowel/consonant | 
| ཕ ↔ མ | Aspirated/nasal consonants | 
| ག ↔ ང | Stop/nasal consonants | 
| ཀ ↔ ག | Aspirated/plain stops | 
| ཅ ↔ ཆ | Aspirated variants |
| ན ↔ མ | Nasal consonants |
| ཐ ↔ ད | Aspirated/plain stops |
| ཐ ↔ ཏ | Aspirated variants |
| ཕ ↔ པ | Aspirated/plain stops |
| ཤ ↔ ས | Sibilant variants |
| ཞ ↔ ཟ | Voiced variants |
| ྱ ↔ &#xFEFF;ྲ| Subjoined letters | ya-btags/ra-btags |
| ྀ ↔ &#xFEFF;ི | Reverse/normal gigu | Unicode variants |
| དྱ ↔ གྱ | Digraphs | Two-character sequences |

**Total: 34 bidirectional rules** (17 pairs)

### 2. Syllable-Level Operations

- **Random Deletion**: Removes entire syllables (3% by default)
- **Random Duplication**: Duplicates syllables (3% by default)
- Syllables detected by Tibetan punctuation: ་ (tsheg), ། (shad), ༎ (double shad), etc.

## Usage

### Basic Usage
```bash
python3 tibrule_augmentation.py input.txt
```

### Custom Ratios
```bash
# Conservative (10% character replacement)
python3 tibrule_augmentation.py input.txt --char-ratio 0.1

# Aggressive (30% character replacement)
python3 tibrule_augmentation.py input.txt --char-ratio 0.3

# Custom syllable operations
python3 tibrule_augmentation.py input.txt --char-ratio 0.2 --syllable-ratio 0.05
```

### Parameters
- `--char-ratio`: Character replacement ratio (0.0-1.0, default: 0.2)
- `--syllable-ratio`: Syllable deletion/addition ratio (0.0-1.0, default: 0.03)
- `--seed`: Random seed for reproducibility (default: 42)

## Augmentation Ratio Recommendations

### For Sentence Pair Augmentation (MT, Parallel Corpora)

| Ratio | Level | Use Case | Characteristics |
|-------|-------|----------|----------------|
| **10-15%** | Conservative | High-quality parallel data | Minimal noise, maintains high similarity |
| **20-25%** | Moderate | General augmentation | **RECOMMENDED** - Balanced diversity |
| **30-40%** | Aggressive | Large-scale augmentation | More diversity, may reduce alignment |
| **50%+** | Very Aggressive | ⚠️ Not recommended | Risk of breaking semantic alignment |

**Why 50% is Too Much for Sentence Pairs:**

1. **Semantic Drift**: With 50% of characters changed, sentences can become semantically different from their pairs
2. **Alignment Quality**: Translation models rely on consistent character-level patterns; too much noise breaks this
3. **Diminishing Returns**: Generally 15-25% augmentation provides optimal diversity without degradation

### Recommended Settings by Task

#### Machine Translation
```bash
# Training set augmentation
python tibrule_augmentation.py train.txt --char-ratio 0.2 --syllable-ratio 0.03

# Validation set (lighter augmentation)
python tibrule_augmentation.py val.txt --char-ratio 0.1 --syllable-ratio 0.01
```

#### OCR Error Simulation
```bash
# Simulate realistic OCR errors
python tibrule_augmentation.py clean_text.txt --char-ratio 0.15 --syllable-ratio 0.02
```

#### Robust Model Training
```bash
# More aggressive for robustness
python tibrule_augmentation.py robust_train.txt --char-ratio 0.3 --syllable-ratio 0.05
```

## How It Works

### Bidirectional Design
All replacements work in both directions to prevent bias. For example:
- འ → བ (50% of འ become བ)
- བ → འ (50% of བ become འ)

This ensures the augmented corpus remains balanced.

### Syllable Detection
Tibetan syllables are detected by their ending punctuation:
- ་ (U+0F0B) - tsheg (main syllable marker)
- ། (U+0F0D) - shad (clause marker)
- Plus other Tibetan punctuation marks

### Random Sampling
- Uses seeded random number generation for reproducibility
- Each character type is processed independently
- Syllable operations randomly choose deletion OR duplication per run

## Output

- **Filename**: `[input_name]_ruleout.txt`
- **Line count**: Preserved (same number of lines as input)
- **Encoding**: UTF-8
- **Examples**: Prints 5 before/after examples to terminal

### Example Output

```
Original:  བོད་ཀྱི་སྐད་ཡིག་ནི་བོད་པའི་མི་རིགས་ཀྱི་སྐད་ཡིག་ཡིན།
Replaced:  བོད་སྐད་ཡྀག་ནི་བོད་པའི་མི་རིགཔ་ཀྱི་སྐད་ཡིག་ཡིན།
          (20% character replacement + 3% syllable operations)
```

## Performance Considerations

- **Speed**: Processes ~1000 lines/second on typical hardware
- **Memory**: Loads entire file into memory (suitable for files up to several GB)
- **Deterministic**: Same seed produces identical results

## Error Handling

- Validates ratio parameters (must be 0-1)
- Checks file existence before processing
- Handles empty lines and files gracefully
- UTF-8 encoding errors are caught and reported

## Support
If you encounter issues: 
1. Check the help: `python tibetan_augmentation.py --help`
2. Test with small files first to validate behavior
3. Adjust ratios based on specific task and data quality

## Future Enhancements

Potential additions (not yet implemented):
- Support for batch processing multiple files
- JSON configuration files for custom rule sets
- Statistical reporting of augmentation patterns
- Parallel processing for very large corpora
- Integration with common ML frameworks

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
