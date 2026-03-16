# Tibetan Dictionary Augmentation

A Python script for processing Tibetan text files by adding abbreviation expansions from a dictionary to randomly selected lines.

## Overview

This script takes a Tibetan text file (approximately 25,000 lines) and a dictionary of abbreviations with their expansions, then randomly selects 10,000 lines and creates two output files:

1. **Input with abbreviations** - The original text with abbreviated forms added to randomly selected lines
2. **Input with expansions** - The original text with expanded forms added to the same lines

Both output files maintain the original line order.

## Features

- Removes `[]` brackets from dictionary entries automatically
- Randomly selects 10,000 lines from the input file (not sequential)
- Maintains original line order in output
- Supports both tokenized (space-separated) and non-tokenized text
- Optional random seed for reproducible results
- Detailed progress reporting and error handling

## Requirements

- Python 3.6 or higher
- No external dependencies (uses only standard library)

## Input File Formats

### Dictionary File Format

The dictionary file should have two columns:
- Column 1: Abbreviated form (in brackets)
- Column 2: Expanded form (in brackets)

Columns can be separated by tabs or multiple spaces.

**Example:**
```
[སྟོངའཱི་]  [སྟོང་པའི་]
[སྟོངག་] [སྟོང་ཕྲག]
[སྟོངྒ་] [སྟོང་ཕྲག]
[ཏུལོ་]  [དུ་གསོལ]
[སྙིཾ༹ན་]   [སྙིང་ཚན་]
```

### Text File Formats

**Tokenized (space-separated syllables):**
```
ར་ འདྲེན་པ་ ཡོངས་ ཀྱི་ ཡེ་ཤེས་ སྣང་བ འི་ རང་གཟུགས་
བྱ་བ འི་ སྒོ་གསུམ་པ་ གཟུགས་ཅན་ ལ ས་ མཐའ་དག་
```

**Non-tokenized (continuous text):**
```
ཡེ་ཤེས་གཟིགས་པའི་འབབ་སྟེགས་སུ་ཞིག་རྒྱུད་དུ་བཅས་ཙམ་
བས་ལེགས་གསུངས་ཐེག་གསུམ་བསྟན་པའི་ཟིལ་དངར་ལོངས་
```

## Usage

### Basic Usage (Tokenized Text)

```bash
python3 dictionary-augmentation.py input.txt abbreviation-dictionary.txt
```

This creates two files:
- `input_abbrev.txt` - with abbreviations added (with space before them)
- `input_dictaug.txt` - with expansions added (with space before them)

### Non-Tokenized Text

```bash
python3 dictionary-augmentation.py input.txt abbreviation-dictionary.txt --non-tokenized
```

This creates two files:
- `input_abbrev.txt` - with abbreviations glued directly
- `input_dictaug.txt` - with expansions glued directly

### With Random Seed (for reproducibility)

```bash
python3 dictionary-augmentation.py input.txt dictionary.txt 42
```

Or for non-tokenized with seed:

```bash
python3 dictionary-augmentation.py input.txt abbreviation-dictionary.txt --non-tokenized 42
```

## Command Line Arguments

```
python3 dictionary-augmentation.py <input_text_file> <dictionary_file> [--non-tokenized] [random_seed]
```

**Arguments:**
- `input_text_file` (required): Path to your Tibetan text file (~25k lines)
- `dictionary_file` (required): Path to your abbreviation dictionary file (~10k entries)
- `--non-tokenized` (optional): Use this flag for non-tokenized input (no space before abbreviations/expansions)
- `random_seed` (optional): Integer value for reproducible random selection

**Output:**
- Two output files are automatically created:
  - `<input_filename>_abbrev.txt` - Original text with abbreviations added
  - `<input_filename>_dictaug.txt` - Original text with expansions added
- For example: `input.txt` → `input_abbrev.txt` and `input_dictaug.txt`

## Examples

### Example 1: Tokenized Text

**Original input line:**
```
ར་ འདྲེན་པ་ ཡོངས་ ཀྱི་ ཡེ་ཤེས་
```

**Dictionary entry:**
```
[སྟོངའཱི་]  [སྟོང་པའི་]
```

**Output in _abbrev.txt:**
```
ར་ འདྲེན་པ་ ཡོངས་ ཀྱི་ ཡེ་ཤེས་ སྟོངའཱི་
```
(Note the space before the abbreviation)

**Output in _dictaug.txt:**
```
ར་ འདྲེན་པ་ ཡོངས་ ཀྱི་ ཡེ་ཤེས་ སྟོང་པའི་
```
(Note the space before the expansion)

### Example 2: Non-Tokenized Text

**Original input line:**
```
བས་ལེགས་གསུངས་ཐེག་གསུམ་བསྟན་པའི་
```

**Dictionary entry:**
```
[སྟོངག་] [སྟོང་ཕྲག]
```

**Output in _abbrev.txt:**
```
བས་ལེགས་གསུངས་ཐེག་གསུམ་བསྟན་པའི་སྟོངག་
```
(Note: no space, glued directly)

**Output in _dictaug.txt:**
```
བས་ལེགས་གསུངས་ཐེག་གསུམ་བསྟན་པའི་སྟོང་ཕྲག
```
(Note: no space, glued directly)

## How It Works

1. **Dictionary Processing**: Reads the dictionary file and removes all `[]` brackets from both columns
2. **Text Loading**: Loads all lines from the input text file
3. **Random Selection**: Randomly selects 10,000 line indices from the total available lines
4. **Abbreviation-Expansion Assignment**: Randomly assigns abbreviation-expansion pairs to the selected lines
5. **Output Generation**: Creates two output files:
   - **_abbrev.txt**: Lines with abbreviations added (on selected lines only)
   - **_dictaug.txt**: Lines with expansions added (on the same selected lines)
   - Both files maintain the original line order
   - Unmodified lines remain exactly as they were

## Output Statistics

The script provides detailed statistics during execution:

```
Reading dictionary file...
Loaded 10000 abbreviation pairs
Reading input text file...
Loaded 25000 lines from input file
Processing 10000 lines with abbreviations...
Mode: Tokenized (with space)
Output files written:
  With abbreviations: input_abbrev.txt
  With expansions: input_dictaug.txt
Total lines: 25000
Lines modified: 10000
Lines unchanged: 15000

Processing complete!
```

## Error Handling

The script includes comprehensive error handling for:
- Missing input files
- Malformed dictionary entries (warning, not error)
- Insufficient lines in input file
- Invalid command line arguments

## Notes

- The script processes exactly 10,000 lines (or fewer if the dictionary has fewer entries)
- Lines are selected randomly, not the first 10,000 lines
- The expansion is added as-is from the dictionary without further processing
- Original line order is always preserved in the output
- Empty lines in the input are treated like any other line

## License

This script is provided as-is for processing Tibetan text files.

## Support

If you encounter issues:
1. Check that your dictionary file has two columns separated by tabs or multiple spaces
2. Ensure your input files are UTF-8 encoded
3. Verify that brackets `[]` appear in the dictionary file (they will be removed automatically)
4. Check the console output for specific error messages or warnings