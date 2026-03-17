# Botok Tokeniser Script

A Python script for tokenising Tibetan text (.txt) files using the [botok tokeniser](https://github.com/OpenPecha/botok), optimised for large files with progress reporting.

This script was developed as part of [PaganTibet](https://www.pagantibet.com/)'s Normalisation workflow. For more information, see our [Normalisation README](https://github.com/pagantibet/normalisation/tree/main?tab=readme-ov-file).

## Overview

This script tokenises parallel text files (source and target) while preserving line structure. It's designed to handle very large files efficiently (tested with multi-GB files) and includes a fix for [botok's incorrect handling of certain Tibetan marks](#botok-punctuation-fix).

## Features

- **Line-by-line processing** - Preserves the original line structure
- **Large file optimisation** - Uses 8MB buffers for efficient I/O
- **Progress reporting** - Shows real-time progress, speed, and percentage complete
- **Botok punctuation fix** - Corrects botok's treatment of ༷ (U+0F37) and ༹ (U+0F39) as punctuation marks
- **Dual file processing** - Handles both source and target files in one run

## Requirements

- Python 3.6 or higher
- botok tokeniser

## Installation

Install the botok tokeniser:

```bash
pip install botok --break-system-packages
```

Or if you're using a virtual environment:

```bash
pip install botok
```

## Usage

### Process both source and target files (default)

1. Place the script in the same directory as your input files
2. Ensure your input files are named:
   - `train_source.txt`
   - `train_target.txt`
3. Run the script:

```bash
python tokenize_files.py
```

### Process a single file

To tokenise just one file:

```bash
python tokenize_files.py my_file.txt
```

This will create `my_file-tok.txt` in the same directory.

You can also specify a custom output filename:

```bash
python tokenize_files.py input.txt output-tokenized.txt
```

### View help

```bash
python tokenize_files.py --help
```

## Input/Output

**Input files:**
- `train_source.txt` - Source text file
- `train_target.txt` - Target text file
- Or any custom file name when using single-file mode

**Output files:**
- `train_source-tok.txt` - Tokenised source text
- `train_target-tok.txt` - Tokenised target text
- `train_source-tok-errors.txt` - Error log (only created if errors occur)
- `train_target-tok-errors.txt` - Error log (only created if errors occur)

**Error log format:**
The error log file contains detailed information about each line that failed to tokenise:
```
Line 5412345:
Error: can't set attribute 'syls'
Content: ཱུྃ་ལས་དཀར་པོའི་...
--------------------------------------------------------------------------------

Line 5412891:
Error: can't set attribute 'syls'
Content: རྡོ་རྗེ་སེམས་དཔའ་...
--------------------------------------------------------------------------------
```

## Example Output

### Processing both files (default mode)

```
Processing train_source.txt...
  File size: 3.50 GB
  Progress: 10,000 lines (2.3%) - 2,500 lines/sec - 80.00 MB/3.50 GB
  Progress: 20,000 lines (4.6%) - 2,450 lines/sec - 160.00 MB/3.50 GB
  ...
  Completed: 1,250,000 lines in 500.2 seconds
  Average speed: 2,499 lines/sec
  Output written to train_source-tok.txt

Processing train_target.txt...
  File size: 3.45 GB
  ...

Total processing time: 1002.5 seconds
Tokenization complete!
```

### Processing a single file

```
Processing my_tibetan_text.txt...
  File size: 1.20 GB
  Progress: 10,000 lines (3.8%) - 2,800 lines/sec - 45.00 MB/1.20 GB
  Warning: Error on line 145823: can't set attribute 'syls'
  Warning: Error on line 234567: can't set attribute 'syls'
  (Further errors will be logged to my_tibetan_text-tok-errors.txt)
  Progress: 20,000 lines (7.5%) - 2,750 lines/sec - 90.00 MB/1.20 GB - 8 errors
  ...
  Completed: 425,000 lines in 152.3 seconds
  Average speed: 2,790 lines/sec
  Errors encountered: 12 lines (written unchanged)
  Error log saved to: my_tibetan_text-tok-errors.txt
  Output written to my_tibetan_text-tok.txt

Total processing time: 152.3 seconds
Tokenization complete!
```

## Performance

The script is optimised for large files:
- **Memory efficient** - Only one line is kept in memory at a time
- **Fast I/O** - Uses large (8MB) read/write buffers
- **Expected speed** - 1,000-5,000 lines per second (depending on system and line length)

For a 3.5GB file with typical line lengths, expect processing to take 10-30 minutes depending on your hardware.

## Botok Punctuation Fix

The script includes a post-processing fix for two Tibetan characters that botok incorrectly treats as punctuation marks:

- **༷** (U+0F37) - mark nyi zla
- **༹** (U+0F39) - tsa rtags

These are actually marks that should remain attached to surrounding text, not standalone punctuation. The script removes any word breaks (spaces) around these characters after tokenisation.

## Technical Details

### Line Structure Preservation

Each line in the input file corresponds to exactly one line in the output file. Empty lines are preserved.

### Token Format

Tokens are separated by single spaces. For example:

**Input:** `བཀྲ་ཤིས་བདེ་ལེགས།`

**Output:** `བཀྲ་ཤིས་ བདེ་ལེགས །`

### Character Encoding

All files are read and written using UTF-8 encoding.

## Troubleshooting

### "botok is not installed"
Install botok using the command in the [Installation](#installation) section above.

### "File not found"
Ensure your input files are named exactly `train_source.txt` and `train_target.txt` and are in the same directory as the script.

### Botok warnings about "non-expanded char"
The script automatically suppresses these warnings. They occur when botok encounters certain Unicode combinations but don't affect the output.

### Botok errors ("can't set attribute 'syls'" or similar)
The script now handles these gracefully:
- Lines that cause tokenisation errors are written to the output unchanged
- All errors are logged to a separate error file (e.g., `filename-tok-errors.txt`)
- The error log contains:
  - Line number where the error occurred
  - Full error message
  - The problematic content
- The first 5 errors are also displayed in the console
- Processing continues to completion
- If no errors occur, no error log file is created

This allows you to:
- Process large files completely, even if some lines cause issues
- Easily identify and review all problematic lines
- Fix or investigate specific errors after processing

### Slow processing
Check your disk I/O speed. For very large files, SSD storage will be significantly faster than HDD.

### Memory issues
The script is designed to use minimal memory. If you encounter memory issues, it's likely due to botok's internal data structures. Try processing smaller batches or increasing available RAM.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.

## Credits

- Uses the [botok](https://github.com/OpenPecha/Botok) tokeniser by OpenPecha
- Includes fix for Tibetan mark handling
