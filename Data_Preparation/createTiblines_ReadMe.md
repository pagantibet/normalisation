# Line Formatter for Tibetan Text

A Python script that reformats Tibetan text files to have lines between 60 and 100 characters long. Supports both **segmented** (space-separated syllables) and **unsegmented** (continuous) Tibetan text.

**Optimized for large files** - processes ~5-6 MB/second on typical hardware.

This script was developed as part of [PaganTibet](https://www.pagantibet.com/)'s Normalisation workflow. For more information, see our [Normalisation README](https://github.com/pagantibet/normalisation/tree/main?tab=readme-ov-file).

## Features

- **Two modes**: Segmented and unsegmented text
- **Smart text cleaning**: Removes non-Tibetan characters automatically
- **Punctuation normalisation**: Removes spaces between Tibetan punctuation, adds space after །། (double shad)
- **Natural break points**: For unsegmented text, breaks at ། (shad), ་ (tsheg), or spaces
- **Never splits indivisible units**: Respects །། as atomic, preserves syllable integrity
- **Progress reporting**: Shows progress for large files
- **Flexible output**: Quiet mode for automation, verbose mode for debugging

## Requirements

- Python 3.x
- No external dependencies required
- Sufficient RAM for your file size (allow ~2x file size for processing)

## Character Encoding

The script uses UTF-8 encoding and works with the full Tibetan Unicode range (U+0F00-U+0FFF).

## Performance

Based on benchmarks:
- **Processing rate**: ~5-6 MB/second on typical CPU
- **10 MB file**: ~2 seconds
- **1 GB file**: ~3 minutes
- **3.5 GB file**: ~10-11 minutes

## Usage

```bash
python3 createTiblines.py <input_file> <output_file> [options]
```

### Basic Examples

```bash
# Segmented text (space-separated syllables)
python3 createTiblines.py input.txt output.txt

# Unsegmented text (continuous Tibetan)
python3 createTiblines.py input.txt output.txt --unsegmented

# Custom length range
python3 createTiblines.py input.txt output.txt --min 70 --max 120

# Quiet mode (minimal output, good for large files)
python3 createTiblines.py input.txt output.txt --quiet --unsegmented
```

### Command Line Options

- `--unsegmented` : Handle unsegmented/continuous Tibetan text (breaks at shad/tsheg)
- `--min N` : Minimum line length (default: 60)
- `--max N` : Maximum line length (default: 100)
- `--quiet` : Minimal output, only final summary (recommended for large files)
- `--verbose` : Print detailed statistics for each output line (not recommended for large files)

## Input Formats

### Segmented Text (Default Mode)

Text where Tibetan syllables are already separated by spaces:

```
p1 ༄༅ ། ། <utt>
རྗོད་བྱེད་ ཚིག་ གི་ རྒྱུད་ བཤད་པ་ མཁས་པ འི་ ཁ་རྒྱན་ ཞེས་ བྱ་བ་ བཞུགས་ སོ ། ། <utt>
```

### Unsegmented Text (`--unsegmented` mode)

Continuous Tibetan text without syllable breaks:

```
p1༄༅།། རྗོད་བྱེད་ཚིག་གི་རྒྱུད་བཤད་པ་མཁས་པའི་ཁ་རྒྱན་ཞེས་བྱ་བ་བཞུགས་སོ།། p2༄༅།།
བླ་མ་དང་མགོན་པོ་འཇམ་པའི་དབྱངས་ལ་ཕྱག་འཚལ་ལོ།།
```

## Output Format

Clean Tibetan text with optimized line lengths:

```
༄༅།། རྗོད་བྱེད་ཚིག་གི་རྒྱུད་བཤད་པ་མཁས་པའི་ཁ་རྒྱན་ཞེས་བྱ་བ་བཞུགས་སོ།། ༄༅།།
བླ་མ་དང་མགོན་པོ་འཇམ་པའི་དབྱངས་ལ་ཕྱག་འཚལ་ལོ།། གང་ཡང་ཚིག་ལ་དབང་བའི་འདུན་ས་ཆེར།།
```

**Output characteristics:**
- All non-Tibetan characters removed (`p1`, `p2`, `<utt>`, etc.)
- Punctuation sequences like `༄༅།།` have no spaces between marks
- Space added after `།།` when followed by more content
- Lines between 60-100 characters (or custom range)

## How It Works

### Text Cleaning (Both Modes)

1. **Removes non-Tibetan characters**: Latin letters, numbers, special markers
2. **Normalises punctuation spacing**:
   - Removes spaces between punctuation marks (except tsheg ་)
   - Adds space after །། when followed by Tibetan content
   - Preserves tsheg ་ as part of syllables

Example transformations:
- `p1༄༅ ། །` → `༄༅།།`
- `སོ།།༄༅` → `སོ།། ༄༅`
- `སོ ། །` → `སོ།།`

### Segmented Mode (Default)

Treats spaces as syllable boundaries:
1. Splits text on spaces to get units
2. Combines units into lines up to max_length
3. Merges short consecutive lines to reach min_length

**Best for**: Pre-segmented corpora, word-by-word annotated texts

### Unsegmented Mode (`--unsegmented`)

Breaks at natural Tibetan punctuation:
1. Identifies break points: `།` (shad), `་` (tsheg), spaces
2. **Never splits `།།`** - treated as atomic unit
3. Breaks at `།།`, then `།`, then `་`, then spaces
4. Combines chunks into lines respecting max_length
5. Merges short lines to reach min_length

**Best for**: Continuous Tibetan text, traditional manuscript format

## Break Point Priority (Unsegmented Mode)

When a line exceeds max_length, the script looks backwards for:
1. **`།།`** (double shad) - highest priority
2. **`།`** (single shad) - high priority  
3. **`་`** (tsheg) - medium priority
4. **Space** - lowest priority

The script **never breaks between the two shads in `།།`**.

## Important Notes

### Constraints

**Segmented mode:**
- Can only break at existing spaces
- Very long space-separated units (>100 chars) remain intact

**Unsegmented mode:**
- Can only break at `།`, `་`, or spaces
- Sequences without these marks (>100 chars) remain intact

### Edge Cases

Short lines that cannot reach min_length without exceeding max_length will remain short. For example:

```
ཞེས་གསུངས་པ་ལྟར།   (16 characters)
```

This phrase has no good break points and cannot be merged with neighbors without exceeding max_length, so it remains as-is.

### Best Practices

1. **Use `--quiet` for large files**: Reduces output and improves performance
2. **Choose the right mode**: Use `--unsegmented` for continuous text, default for pre-segmented text
3. **Mixed content is fine**: Non-Tibetan markers are automatically removed
4. **Adjust length constraints**: Use `--min` and `--max` for different formatting needs

## Examples

### Segmented Text Example

**Input:**
```
p1 ༄༅ ། ། <utt>
རྗོད་བྱེད་ ཚིག་ གི་ རྒྱུད་ བཤད་པ་ མཁས་པ འི་ <utt>
```

**Command:**
```bash
python line_formatter.py input.txt output.txt
```

**Output:**
```
༄༅།། རྗོད་བྱེད་ ཚིག་ གི་ རྒྱུད་ བཤད་པ་ མཁས་པ འི་
```

### Unsegmented Text Example

**Input:**
```
p1༄༅།། རྗོད་བྱེད་ཚིག་གི་རྒྱུད་བཤད་པ་མཁས་པའི་ཁ་རྒྱན་ཞེས་བྱ་བ་བཞུགས་སོ།། p2༄༅།། བླ་མ་དང་མགོན་པོ་འཇམ་པའི་དབྱངས་ལ་ཕྱག་འཚལ་ལོ།། 
```

**Command:**
```bash
python line_formatter.py input.txt output.txt --unsegmented
```

**Output:**
```
༄༅།། རྗོད་བྱེད་ཚིག་གི་རྒྱུད་བཤད་པ་མཁས་པའི་ཁ་རྒྱན་ཞེས་བྱ་བ་བཞུགས་སོ།། ༄༅།།
བླ་མ་དང་མགོན་པོ་འཇམ་པའི་དབྱངས་ལ་ཕྱག་འཚལ་ལོ།།
```

## Tibetan Punctuation Handled

The script recognizes and normalizes these Tibetan punctuation marks:

- `།` (U+0F0D shad) - primary break point
- `༄` `༅` `༆` `༇` `༈` `༉` `༊` - various markers
- `༎` `༏` `༐` `༑` `༒` `༓` `༔` - various shads and marks
- `༕` `༖` `༗` `༘` `༙` `༚` `༛` `༜` `༝` `༞` `༟` - additional marks
- `༴` `༶` `༸` - special marks
- `༺` `༻` `༼` `༽` `༾` `༿` - brackets and marks

**Note:** Tsheg (་ U+0F0B) is NOT treated as punctuation - it's kept as part of syllables in segmented mode and used as a break point in unsegmented mode.

## Output Statistics

The script provides detailed processing information:

```
Settings: min_length=60, max_length=100, mode=unsegmented

Reading input file: input.txt (mode: unsegmented)
Processing 10 lines...
Formatting lines...
Merging short lines...
Writing output file: output.txt

============================================================
Processing complete!
============================================================
Input lines:        10
Output lines:       11
Lines in range:     10
Lines out of range: 1
Processing time:    0.00 seconds
Output saved to:    output.txt
============================================================
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
