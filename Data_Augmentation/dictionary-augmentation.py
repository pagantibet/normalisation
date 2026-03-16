#!/usr/bin/env python3
"""
Tibetan Abbreviation Processor

This script processes a Tibetan text file by:
1. Reading abbreviations from a dictionary file
2. Randomly selecting lines from the input text
3. Adding abbreviations and their expansions to the output
"""

import random
import re
import sys
from pathlib import Path


def read_dictionary(dict_file):
    """
    Read the dictionary file and extract abbreviation-expansion pairs.
    Removes [] brackets from the content.
    
    Args:
        dict_file: Path to the dictionary file
        
    Returns:
        List of tuples (abbreviation, expansion)
    """
    abbreviations = []
    
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Remove brackets
            line = line.replace('[', '').replace(']', '')
            line = line.strip()
            
            if not line:
                continue
            
            # Split by tab or multiple spaces (common delimiters)
            parts = re.split(r'\t+|\s{2,}', line, maxsplit=1)
            
            if len(parts) >= 2:
                abbrev = parts[0].strip()
                expansion = parts[1].strip()
                abbreviations.append((abbrev, expansion))
            else:
                print(f"Warning: Line {line_num} in dictionary doesn't have 2 columns: {line[:50]}...", 
                      file=sys.stderr)
    
    return abbreviations


def process_tibetan_text(input_file, dict_file, output_file_abbrev, output_file_expanded, tokenized=True, seed=None):
    """
    Process the Tibetan text file with abbreviations.
    
    Args:
        input_file: Path to input text file (~25k lines)
        dict_file: Path to dictionary file (~10k entries)
        output_file_abbrev: Path to output file with abbreviations added
        output_file_expanded: Path to output file with expansions added
        tokenized: If True, add space before abbrev/expansion; if False, glue directly (default: True)
        seed: Random seed for reproducibility (optional)
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Read the dictionary
    print("Reading dictionary file...")
    abbreviations = read_dictionary(dict_file)
    print(f"Loaded {len(abbreviations)} abbreviation pairs")
    
    if not abbreviations:
        print("Error: No abbreviations found in dictionary file!", file=sys.stderr)
        return
    
    # Read all lines from input file
    print("Reading input text file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n\r') for line in f]
    
    total_lines = len(lines)
    print(f"Loaded {total_lines} lines from input file")
    
    # Determine how many abbreviations to use (minimum of available abbrevs or 10k)
    num_to_process = min(len(abbreviations), 10000)
    
    # Randomly select line indices (excluding empty lines if any)
    available_indices = list(range(total_lines))
    
    if len(available_indices) < num_to_process:
        print(f"Warning: Only {len(available_indices)} lines available, but need {num_to_process}", 
              file=sys.stderr)
        num_to_process = len(available_indices)
    
    # Randomly select indices
    selected_indices = random.sample(available_indices, num_to_process)
    selected_indices_set = set(selected_indices)
    
    # Randomly shuffle the abbreviations list to use
    abbrev_to_use = random.sample(abbreviations, num_to_process)
    
    # Create a mapping of line index to (abbreviation, expansion)
    index_to_abbrev = {}
    for idx, (abbrev, expansion) in zip(sorted(selected_indices), abbrev_to_use):
        index_to_abbrev[idx] = (abbrev, expansion)
    
    print(f"Processing {num_to_process} lines with abbreviations...")
    print(f"Mode: {'Tokenized (with space)' if tokenized else 'Non-tokenized (no space)'}")
    
    # Determine separator based on tokenization
    separator = ' ' if tokenized else ''
    
    # Process and write both output files
    # File 1: Input with abbreviations added
    with open(output_file_abbrev, 'w', encoding='utf-8') as f_abbrev:
        for i, line in enumerate(lines):
            if i in index_to_abbrev:
                abbrev, expansion = index_to_abbrev[i]
                # Add the abbreviation to the line
                output_line = f"{line}{separator}{abbrev}"
            else:
                output_line = line
            
            f_abbrev.write(output_line + '\n')
    
    # File 2: Input with expansions added (at the same positions)
    with open(output_file_expanded, 'w', encoding='utf-8') as f_expanded:
        for i, line in enumerate(lines):
            if i in index_to_abbrev:
                abbrev, expansion = index_to_abbrev[i]
                # Add the expansion to the line with appropriate separator
                output_line = f"{line}{separator}{expansion}"
            else:
                output_line = line
            
            f_expanded.write(output_line + '\n')
    
    print(f"Output files written:")
    print(f"  With abbreviations: {output_file_abbrev}")
    print(f"  With expansions: {output_file_expanded}")
    print(f"Total lines: {total_lines}")
    print(f"Lines modified: {num_to_process}")
    print(f"Lines unchanged: {total_lines - num_to_process}")


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 3:
        print("Usage: python dictionary-augmentation.py <input_text_file> <dictionary_file> [--non-tokenized] [random_seed]")
        print("\nArguments:")
        print("  input_text_file  : Text file with ~25k Tibetan lines")
        print("  dictionary_file  : Dictionary with abbreviations in column 1, expansions in column 2")
        print("  --non-tokenized  : (Optional) Use this flag for non-tokenized input (no space before expansion)")
        print("  random_seed      : (Optional) Integer seed for reproducibility")
        print("\nOutput:")
        print("  Two output files are automatically created:")
        print("    <input_file>_abbrev.txt  - with abbreviations added")
        print("    <input_file>_dictaug.txt - with expansions added")
        print("\nExamples:")
        print("  Tokenized:     python dictionary-augmentation.py input.txt dict.txt 42")
        print("                 (creates input_abbrev.txt and input_dictaug.txt)")
        print("  Non-tokenized: python dictionary-augmentation.py input.txt dict.txt --non-tokenized 42")
        sys.exit(1)
    
    input_file = sys.argv[1]
    dict_file = sys.argv[2]
    
    # Generate output filenames from input filename
    input_path = Path(input_file)
    output_file_abbrev = str(input_path.parent / f"{input_path.stem}_abbrev{input_path.suffix}")
    output_file_expanded = str(input_path.parent / f"{input_path.stem}_dictaug{input_path.suffix}")
    
    # Parse optional arguments
    tokenized = True
    seed = None
    
    for arg in sys.argv[3:]:
        if arg == '--non-tokenized':
            tokenized = False
        else:
            try:
                seed = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}", file=sys.stderr)
    
    # Validate input files exist
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(dict_file).exists():
        print(f"Error: Dictionary file not found: {dict_file}", file=sys.stderr)
        sys.exit(1)
    
    # Process the files
    process_tibetan_text(input_file, dict_file, output_file_abbrev, output_file_expanded, tokenized, seed)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()