#!/usr/bin/env python3
"""
Script to reformat Tibetan text lines to be between 60 and 100 characters long.

Supports two modes:
1. Segmented text: syllables already separated by spaces
2. Unsegmented text: continuous Tibetan text, breaks at shad/tsheg

This version:
- Removes non-Tibetan characters (like <utt>, p1, p2, etc.)
- Preserves spaces between Tibetan syllables (segmented mode)
- Removes spaces between Tibetan punctuation marks (except tsheg ་)
- For unsegmented text, breaks long lines at ། or ་ or spaces
- Adds space after །། when appropriate
- Keeps tsheg as part of syllables in segmented mode
- Optimized for large files with minimal output

IMPORTANT: If a single unit (text without break points) is longer than max_length
it will remain as-is, since we cannot break without natural break points.

# Quiet mode recommended for larger (e.g. 3.5GB) files:
python3 createTiblines.py input.txt output.txt --quiet

# Process unsegmented text
python3 createTiblines.py input.txt output.txt --unsegmented

# With custom length (e.g., 150 chars max as you mentioned)
python3 createTiblines.py input.txt output.txt --unsegmented --max 150

# For large files
python3 createTiblines.py input.txt output.txt --unsegmented --quiet

"""

import re
import sys
import time

# Tibetan punctuation marks (excluding tsheg ་)
TIBETAN_PUNCTUATION = set([
    '།',   # U+0F0D shad
    '༄',   # U+0F04 initial yig mgo mdun ma
    '༅',   # U+0F05 closing yig mgo sgab ma
    '༆',   # U+0F06 caret yig mgo phur shad ma
    '༇',   # U+0F07 yig mgo tsheg shad ma
    '༈',   # U+0F08 sbrul shad
    '༉',   # U+0F09 bskur yig mgo
    '༊',   # U+0F0A bka' shog yig mgo
    # '༌',   # U+0F0C delimiter tsheg bstar
    '༎',   # U+0F0E nyis shad
    '༏',   # U+0F0F tsheg shad
    '༐',   # U+0F10 nyis tsheg shad
    '༑',   # U+0F11 rin chen spungs shad
    '༒',   # U+0F12 rgya gram shad
    '༓',   # U+0F13 caret dzud rtags me long can
    '༔',   # U+0F14 gter tsheg
    '༕',   # U+0F15 chad rtags
    '༖',   # U+0F16 lhag rtags
    '༗',   # U+0F17 snga phyir
    '༘',   # U+0F18 che mgo (below)
    '༙',   # U+0F19 nyi zla (below)
    '༚',   # U+0F1A sbrul yig (below)
    '༛',   # U+0F1B bska shog gi mgo rgyan (below)
    '༜',   # U+0F1C ang khang gyon (below)
    '༝',   # U+0F1D ang khang gyas (below)
    '༞',   # U+0F1E ang khang (below)
    '༟',   # U+0F1F ang khang (below)
    '༴',   # U+0F34 mark bsdus rtags
    '༶',   # U+0F36 mark caret -dzud rtags bzhi mig can
    # '༷',   # U+0F37 mark nyi zla
    '༸',   # U+0F38 che mgo
    '༺',   # U+0F3A gug rtags gyon
    '༻',   # U+0F3B gug rtags gyas
    '༼',   # U+0F3C ang khang gyon
    '༽',   # U+0F3D ang khang gyas
    '༾',   # U+0F3E yar tshes
    '༿',   # U+0F3F mar tshes
])

def is_tibetan_char(char):
    """Check if a character is Tibetan (letter or punctuation)."""
    code = ord(char)
    return (0x0F00 <= code <= 0x0FFF)

def is_tibetan_punctuation(char):
    """Check if a character is Tibetan punctuation (excluding tsheg)."""
    return char in TIBETAN_PUNCTUATION

def clean_text(text):
    """
    Remove non-Tibetan characters and clean up spaces between punctuation.
    Preserves spaces between Tibetan syllables.
    Adds space after །། when followed directly by Tibetan content.
    """
    # First, remove any non-Tibetan characters (Latin, numbers, <utt>, etc.)
    # Keep only Tibetan characters and spaces
    cleaned_chars = []
    for char in text:
        if is_tibetan_char(char) or char == ' ':
            cleaned_chars.append(char)
    
    text = ''.join(cleaned_chars)
    
    # Add space after །། if followed immediately by any Tibetan character (not space, not another ।)
    text_with_shad_spaces = []
    i = 0
    while i < len(text):
        text_with_shad_spaces.append(text[i])
        # Check if we're at a །། followed by a non-space, non-shad Tibetan character
        if (i + 1 < len(text) and 
            text[i] == '།' and text[i+1] == '།' and
            i + 2 < len(text) and text[i+2] != ' ' and text[i+2] != '།'):
            # Add the second shad and a space
            text_with_shad_spaces.append(text[i+1])  # Add second །
            text_with_shad_spaces.append(' ')  # Add space
            i += 2
            continue
        i += 1
    
    text = ''.join(text_with_shad_spaces)
    
    # Now remove spaces between punctuation marks (but keep tsheg, and keep the །། spaces we just added)
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        
        if char == ' ':
            # Check if this space is between two punctuation marks
            prev_char = None
            for j in range(len(result) - 1, -1, -1):
                if result[j] != ' ':
                    prev_char = result[j]
                    break
            
            next_char = None
            for j in range(i + 1, len(text)):
                if text[j] != ' ':
                    next_char = text[j]
                    break
            
            # Special case: keep space after །།
            if len(result) >= 2 and result[-1] == '།' and result[-2] == '།':
                result.append(char)
            # Only keep the space if at least one adjacent char is NOT punctuation
            elif prev_char is None or next_char is None:
                result.append(char)
            elif is_tibetan_punctuation(prev_char) and is_tibetan_punctuation(next_char):
                pass  # Skip space between punctuation
            else:
                result.append(char)
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)

def split_unsegmented_text(text, max_length):
    """
    Split unsegmented Tibetan text at natural break points.
    Priority: །། (double shad with space after), ། (shad), ་ (tsheg), space
    Never splits །།
    Always adds space after །། when it's followed by more content
    """
    chunks = []
    current = ""
    i = 0
    
    while i < len(text):
        char = text[i]
        current += char
        
        # If we're at or near max_length, look for a break point
        if len(current) >= max_length:
            # Find the best break point in current chunk
            best_break = -1
            break_type = None
            
            # Look backwards for a good break point
            j = len(current) - 1
            while j > 0 and best_break == -1:
                # Check for །། (don't split between them)
                if current[j] == '།' and j > 0 and current[j-1] == '།':
                    # Found །།, this is a good break point after both shads
                    best_break = j
                    break_type = 'double_shad'
                    break
                # Check for single །
                elif current[j] == '།':
                    # Make sure it's not part of །།
                    if not (j + 1 < len(current) and current[j+1] == '།'):
                        best_break = j
                        break_type = 'shad'
                        break
                j -= 1
            
            # If no shad found, look for ་
            if best_break == -1:
                for j in range(len(current) - 1, 0, -1):
                    if current[j] == '་':
                        best_break = j
                        break_type = 'tsheg'
                        break
            
            # If no tsheg found, look for space
            if best_break == -1:
                for j in range(len(current) - 1, 0, -1):
                    if current[j] == ' ':
                        best_break = j
                        break_type = 'space'
                        break
            
            # If we found a break point
            if best_break > 0:
                chunk = current[:best_break + 1].strip()
                # Add space after །། if it's a double shad break  
                if break_type == 'double_shad' and not chunk.endswith(' '):
                    chunk += ' '
                chunks.append(chunk)
                current = current[best_break + 1:]
            else:
                # No break point found, have to keep going
                pass
        
        i += 1
    
    # Add remaining text
    if current.strip():
        # Check if it ends with །། and add space if needed (for consistency)
        current = current.strip()
        chunks.append(current)
    
    return chunks

def format_lines_segmented(all_units, min_length, max_length):
    """Format pre-segmented text (space-separated syllables)."""
    formatted_lines = []
    current_line = []
    
    for unit in all_units:
        if current_line:
            proposed_length = len(' '.join(current_line)) + 1 + len(unit)
        else:
            proposed_length = len(unit)
        
        if proposed_length <= max_length:
            current_line.append(unit)
        else:
            if current_line:
                formatted_lines.append(' '.join(current_line))
            current_line = [unit]
    
    if current_line:
        formatted_lines.append(' '.join(current_line))
    
    return formatted_lines

def format_lines_unsegmented(text, min_length, max_length):
    """Format unsegmented text by breaking at shad/tsheg."""
    # Split into chunks
    chunks = split_unsegmented_text(text, max_length)
    
    formatted_lines = []
    current_line = ""
    
    for chunk in chunks:
        # Try to add chunk to current line
        if not current_line:
            current_line = chunk
        else:
            # Determine spacing - add space after །། if not already there
            if current_line.endswith('།། '):
                proposed = current_line + chunk
            elif current_line.endswith('།།'):
                proposed = current_line + ' ' + chunk
            else:
                proposed = current_line + ' ' + chunk
            
            if len(proposed) <= max_length:
                current_line = proposed
            else:
                # Save current line and start new one
                # Make sure we add space after །། if needed
                line_to_save = current_line
                if line_to_save.endswith('།།') and not line_to_save.endswith('།། '):
                    line_to_save = line_to_save + ' '
                formatted_lines.append(line_to_save.rstrip())
                current_line = chunk
    
    if current_line:
        formatted_lines.append(current_line.strip())
    
    return formatted_lines

def merge_short_lines(formatted_lines, min_length, max_length):
    """Merge consecutive short lines if possible."""
    merged_lines = []
    i = 0
    
    while i < len(formatted_lines):
        current = formatted_lines[i]
        
        if len(current) >= min_length:
            merged_lines.append(current)
            i += 1
        else:
            # Try to merge with next lines
            merged = current
            j = i + 1
            
            while j < len(formatted_lines):
                next_line = formatted_lines[j]
                # Determine spacing
                if merged.endswith('།། ') or merged.endswith('།།'):
                    if merged.endswith('།།'):
                        proposed = merged + ' ' + next_line
                    else:
                        proposed = merged + next_line
                else:
                    proposed = merged + ' ' + next_line
                
                if len(proposed) <= max_length:
                    merged = proposed
                    j += 1
                    if len(merged) >= min_length:
                        break
                else:
                    break
            
            merged_lines.append(merged)
            i = j
    
    return merged_lines

def format_lines(input_file, output_file, min_length=60, max_length=100, 
                 unsegmented=False, verbose=False, quiet=False):
    """
    Reformat text lines to be between min_length and max_length characters.
    """
    start_time = time.time()
    
    if not quiet:
        mode = "unsegmented" if unsegmented else "segmented"
        print(f"Reading input file: {input_file} (mode: {mode})")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    input_line_count = len(lines)
    
    if not quiet:
        print(f"Processing {input_line_count:,} lines...")
    
    # Clean all text
    all_text = []
    for i, line in enumerate(lines):
        if not quiet and i > 0 and i % 100000 == 0:
            print(f"  Cleaned {i:,} lines...")
        cleaned = clean_text(line.strip())
        if cleaned:
            all_text.append(cleaned)
    
    # Join all text
    full_text = ' '.join(all_text)
    
    if not quiet:
        print("Formatting lines...")
    
    # Format based on mode
    if unsegmented:
        formatted_lines = format_lines_unsegmented(full_text, min_length, max_length)
    else:
        all_units = full_text.split()
        if not quiet:
            print(f"Total units after cleaning: {len(all_units):,}")
        formatted_lines = format_lines_segmented(all_units, min_length, max_length)
    
    # Merge short lines
    if not quiet:
        print("Merging short lines...")
    merged_lines = merge_short_lines(formatted_lines, min_length, max_length)
    
    # Write output
    if not quiet:
        print(f"Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in merged_lines:
            f.write(line + '\n')
    
    # Statistics
    out_of_range = sum(1 for line in merged_lines if not (min_length <= len(line) <= max_length))
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Input lines:        {input_line_count:,}")
    print(f"Output lines:       {len(merged_lines):,}")
    print(f"Lines in range:     {len(merged_lines) - out_of_range:,}")
    print(f"Lines out of range: {out_of_range:,}")
    print(f"Processing time:    {elapsed_time:.2f} seconds")
    print(f"Output saved to:    {output_file}")
    print(f"{'='*60}")
    
    if verbose:
        print(f"\nDetailed line statistics:")
        for i, line in enumerate(merged_lines, 1):
            length = len(line)
            in_range = min_length <= length <= max_length
            status = "✓" if in_range else "✗"
            print(f"  Line {i}: {length} chars {status}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reformat Tibetan text to have lines between specified character lengths.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segmented text (space-separated syllables)
  python line_formatter.py input.txt output.txt
  
  # Unsegmented text (continuous Tibetan)
  python line_formatter.py input.txt output.txt --unsegmented
  
  # Custom length range
  python line_formatter.py input.txt output.txt --min 70 --max 120
  
  # Quiet mode (minimal output)
  python line_formatter.py input.txt output.txt --quiet
        """
    )
    
    parser.add_argument('input_file', help='Input text file')
    parser.add_argument('output_file', help='Output text file')
    parser.add_argument('--min', type=int, default=60, help='Minimum line length (default: 60)')
    parser.add_argument('--max', type=int, default=100, help='Maximum line length (default: 100)')
    parser.add_argument('--unsegmented', action='store_true', 
                       help='Handle unsegmented text (breaks at shad/tsheg instead of spaces)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed per-line statistics')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (only summary)')
    
    args = parser.parse_args()
    
    if not args.quiet:
        mode = "unsegmented" if args.unsegmented else "segmented"
        print(f"Settings: min_length={args.min}, max_length={args.max}, mode={mode}\n")
    
    format_lines(args.input_file, args.output_file, args.min, args.max, 
                args.unsegmented, args.verbose, args.quiet)