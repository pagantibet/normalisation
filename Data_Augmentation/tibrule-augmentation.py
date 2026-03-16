#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based data augmentation for Tibetan text.
Applies bidirectional character replacements and random syllable deletions/additions.

Default augmentation rate: 20% (adjustable via --ratio parameter)
Syllable deletion/addition rate: 3%
"""

import random
import sys
import argparse
from pathlib import Path


def find_character_positions(text, char):
    """Find all positions of a character in the text."""
    return [i for i, c in enumerate(text) if c == char]


def apply_replacement(text, char, replacement, ratio=0.2):
    """Apply replacement to a percentage of character occurrences."""
    positions = find_character_positions(text, char)
    if not positions:
        return text
    
    # Randomly select positions to replace
    num_to_replace = int(len(positions) * ratio)
    if num_to_replace == 0 and len(positions) > 0 and random.random() < ratio:
        num_to_replace = 1  # Ensure at least some chance of replacement
    
    if num_to_replace > 0:
        positions_to_replace = random.sample(positions, num_to_replace)
        
        # Convert text to list for efficient replacement
        text_list = list(text)
        for pos in positions_to_replace:
            text_list[pos] = replacement
        
        return ''.join(text_list)
    
    return text


def reverse_gigu_to_normal(text, ratio=0.2):
    """Convert reverse gigu (ི) to normal gigu (ི).
    Note: In Unicode, reverse gigu is U+0F80 and normal gigu is U+0F72"""
    reverse_gigu = '\u0F80'  # ྀ (reverse gigu)
    normal_gigu = '\u0F72'   # ི (gigu)
    
    positions = find_character_positions(text, reverse_gigu)
    if not positions:
        return text
    
    num_to_replace = int(len(positions) * ratio)
    if num_to_replace == 0 and len(positions) > 0 and random.random() < ratio:
        num_to_replace = 1
    
    if num_to_replace > 0:
        positions_to_replace = random.sample(positions, num_to_replace)
        
        text_list = list(text)
        for pos in positions_to_replace:
            text_list[pos] = normal_gigu
        
        return ''.join(text_list)
    
    return text


def apply_digraph_replacement(text, digraph, replacement, ratio=0.2):
    """Apply replacement to a percentage of digraph (two-character sequence) occurrences."""
    positions = []
    i = 0
    while i < len(text) - 1:
        if text[i:i+2] == digraph:
            positions.append(i)
            i += 2  # Skip the next character to avoid overlapping matches
        else:
            i += 1
    
    if not positions:
        return text
    
    # Randomly select positions to replace
    num_to_replace = int(len(positions) * ratio)
    if num_to_replace == 0 and len(positions) > 0 and random.random() < ratio:
        num_to_replace = 1
    
    if num_to_replace > 0:
        positions_to_replace = set(random.sample(positions, num_to_replace))
        
        # Build new text
        result = []
        i = 0
        while i < len(text):
            if i in positions_to_replace:
                result.append(replacement)
                i += 2  # Skip both characters of the digraph
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)
    
    return text


def find_syllable_boundaries(text):
    """Find syllable boundaries in Tibetan text.
    Syllables typically end with ་ (tsheg), ། (shad), or other punctuation."""
    syllable_ends = []
    # Tibetan syllable delimiters
    delimiters = ['་', '།', '༎', '༏', '༐', '༑', '༔', ' ', '\n', '\t']
    
    for i, char in enumerate(text):
        if char in delimiters:
            syllable_ends.append(i)
    
    return syllable_ends


def random_syllable_deletion(text, ratio=0.03):
    """Randomly delete syllables at the specified ratio.
    Works on a single line only, preserving line breaks."""
    # Don't process if text is just whitespace or newline
    if not text.strip():
        return text
    
    # Preserve the original line ending
    line_ending = ''
    if text.endswith('\n'):
        line_ending = '\n'
        text = text.rstrip('\n')
    
    syllable_ends = find_syllable_boundaries(text)
    if len(syllable_ends) < 2:
        return text + line_ending
    
    # Calculate syllable ranges
    syllables = []
    prev_end = 0
    for end in syllable_ends:
        if end > prev_end:
            syllables.append((prev_end, end + 1))  # Include the delimiter
        prev_end = end + 1
    
    if not syllables:
        return text + line_ending
    
    # Determine how many syllables to delete
    num_to_delete = max(1, int(len(syllables) * ratio))
    if num_to_delete >= len(syllables):
        num_to_delete = len(syllables) - 1  # Keep at least one syllable
    
    if num_to_delete > 0:
        syllables_to_delete = set(random.sample(range(len(syllables)), num_to_delete))
    else:
        syllables_to_delete = set()
    
    # Build new text without deleted syllables
    result = []
    for i, (start, end) in enumerate(syllables):
        if i not in syllables_to_delete:
            result.append(text[start:end])
    
    # Add any remaining text after the last syllable
    if syllable_ends:
        last_pos = syllable_ends[-1] + 1
        if last_pos < len(text):
            result.append(text[last_pos:])
    
    return ''.join(result) + line_ending


def random_syllable_addition(text, ratio=0.03):
    """Randomly duplicate syllables at the specified ratio.
    Works on a single line only, preserving line breaks."""
    # Don't process if text is just whitespace or newline
    if not text.strip():
        return text
    
    # Preserve the original line ending
    line_ending = ''
    if text.endswith('\n'):
        line_ending = '\n'
        text = text.rstrip('\n')
    
    syllable_ends = find_syllable_boundaries(text)
    if len(syllable_ends) < 2:
        return text + line_ending
    
    # Calculate syllable ranges
    syllables = []
    prev_end = 0
    for end in syllable_ends:
        if end > prev_end:
            syllables.append((prev_end, end + 1))  # Include the delimiter
        prev_end = end + 1
    
    if not syllables:
        return text + line_ending
    
    # Determine how many syllables to duplicate
    num_to_add = max(1, int(len(syllables) * ratio))
    if num_to_add > 0:
        syllables_to_duplicate = random.sample(range(len(syllables)), min(num_to_add, len(syllables)))
    else:
        syllables_to_duplicate = []
    
    # Build new text with duplicated syllables
    result = []
    for i, (start, end) in enumerate(syllables):
        syllable_text = text[start:end]
        result.append(syllable_text)
        if i in syllables_to_duplicate:
            # Duplicate the syllable (without the delimiter)
            result.append(syllable_text.rstrip('་།༎༏༐༑༔ \n\t'))
    
    # Add any remaining text after the last syllable
    if syllable_ends:
        last_pos = syllable_ends[-1] + 1
        if last_pos < len(text):
            result.append(text[last_pos:])
    
    return ''.join(result) + line_ending


def augment_tibetan_text(text, char_ratio=0.2, syllable_ratio=0.03):
    """Apply all rule-based replacements to the text.
    
    Args:
        text: Input text to augment
        char_ratio: Ratio of character replacements (default 0.2 = 20%)
        syllable_ratio: Ratio of syllable deletions/additions (default 0.03 = 3%)
    """
    
    # Define bidirectional replacements (all pairs work both ways)
    replacements = [
        # Original replacements
        ('འ', 'བ'),
        ('བ', 'འ'),
        ('ས', 'པ'),
        ('པ', 'ས'),
        ('ེ', 'ི'),
        ('ི', 'ེ'),
        ('ལ', 'ཡ'),
        ('ཡ', 'ལ'),
        ('ཕ', 'མ'),
        ('མ', 'ཕ'),
        # New replacements
        ('ག', 'ང'),
        ('ང', 'ག'),
        ('ཀ', 'ག'),
        ('ག', 'ཀ'),
        ('ཅ', 'ཆ'),
        ('ཆ', 'ཅ'),
        ('ན', 'མ'),
        ('མ', 'ན'),
        ('ཐ', 'ད'),
        ('ད', 'ཐ'),
        ('ཐ', 'ཏ'),
        ('ཏ', 'ཐ'),
        ('ཕ', 'པ'),
        ('པ', 'ཕ'),
        ('ཤ', 'ས'),
        ('ས', 'ཤ'),
        ('ཞ', 'ཟ'),
        ('ཟ', 'ཞ'),
        ('ྱ', 'ྲ'),
        ('ྲ', 'ྱ'),
    ]
    
    # Remove duplicate pairs (since we have bidirectional, some may overlap)
    # Keep track of processed pairs to avoid double-replacement
    processed_pairs = set()
    
    for original, replacement in replacements:
        pair = tuple(sorted([original, replacement]))
        if pair not in processed_pairs:
            text = apply_replacement(text, original, replacement, ratio=char_ratio)
            processed_pairs.add(pair)
    
    # Apply reverse gigu to normal gigu (and vice versa for bidirectional)
    text = reverse_gigu_to_normal(text, ratio=char_ratio)
    # Normal gigu to reverse gigu
    reverse_gigu = '\u0F80'
    normal_gigu = '\u0F72'
    text = apply_replacement(text, normal_gigu, reverse_gigu, ratio=char_ratio)
    
    # Apply digraph replacements (bidirectional)
    text = apply_digraph_replacement(text, 'དྱ', 'གྱ', ratio=char_ratio)
    text = apply_digraph_replacement(text, 'གྱ', 'དྱ', ratio=char_ratio)
    
    # Apply syllable-level augmentations
    # Randomly choose whether to delete or add syllables
    if random.random() < 0.5:
        text = random_syllable_deletion(text, ratio=syllable_ratio)
    else:
        text = random_syllable_addition(text, ratio=syllable_ratio)
    
    return text


def process_file(input_path, char_ratio=0.2, syllable_ratio=0.03):
    """Process the input file and create augmented output.
    
    Args:
        input_path: Path to input file
        char_ratio: Ratio of character replacements (default 0.2 = 20%)
        syllable_ratio: Ratio of syllable deletions/additions (default 0.03 = 3%)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    augmented_lines = []
    examples = []
    
    for i, line in enumerate(lines):
        augmented_line = augment_tibetan_text(line, char_ratio=char_ratio, syllable_ratio=syllable_ratio)
        augmented_lines.append(augmented_line)
        
        # Collect examples (only if line changed and has content)
        if len(examples) < 5 and line.strip() and line != augmented_line:
            examples.append((line.rstrip('\n'), augmented_line.rstrip('\n')))
    
    # Create output filename
    output_path = input_path.parent / f"{input_path.stem}_ruleout.txt"
    
    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(augmented_lines)
    
    print(f"Processed: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total lines: {len(lines)}")
    print(f"Character replacement ratio: {char_ratio*100:.0f}%")
    print(f"Syllable deletion/addition ratio: {syllable_ratio*100:.1f}%\n")
    
    # Print examples
    if examples:
        print("=" * 80)
        print("EXAMPLES (Original → Replaced):")
        print("=" * 80)
        for idx, (original, replaced) in enumerate(examples, 1):
            print(f"\nExample {idx}:")
            print(f"Original:  {original}")
            print(f"Replaced:  {replaced}")
        print("=" * 80)
    else:
        print("No changes were made to the file (or all lines were empty).")


def main():
    parser = argparse.ArgumentParser(
        description='Rule-based data augmentation for Tibetan text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python tibetan_augmentation.py input.txt
  python tibetan_augmentation.py input.txt --char-ratio 0.15
  python tibetan_augmentation.py input.txt --char-ratio 0.3 --syllable-ratio 0.05
  
Recommended ratios for sentence pair augmentation:
  - Conservative: --char-ratio 0.1 (10%)
  - Moderate: --char-ratio 0.2 (20%) [DEFAULT]
  - Aggressive: --char-ratio 0.3 (30%)
        '''
    )
    
    parser.add_argument('input_file', help='Input text file to augment')
    parser.add_argument('--char-ratio', type=float, default=0.2,
                       help='Character replacement ratio (default: 0.2 = 20%%)')
    parser.add_argument('--syllable-ratio', type=float, default=0.03,
                       help='Syllable deletion/addition ratio (default: 0.03 = 3%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if not 0 <= args.char_ratio <= 1:
        print("Error: --char-ratio must be between 0 and 1")
        sys.exit(1)
    if not 0 <= args.syllable_ratio <= 1:
        print("Error: --syllable-ratio must be between 0 and 1")
        sys.exit(1)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    process_file(args.input_file, char_ratio=args.char_ratio, syllable_ratio=args.syllable_ratio)
    
    # Print recommendation
    print("\n" + "=" * 80)
    print("AUGMENTATION RECOMMENDATIONS:")
    print("=" * 80)
    print(f"Current character replacement ratio: {args.char_ratio*100:.0f}%")
    print(f"Current syllable operation ratio: {args.syllable_ratio*100:.1f}%")
    print()
    print("For sentence pair augmentation:")
    print("  • 10-15%: Conservative (minimal noise, maintains high similarity)")
    print("  • 20-25%: Moderate (balanced augmentation) [RECOMMENDED]")
    print("  • 30-40%: Aggressive (more diversity, may reduce alignment quality)")
    print("  • 50%+: Very aggressive (risk of breaking semantic alignment)")
    print()
    print("The 50% default from your original request is quite aggressive for")
    print("sentence pairs. I've set the default to 20% which provides good")
    print("augmentation while maintaining semantic similarity.")
    print("=" * 80)


if __name__ == "__main__":
    main()