#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing script for normalised Tibetan texts.
Processes abbreviations and fixes punctuation/spacing issues, following the normalisation conventions of the PaganTibet project. For more see:
Meelen, M., & Griffiths, R. M. (2026). Normalisation Manual (2.0). Zenodo. https://doi.org/10.5281/zenodo.18984001.

The abbreviation dictionary used in Meelen, M. & Griffiths, R.M. (2026) 'Historical Tibetan Normalisation: rule-based vs neural & n-gram LM methods for extremely low-resource languages' in Proceedings of the AI4CHIEF conference, Springer. This is available on HuggingFace: 
https://huggingface.co/datasets/pagantibet/Tibetan-abbreviation-dictionary

Requirements:
- Python 3.6 or higher

Input files:
- Tab-separated dictionary file (.txt) formatted:
  Diplomatic    Normalised
  [abbrev1]     [expansion1]
  [abbrev2]     [expansion2]
"""

import re
from pathlib import Path


def load_abbreviation_dictionary(dict_path):
    """
    Load abbreviation dictionary from a tab-separated file.
    Extracts content within brackets [] from both columns.
    Returns a dictionary sorted by abbreviation length (longest first).
    """
    abbrev_dict = {}
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Skip header if present
        start_idx = 1 if lines and '\t' in lines[0] and 'Diplomatic' in lines[0] else 0
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or '\t' not in line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            # Extract content within brackets
            abbrev_match = re.search(r'\[([^\]]+)\]', parts[0])
            expand_match = re.search(r'\[([^\]]+)\]', parts[1])
            
            if abbrev_match and expand_match:
                abbrev = abbrev_match.group(1)
                expansion = expand_match.group(1)
                
                # Ensure both abbreviation and expansion end with tsheg (་)
                if abbrev and not abbrev.endswith('་'):
                    abbrev += '་'
                if expansion and not expansion.endswith('་'):
                    expansion += '་'
                
                abbrev_dict[abbrev] = expansion
    
    # Sort by length (longest first) to handle overlapping abbreviations
    sorted_dict = dict(sorted(abbrev_dict.items(), key=lambda x: len(x[0]), reverse=True))
    
    return sorted_dict


def expand_abbreviations(text, abbrev_dict):
    """
    Expand abbreviations in text using exact matches.
    Processes longest abbreviations first to avoid partial replacements and overlapping.
    Returns the modified text and a list of changes made.
    """
    changes = []
    
    for abbrev, expansion in abbrev_dict.items():
        # Count occurrences before replacement
        count = text.count(abbrev)
        if count > 0:
            text = text.replace(abbrev, expansion)
            changes.append({
                'type': 'abbreviation',
                'from': abbrev,
                'to': expansion,
                'count': count
            })
    
    return text, changes


def fix_punctuation_spacing(text):
    """
    Fix punctuation and spacing issues:
    1. Change ༑ and ༏ to །
    2. Add blank space after ། if not already present with the exception of double shad །།
    3. Remove double tsheg (་་ to ་)
    Returns the modified text and a list of changes made.
    """
    changes = []
    
    # Step 1: Change ༑ and ༏ to །
    count_1 = text.count('༑')
    if count_1 > 0:
        text = text.replace('༑', '།')
        changes.append({
            'type': 'punctuation',
            'from': '༑',
            'to': '།',
            'count': count_1
        })
    
    count_2 = text.count('༏')
    if count_2 > 0:
        text = text.replace('༏', '།')
        changes.append({
            'type': 'punctuation',
            'from': '༏',
            'to': '།',
            'count': count_2
        })
    
    # Step 2: Add space after ། if not already present
    # Exception: Do not add space between double shad (།།)
    original = text
    text = re.sub(r'།(?![།\s])', '། ', text)

    if text != original:
        space_count = text.count('། ') - original.count('། ')
        if space_count > 0:
            changes.append({
                'type': 'spacing',
                'from': '།',
                'to': '། ',
                'count': space_count
            })
    
    # Step 3: Replace double tsheg (་་) with single tsheg (་)
    double_count = 0
    while '་་' in text:
        double_count += text.count('་་')
        text = text.replace('་་', '་')
    
    if double_count > 0:
        changes.append({
            'type': 'tsheg',
            'from': '་་',
            'to': '་',
            'count': double_count
        })
    
    return text, changes


def postprocess_tibetan_text(input_text, abbrev_dict):
    """
    Main postprocessing function.
    Applies all transformations in the correct order.
    Returns the processed text and a list of all changes made.
    """
    all_changes = []
    
    # Step 1: Expand abbreviations (done first)
    text, abbrev_changes = expand_abbreviations(input_text, abbrev_dict)
    all_changes.extend(abbrev_changes)
    
    # Step 2: Fix punctuation and spacing
    text, punct_changes = fix_punctuation_spacing(text)
    all_changes.extend(punct_changes)
    
    return text, all_changes


def process_file(input_path, output_path, dict_path):
    """
    Process a file with the postprocessing pipeline.
    """
    # Load dictionary
    abbrev_dict = load_abbreviation_dictionary(dict_path)
    print(f"Loaded {len(abbrev_dict)} abbreviations from dictionary")
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process text
    processed_text, changes = postprocess_tibetan_text(text, abbrev_dict)
    
    # Print first 10 changes. This number can be changed as needed.
    print("\n" + "="*60)
    print("FIRST 10 CHANGES MADE:")
    print("="*60)
    
    for i, change in enumerate(changes[:10], 1):
        change_type = change['type'].upper()
        from_str = change['from']
        to_str = change['to']
        count = change['count']
        
        print(f"{i}. [{change_type}] '{from_str}' → '{to_str}' ({count} occurrence{'s' if count > 1 else ''})")
    
    if len(changes) > 10:
        print(f"\n... and {len(changes) - 10} more changes")
    
    print(f"\nTotal changes made: {len(changes)}")
    print("="*60 + "\n")
    
    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    print(f"Processed text written to {output_path}")
    
    return processed_text, changes


def main():
    """
    Example usage of the postprocessing script.
    """
    # Example with your test case
    dict_data = {
        'རིལུ་': 'རིལ་བུ་',
        'རིརྴོར་': 'རིར་ཤོར་',
        'རིསྃ་': 'རིམས་',
        'རུསཆགདནྱོདེགི་': 'རུས་ཆག་འདོན་བྱེད་ཅིག་',
        'རུསའི་': 'རུས་པའི་',
        'རུསྴིག་': 'རུས་ཤིག་',
        'རཻ་': 'རེ་རེ་',
        'ཀྕོུ་': 'ཀུ་ཅོ་',
        'ཀྶྒྲི་': 'ཀི་སྒྲ་',
        'རིེན་': 'རིན་ཆེན་',
        'ཐཾད་': 'ཐམས་ཅད་',
        'སློོབ་': 'སློབ་དཔོན་'
    }
    
    test_text = "ཕྱིར༑ རིེན་རིལ་ཐཾད་རུས་སློོབ་"
    print("Original text:")
    print(test_text)
    print("\nProcessed text:")
    processed, changes = postprocess_tibetan_text(test_text, dict_data)
    print(processed)
    print("\nExpected:")
    print("ཕྱིར། རིན་ཆེན་རིལ་ཐམས་ཅད་རུས་སློབ་དཔོན་")
    
    print("\n" + "="*60)
    print("CHANGES MADE:")
    print("="*60)
    for i, change in enumerate(changes, 1):
        print(f"{i}. [{change['type'].upper()}] '{change['from']}' → '{change['to']}' ({change['count']} occurrence{'s' if change['count'] > 1 else ''})")
    print("="*60)
    
    # To process actual files:
    process_file(
         input_path='input_file.txt',
         output_path='output_tibetan.txt',
         dict_path='abbreviation_file.txt'
     )


if __name__ == '__main__':
    main()
