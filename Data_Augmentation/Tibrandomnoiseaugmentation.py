#!/usr/bin/env python3
"""
Tibetan Data Augmentation with Random Noise Insertion

This script implements noise insertion for Tibetan text based on the formula:
P(noise) = len(s) × (ratio / max_text_length) / 100

where:
- len(s) is the length of the current text
- ratio is a parameter controlling noise intensity
- max_text_length is the maximum text length in the dataset
"""

import random
from typing import List, Optional


class TibetanAugmenter:
    """
    Augmenter for Tibetan text data using random noise insertion.
    """
    
    # Common Tibetan Unicode characters for noise insertion
    TIBETAN_CHARACTERS = [
        'ཀ', 'ཁ', 'ག', 'ང', 'ཅ', 'ཆ', 'ཇ', 'ཉ', 'ཏ', 'ཐ', 'ད', 'ན', 'པ', 'ཕ', 'བ', 'མ',
        'ཙ', 'ཚ', 'ཛ', 'ཝ', 'ཞ', 'ཟ', 'འ', 'ཡ', 'ར', 'ལ', 'ཤ', 'ས', 'ཧ', 'ཨ',
        '་', '།', '༄', '༅', 'ི', 'ུ', 'ེ', 'ོ', 'ྀ', 'ཱ', 'ྭ', 'ྱ', 'ྲ', 'ླ'
    ]
    
    def __init__(self, ratio: float = 10.0, max_text_length: int = 1000):
        """
        Initialize the Tibetan augmenter.
        
        Args:
            ratio: Controls the noise intensity (default: 10.0)
            max_text_length: Maximum text length in the dataset (default: 1000)
        """
        self.ratio = ratio
        self.max_text_length = max_text_length
    
    def calculate_noise_probability(self, text: str) -> float:
        """
        Calculate the probability of noise insertion based on the formula:
        P(noise) = len(s) × (ratio / max_text_length) / 100
        
        Args:
            text: Input text string
            
        Returns:
            Probability value between 0 and 1
        """
        text_length = len(text)
        # probability = (text_length * (self.ratio / self.max_text_length)) / 100
        probability = self.ratio / 100 #used this to make sure short lines would get noise too
        return min(probability, 1.0)  # Cap at 1.0
    
    def insert_noise(self, text: str, noise_chars: Optional[List[str]] = None) -> str:
        """
        Insert random noise into the text based on calculated probability.
        
        Args:
            text: Input Tibetan text
            noise_chars: Optional list of characters to use as noise
                        (defaults to TIBETAN_CHARACTERS)
            
        Returns:
            Augmented text with inserted noise
        """
        if not text:
            return text
        
        if noise_chars is None:
            noise_chars = self.TIBETAN_CHARACTERS
        
        # Calculate noise probability
        noise_prob = self.calculate_noise_probability(text)
        
        # Convert text to list for easier manipulation
        chars = list(text)
        result = []
        
        for char in chars:
            result.append(char)
            
            # Decide whether to insert noise after this character
            if random.random() < noise_prob:
                noise_char = random.choice(noise_chars)
                result.append(noise_char)
        
        return ''.join(result)
    
    def augment_batch(self, texts: List[str], num_augmentations: int = 1) -> List[str]:
        """
        Augment a batch of texts.
        
        Args:
            texts: List of input texts
            num_augmentations: Number of augmented versions per text
            
        Returns:
            List of augmented texts
        """
        augmented = []
        for text in texts:
            for _ in range(num_augmentations):
                augmented.append(self.insert_noise(text))
        return augmented


def process_file(input_file: str, ratio: float = 10.0, max_text_length: int = 1000):
    """
    Process a text file and create an augmented output file.
    
    Args:
        input_file: Path to input .txt file
        ratio: Controls the noise intensity (default: 10.0)
        max_text_length: Maximum text length in the dataset (default: 1000)
    """
    import os
    
    # Create output filename: remove .txt and add _noiseout.txt
    if input_file.endswith('.txt'):
        output_file = input_file[:-4] + '_noiseout.txt'
    else:
        output_file = input_file + '_noiseout.txt'
    
    # Initialize augmenter
    augmenter = TibetanAugmenter(ratio=ratio, max_text_length=max_text_length)
    
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Process each line
    augmented_lines = []
    for line in lines:
        # Remove newline, process, then add it back
        line_content = line.rstrip('\n')
        if line_content:  # Only process non-empty lines
            augmented = augmenter.insert_noise(line_content)
            augmented_lines.append(augmented + '\n')
        else:  # Keep empty lines as-is
            augmented_lines.append('\n')
    
    # Write output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(augmented_lines)
        print(f"Successfully processed {len(lines)} lines")
        print(f"Input file:  {input_file}")
        print(f"Output file: {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}")


def main():
    """
    Main function with command-line interface.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tibetan_augmentation.py <input_file.txt> [ratio] [max_text_length]")
        print("\nExample:")
        print("  python tibetan_augmentation.py input.txt")
        print("  python tibetan_augmentation.py input.txt 15.0 100")
        print("\nParameters:")
        print("  ratio: Controls noise intensity (default: 10.0)")
        print("  max_text_length: Maximum text length in dataset (default: 1000)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    max_text_length = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    process_file(input_file, ratio, max_text_length)


if __name__ == "__main__":
    main()