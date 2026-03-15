#!/usr/bin/env python3
"""
Tokenize source and target files using botok tokenizer (optimized for large files).
Processes train_source.txt and train_target.txt, creating tokenized versions
with "-tok" suffix while preserving line structure.

This version should have the tsa rtags and other non-punctuation marks that botok gets wrong fixed.

Main functionality is for tokenising GoldNorm data so it takes both source and target files as input.
So make sure the input files are called train-source.txt and train-target.txt
The benefit is that it checks if the number of lines in both is still the same, 
which is essential for creating training data for the encoder-decoder.

python3 botokenise_src-tgt.py

Also added an option for single file input:
python3 botokenise_src-tgt.py my_file.txt

"""


#!/usr/bin/env python3
"""
Tokenize source and target files using botok tokenizer (optimized for large files).
Processes train_source.txt and train_target.txt, creating tokenized versions
with "-tok" suffix while preserving line structure.

Can also process a single file by providing the filename as an argument.
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

try:
    from botok import WordTokenizer
except ImportError:
    print("Error: botok is not installed.")
    print("Please install it using: pip install botok --break-system-packages")
    sys.exit(1)

# Suppress botok warnings about non-expanded chars
warnings.filterwarnings('ignore', category=UserWarning, module='botok')


def format_bytes(bytes_size):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def tokenize_file(input_path, output_path, progress_interval=10000):
    """
    Tokenize a file using botok, preserving line structure.
    Optimized for large files with progress reporting.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        progress_interval: Report progress every N lines
    """
    print(f"Processing {input_path}...")
    
    # Get file size for progress reporting
    file_size = input_path.stat().st_size
    print(f"  File size: {format_bytes(file_size)}")
    
    # Create error log file path
    error_log_path = output_path.parent / f"{output_path.stem}-errors.txt"
    
    # Initialize the tokenizer once
    tokenizer = WordTokenizer()
    
    line_count = 0
    error_count = 0
    start_time = time.time()
    bytes_processed = 0
    
    with open(input_path, 'r', encoding='utf-8', buffering=8192*1024) as infile, \
         open(output_path, 'w', encoding='utf-8', buffering=8192*1024) as outfile, \
         open(error_log_path, 'w', encoding='utf-8') as error_log:
        
        # Write header to error log
        error_log.write(f"Error log for: {input_path}\n")
        error_log.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        error_log.write("=" * 80 + "\n\n")
        
        for line in infile:
            line_count += 1
            bytes_processed += len(line.encode('utf-8'))
            
            # Strip trailing whitespace but preserve empty lines
            line = line.rstrip('\n\r')
            
            if line.strip():  # Non-empty line
                try:
                    # Tokenize the line
                    tokens = tokenizer.tokenize(line)
                    # Extract text from tokens and join with spaces
                    tokenized_text = ' '.join([token.text for token in tokens])
                    
                    # Fix botok's incorrect treatment of U+0F37 (༷) and U+0F39 (༹)
                    # These are marks, not punctuation, so they shouldn't have spaces around them
                    tokenized_text = tokenized_text.replace(' ༷ ', '༷')
                    tokenized_text = tokenized_text.replace(' ༷', '༷')
                    tokenized_text = tokenized_text.replace('༷ ', '༷')
                    tokenized_text = tokenized_text.replace(' ༹ ', '༹')
                    tokenized_text = tokenized_text.replace(' ༹', '༹')
                    tokenized_text = tokenized_text.replace('༹ ', '༹')
                    
                    outfile.write(tokenized_text + '\n')
                except Exception as e:
                    # If tokenization fails, write the original line unchanged
                    error_count += 1
                    outfile.write(line + '\n')
                    
                    # Log error to file
                    error_log.write(f"Line {line_count}:\n")
                    error_log.write(f"Error: {str(e)}\n")
                    error_log.write(f"Content: {line}\n")
                    error_log.write("-" * 80 + "\n\n")
                    
                    # Log first few errors to console
                    if error_count <= 5:
                        print(f"  Warning: Error on line {line_count}: {str(e)[:100]}")
                    elif error_count == 6:
                        print(f"  (Further errors will be logged to {error_log_path.name})")
            else:  # Empty line
                outfile.write('\n')
            
            # Progress reporting
            if line_count % progress_interval == 0:
                elapsed = time.time() - start_time
                lines_per_sec = line_count / elapsed if elapsed > 0 else 0
                percent = (bytes_processed / file_size * 100) if file_size > 0 else 0
                error_msg = f" - {error_count} errors" if error_count > 0 else ""
                print(f"  Progress: {line_count:,} lines ({percent:.1f}%) - "
                      f"{lines_per_sec:.0f} lines/sec - "
                      f"{format_bytes(bytes_processed)}/{format_bytes(file_size)}{error_msg}")
    
    elapsed = time.time() - start_time
    print(f"  Completed: {line_count:,} lines in {elapsed:.1f} seconds")
    print(f"  Average speed: {line_count/elapsed:.0f} lines/sec")
    if error_count > 0:
        print(f"  Errors encountered: {error_count} lines (written unchanged)")
        print(f"  Error log saved to: {error_log_path}")
    else:
        # Remove empty error log if no errors
        error_log_path.unlink()
        print(f"  No errors encountered")
    print(f"  Output written to {output_path}")


def main():
    """Main function to process both source and target files, or a single file."""
    parser = argparse.ArgumentParser(
        description='Tokenize Tibetan text files using botok tokenizer.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process both source and target files (default)
  python tokenize_files.py
  
  # Process a single file
  python tokenize_files.py my_file.txt
  
  # Process a single file with custom output name
  python tokenize_files.py input.txt output-tok.txt
        """
    )
    parser.add_argument('input_file', nargs='?', help='Input file to tokenize (optional)')
    parser.add_argument('output_file', nargs='?', help='Output file name (optional, default: input_file-tok.txt)')
    
    args = parser.parse_args()
    
    total_start = time.time()
    
    # If a specific file is provided, process only that file
    if args.input_file:
        input_path = Path(args.input_file)
        
        if not input_path.exists():
            print(f"Error: {args.input_file} not found")
            sys.exit(1)
        
        # Determine output filename
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            # Insert "-tok" before the extension
            stem = input_path.stem
            suffix = input_path.suffix
            output_path = input_path.parent / f"{stem}-tok{suffix}"
        
        try:
            tokenize_file(input_path, output_path)
        except Exception as e:
            print(f"Error processing {args.input_file}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Otherwise, process default source and target files
    else:
        files_to_process = [
            ('train_source.txt', 'train_source-tok.txt'),
            ('train_target.txt', 'train_target-tok.txt')
        ]
        
        for input_file, output_file in files_to_process:
            input_path = Path(input_file)
            output_path = Path(output_file)
            
            # Check if input file exists
            if not input_path.exists():
                print(f"Warning: {input_file} not found, skipping...")
                continue
            
            try:
                tokenize_file(input_path, output_path)
                print()  # Blank line between files
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    total_elapsed = time.time() - total_start
    print(f"Total processing time: {total_elapsed:.1f} seconds")
    print("Tokenization complete!")


if __name__ == "__main__":
    main()