# Tibetan Text Normalisation - Unified Inference System

Complete inference system for Tibetan text normalisation supporting **6 different approaches**: neural seq2seq, language model (LM) ranking, rule-based normalisation, and various combinations.

This was developed as part of [PaganTibet](https://www.pagantibet.com/)'s Normalisation workflow. For more information, see our [Normalisation README](https://github.com/pagantibet/normalisation/tree/main?tab=readme-ov-file).

## Table of Contents

- [Quick Start](#quick-start)
- [The 6 Modes](#the-6-modes)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [File Requirements](#file-requirements)
- [Performance Guide](#performance-guide)
- [Troubleshooting](#troubleshooting)


## Quick Start

### Neural + LM + Rules (full pipeline - recommended)
```bash
python3 tibetan_inference_with_rules.py \
    --mode neural+lm+rules \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --rules_dict abbreviations.txt \
    --input_file GoldTest_source.txt
```

Output will be automatically saved to `GoldTest_source_prediction.txt`.



## The 6 Modes

### Mode 1: `rules` - Rule-based Only
**What it does:** Dictionary-based abbreviation expansion + punctuation fixes  
**Best for:** Baseline comparison, very fast processing  
**Speed:** Very fast (~1000+ texts/sec)

```bash
python3 tibetan_inference_with_rules.py \
    --mode rules \
    --rules_dict abbreviations.txt \
    --input_file input.txt
```

**What the rules do:**
1. Expand abbreviations from dictionary
2. Fix punctuation: ༑ and ༎ → །
3. Add spaces after །
4. Remove double tsheg: ་་ → ་

**Requirements:**
- Abbreviation dictionary file (tab-separated)



### Mode 2: `neural` - Seq2seq Only
**What it does:** Pure neural sequence-to-sequence normalization  
**Best for:** Fast inference, when you trust your model  
**Speed:** Very fast (~100-200 texts/sec on GPU)

```bash
python3 tibetan_inference_with_rules.py \
    --mode neural \
    --model_path tibetan_model_6000.pt \
    --input_file input.txt
```



### Mode 2: `neural` - Seq2seq Only
**What it does:** Pure neural sequence-to-sequence normalization  
**Best for:** Fast inference, when you trust your model  
**Speed:** Very fast (~100-200 texts/sec on GPU)

```bash
python3 tibetan_inference_with_rules.py \
    --mode neural \
    --model_path tibetan_model_6000.pt \
    --input_file input.txt
```

**Requirements:**
- Seq2seq model file (`.pt`)



### Mode 3: `neural+lm` - Seq2seq + KenLM
**What it does:** Neural normalization with language model reranking  
**Best for:** Better quality when you have a strong language model  
**Speed:** Fast (~50-100 texts/sec with KenLM, ~5-20 with Python LM)

```bash
python3 tibetan_inference_with_rules.py \
    --mode neural+lm \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --input_file input.txt
```

**Requirements:**
- Seq2seq model file (`.pt`)
- KenLM ARPA file (`.arpa`)



### Mode 3: `neural+lm` - Seq2seq + KenLM
**What it does:** Neural normalization with language model reranking  
**Best for:** Better quality when you have a strong language model  
**Speed:** Fast (~50-100 texts/sec with KenLM, ~5-20 with Python LM)

```bash
python3 tibetan_inference_with_rules.py \
    --mode neural+lm \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --input_file input.txt
```

**Requirements:**
- Seq2seq model file (`.pt`)
- KenLM ARPA file (`.arpa`)

**LM Backend Options:**
- `--lm_backend kenlm` - Fast (requires KenLM installation)
- `--lm_backend python` - Slower but no installation needed
- `--lm_backend auto` - Auto-detect (default)


### Mode 4: `neural+lm+rules` - Neural + LM → Rules - RECOMMENDED
**What it does:** Neural + LM normalization, then rules as postprocessing  
**Best for:** Highest quality, uses all available resources  
**Speed:** Moderate (~40-80 texts/sec)

```bash
python3 tibetan_inference_with_rules.py \
    --mode neural+lm+rules \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --rules_dict abbreviations.txt \
    --input_file input.txt
```

**Pipeline:**
```
Input → Neural model → KenLM reranking → Rules postprocessing → Output
```

**Requirements:**
- Seq2seq model file (`.pt`)
- KenLM ARPA file (`.arpa`)
- Abbreviation dictionary file



### Mode 5: `rules+neural+lm` - Rules → Neural + LM 
**What it does:** Apply rules as preprocessing, then neural + KenLM normalization  
**Best for:** When input has abbreviations to expand before neural processing  
**Speed:** Moderate (~40-80 texts/sec)

```bash
python3 tibetan_inference_with_rules.py \
    --mode rules+neural+lm \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --rules_dict abbreviations.txt \
    --input_file input.txt
```

**Pipeline:**
```
Input → Rules preprocessing → Neural model → KenLM reranking → Output
```

**Requirements:**
- Seq2seq model file (`.pt`)
- KenLM ARPA file (`.arpa`)
- Abbreviation dictionary file



### Mode 6: `rules+neural` - Rules → Neural
**What it does:** Apply rules as preprocessing, then neural normalization  
**Best for:** When you want rules to clean input before neural processing  
**Speed:** Fast (~80-150 texts/sec)

```bash
python3 tibetan_inference_with_rules.py \
    --mode rules+neural \
    --model_path tibetan_model_6000.pt \
    --rules_dict abbreviations.txt \
    --input_file input.txt
```

**Pipeline:**
```
Input → Rules preprocessing → Neural model → Output
```

**Requirements:**
- Seq2seq model file (`.pt`)
- Abbreviation dictionary file



## Installation

### Basic Requirements (Always Needed)
```bash
pip install torch
```

### Optional: KenLM (for fastest LM speed)
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libboost-all-dev
pip install https://github.com/kpu/kenlm/archive/master.zip

# macOS
xcode-select --install
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Optional: Pure Python ARPA (no installation needed)
If you can't install KenLM, the script includes a pure Python ARPA reader. Just make sure `arpa_lm_python.py` is in the same directory.



## Usage Examples

### Example 1: Single Text Normalization
```bash
python3 tibetan_inference_with_rules.py \
    --mode neural \
    --model_path tibetan_model_6000.pt \
    --text "བོད་ཡིག་གི་སྐད་ཡིག"
```

Output:
```
Input: བོད་ཡིག་གི་སྐད་ཡིག
Output: བོད་ཡིག་གི་སྐད་ཡིག་
Time: 0.023s
```

### Example 2: Batch Processing (Auto Output Filename)
```bash
python3 tibetan_inference_with_rules.py \
    --mode neural+lm+rules \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --rules_dict abbreviations.txt \
    --input_file test_data.txt
```

Output will be saved to: `test_data_prediction.txt` (automatic!)

### Example 3: Custom Output Filename
```bash
python3 tibetan_inference_with_rules.py \
    --mode neural+lm \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --input_file test_data.txt \
    --output_file my_results.txt
```

### Example 4: Interactive Mode
```bash
python3 tibetan_inference_with_rules.py \
    --mode neural \
    --model_path tibetan_model_6000.pt \
    --interactive
```

Then type your text and get instant results:
```
Interactive mode (Ctrl+C or 'quit' to exit)
Mode: neural
------------------------------------------------------------

Input: བོད་ཡིག
Output: བོད་ཡིག་ (0.023s)

Input: quit
Exiting...
```

### Example 5: Comparing All Modes
```bash
# Run all 5 modes on the same data
for mode in neural neural+lm rules rules+neural neural+lm+rules; do
    python3 tibetan_inference_with_rules.py \
        --mode $mode \
        --model_path tibetan_model_6000.pt \
        --kenlm_path model_5gram_char.arpa \
        --lm_backend python \
        --rules_dict abbreviations.txt \
        --input_file test.txt \
        --output_file results_${mode}.txt
done
```



## File Requirements

### 1. Seq2seq Model File (`.pt`)
Your trained neural model. The script auto-detects architecture from the checkpoint.

**Required for modes:** `neural`, `neural+lm`, `rules+neural`, `neural+lm+rules`

**Check your model:**
```bash
python3 -c "import torch; c=torch.load('model.pt', map_location='cpu', weights_only=False); print(c['args'])"
```

### 2. KenLM ARPA File (`.arpa`)
Character-based language model in ARPA format.

**Required for modes:** `neural+lm`, `neural+lm+rules`

**Format:** Standard ARPA format (5-gram recommended)

**Note:** `.bin` files won't work with pure Python backend, only `.arpa`

### 3. Abbreviation Dictionary (`.txt`)
Tab-separated file with diplomatic → normalized mappings.

**Required for modes:** `rules`, `rules+neural`, `neural+lm+rules`

**Format:**
```
Diplomatic    Normalised
[རྡོ་]        [རྡོ་རྗེ་]
[རིག་]       [རིགས་]
```

**Example:**
```
Diplomatic    Normalised
[རི་རུ་]      [རི་རུ་བ་]
[ཚང་པ་]      [ཚང་པ་བ་]
```

### 4. Input File Format
Plain text file, one sentence per line, UTF-8 encoding.

**Example:**
```
བོད་ཡིག་གི་སྐད་ཡིག
གཞན་ཡང་འདི་ལྟར་བྱས
དེ་བཞིན་དུ་གསུངས་སོ
```



## Advanced Options

### Beam Search Parameters

**`--beam_width`** (default: 5)
- Controls how many hypotheses to explore
- Higher = better quality but slower
- Typical range: 3-10

```bash
# Fast (lower quality)
--beam_width 3

# Balanced (default)
--beam_width 5

# High quality (slower)
--beam_width 10
```

### KenLM Weight

**`--lm_weight`** (default: 0.2)
- Controls influence of language model
- Range: 0.0-1.0
- Your 8M KenLM is strong, so 0.2-0.3 works well

```bash
# Trust neural more
--lm_weight 0.1

# Balanced
--lm_weight 0.2

# Trust LM more
--lm_weight 0.3
```

### Length Penalty

**`--length_penalty`** (default: 0.6)
- Prevents bias towards shorter outputs
- Higher = prefer longer outputs
- Typical range: 0.5-0.8

```bash
# Prefer shorter
--length_penalty 0.5

# Balanced
--length_penalty 0.6

# Prefer longer
--length_penalty 0.7
```

### Example with All Parameters
```bash
python3 tibetan_inference_with_rules.py \
    --mode neural+lm+rules \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --rules_dict abbreviations.txt \
    --input_file test.txt \
    --beam_width 7 \
    --lm_weight 0.25 \
    --length_penalty 0.7
```



## Performance Guide

### Speed Expectations (200 texts)

| Mode | GPU Time | CPU Time | Quality |
|------|----------|----------|---------|
| 1. `rules` | 0.2 sec | 0.5 sec | Baseline |
| 2. `neural` | 30 sec | 2-5 min | Good |
| 3. `neural+lm` (KenLM) | 60 sec | 5-10 min | Better |
| 3. `neural+lm` (Python) | 2-4 min | 10-20 min | Better |
| 4. `neural+lm+rules` | 90 sec | 8-15 min | Best |
| 5. `rules+neural+lm` | 90 sec | 8-15 min | Best |
| 6. `rules+neural` | 40 sec | 3-6 min | Good |

### Optimization Tips

**For Speed:**
```bash
# Use GPU (automatic if available)
# Reduce beam width
--beam_width 3

# Skip KenLM or use KenLM instead of Python
--mode neural
```

**For Quality:**
```bash
# Full pipeline
--mode neural+lm+rules

# Increase beam width
--beam_width 10

# Tune LM weight
--lm_weight 0.25
```

### Memory Usage

- **Model**: 100-500 MB (depends on d_model)
- **KenLM**: 2-3 GB (for 8M lines)
- **Total GPU**: 3-5 GB recommended
- **CPU only**: Works but 10x slower



## Troubleshooting

### Problem: Model architecture mismatch error
```
RuntimeError: Error(s) in loading state_dict for TransformerModel
```

**Solution:** Check which model you're using:
```bash
python3 -c "import torch; c=torch.load('model.pt', map_location='cpu', weights_only=False); print(c['args'].train_src, c['args'].train_tgt)"
```

Make sure:
- Model trained on non-tokenized data matches non-tokenized test data
- Model trained on tokenized data matches tokenized test data

### Problem: Random spaces in output
```
Output: ལག་ནས་ ཕྱ་ཁུག་ ཅིག །
```

**Cause:** Model trained on tokenized data, testing on non-tokenized  
**Solution:** Use the correct model (check training args above)

### Problem: KenLM not found
```
ImportError: No module named 'kenlm'
```

**Solution 1:** Install KenLM (see Installation section)  
**Solution 2:** Use pure Python backend:
```bash
--lm_backend python
```

### Problem: ARPA file loading error with Python backend
```
Error loading Python ARPA
```

**Check:**
- File is `.arpa` format (not `.bin`)
- File path is correct
- File encoding is UTF-8

### Problem: Very slow inference
```
Processing 200 texts taking 30+ minutes
```

**Check:**
1. Using GPU? → `Using device: cuda` should appear
2. Beam width too high? → Try `--beam_width 3`
3. Python LM backend? → Install KenLM for 10x speedup
4. Model has 6 layers? → More layers = slower

**Quick fix:**
```bash
# Use fastest settings
--mode neural \
--beam_width 3
```

### Problem: Out of memory on GPU
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce beam width
--beam_width 3

# Or force CPU
CUDA_VISIBLE_DEVICES="" python3 tibetan_inference_with_rules.py ...
```

### Problem: `<unk>` tokens in output
**Solution:** This is fixed in the current version. The script blocks `<unk>` during generation. If you still see them, you're using an old version.



## Comparing Modes

To systematically compare all modes:

```bash
#!/bin/bash
# compare_modes.sh

INPUT="GoldTest_source.txt"
MODEL="tibetan_model_6000.pt"
KENLM="model_5gram_char.arpa"
RULES="abbreviations.txt"

# Mode 1: Rules only
python3 tibetan_inference_with_rules.py \
    --mode rules \
    --rules_dict $RULES \
    --input_file $INPUT \
    --output_file results_rules.txt

# Mode 2: Neural only
python3 tibetan_inference_with_rules.py \
    --mode neural \
    --model_path $MODEL \
    --input_file $INPUT \
    --output_file results_neural.txt

# Mode 3: Neural + LM
python3 tibetan_inference_with_rules.py \
    --mode neural+lm \
    --model_path $MODEL \
    --kenlm_path $KENLM \
    --lm_backend python \
    --input_file $INPUT \
    --output_file results_neural_lm.txt

# Mode 4: Neural + LM + Rules
python3 tibetan_inference_with_rules.py \
    --mode neural+lm+rules \
    --model_path $MODEL \
    --kenlm_path $KENLM \
    --lm_backend python \
    --rules_dict $RULES \
    --input_file $INPUT \
    --output_file results_neural_lm_rules.txt

# Mode 5: Rules + Neural + LM
python3 tibetan_inference_with_rules.py \
    --mode rules+neural+lm \
    --model_path $MODEL \
    --kenlm_path $KENLM \
    --lm_backend python \
    --rules_dict $RULES \
    --input_file $INPUT \
    --output_file results_rules_neural_lm.txt

# Mode 6: Rules + Neural
python3 tibetan_inference_with_rules.py \
    --mode rules+neural \
    --model_path $MODEL \
    --rules_dict $RULES \
    --input_file $INPUT \
    --output_file results_rules_neural.txt

echo "All modes complete! Compare results_*.txt files"
```

Then evaluate with your metrics script.



## Which Mode Should I Use?

### For Research/Publication
**Use:** `neural+lm+rules` (Mode 4) or `rules+neural+lm` (Mode 5)
- Highest quality
- Uses all available information
- Mode 4 if neural output needs cleanup, Mode 5 if input needs preprocessing
- Worth the extra computation time

### For Production/Large-Scale Processing
**Use:** `neural+lm` (Mode 3) with KenLM
- Good balance of quality and speed
- ~50-100 texts/sec with KenLM installed
- Reliable and well-tested

### For Real-time Applications
**Use:** `neural` (Mode 2)
- Fastest neural option
- ~100-200 texts/sec
- Still produces good results

### For Baseline Comparison
**Use:** `rules` (Mode 1)
- See what simple rules can achieve
- Very fast
- Good for understanding what neural adds

### For Ablation Studies
**Compare all 6 modes** to understand:
- What does the neural model contribute?
- How much does KenLM help?
- Do rules add value?
- Does rules preprocessing or postprocessing work better?
- What's the best combination?



## Example Workflow

```bash
# 1. Check your model
python3 -c "import torch; c=torch.load('tibetan_model_6000.pt', map_location='cpu', weights_only=False); print(c['args'])"

# 2. Run quick test on single text
python3 tibetan_inference_with_rules.py \
    --mode neural \
    --model_path tibetan_model_6000.pt \
    --text "བོད་ཡིག"

# 3. Run full pipeline on test set
python3 tibetan_inference_with_rules.py \
    --mode neural+lm+rules \
    --model_path tibetan_model_6000.pt \
    --kenlm_path model_5gram_char.arpa \
    --lm_backend python \
    --rules_dict abbreviations.txt \
    --input_file GoldTest_source.txt

# 4. Output automatically saved to GoldTest_source_prediction.txt

# 5. Evaluate with your metrics
python3 evaluate.py \
    --predictions GoldTest_source_prediction.txt \
    --references GoldTest_target.txt
```



##  Additional Files

### Other Scripts Available:

1. **`arpa_lm_python.py`** - Pure Python ARPA reader (must be in same directory)
2. **`tibetan_inference_flexible.py`** - Older version without rules support
3. **`tibetan_inference_kenlm.py`** - Original script (neural + KenLM only)

### Recommended: Use `tibetan_inference_with_rules.py`
It's the most complete and flexible version.



## Support

If you encounter issues:

1. Check model was trained on correct data format (tokenized vs non-tokenized)
2. Verify all required files are present
3. Try the simplest mode first (`--mode neural`)
4. Check GPU is being used (`Using device: cuda`)
5. Review the Troubleshooting section above

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pagantibet/normalisation/blob/main/LICENSE) file for details.
