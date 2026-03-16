"""
Enhanced inference script for Tibetan text normalization with six modes:
1. Rule-based only (dictionary + punctuation rules)
2. Seq2seq only (neural model)
3. Seq2seq + KenLM (neural + language model)
4. Neural + LM + Rules (neural + LM, then rules postprocessing)
5. Rules + Neural + LM (rules preprocessing, then neural + LM)
6. Rules + Neural (rules preprocessing, then neural)

Usage:
    # Rule-based only
    python3 tibetan-inference-flexible.py --mode rules --rules_dict abbreviations.txt
    
    # Neural only
    python3 tibetan-inference-flexible.py --model_path model.pt --mode neural
    
    # Neural + KenLM
    python3 tibetan-inference-flexible.py --model_path model.pt --kenlm_path model.arpa --mode neural+lm
    
    # Neural + LM + Rules postprocessing
    python3 tibetan-inference-flexible.py --model_path model.pt --kenlm_path model.arpa --rules_dict abbrev.txt --mode neural+lm+rules
    
    # Rules preprocessing + Neural + LM
    python3 tibetan-inference-flexible.py --model_path model.pt --kenlm_path model.arpa --rules_dict abbrev.txt --mode rules+neural+lm
    
    # Rules preprocessing + Neural
    python3 tibetan-inference-flexible.py --model_path model.pt --rules_dict abbrev.txt --mode rules+neural
"""


import torch
import torch.nn as nn
import math
import argparse
import sys
import time
import re
from datetime import datetime

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)
import numpy as np
np.random.seed(RANDOM_SEED)

# Try to import KenLM
try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False

# Try to import pure Python ARPA
try:
    from arpa_lm_python import ArpaLM
    PYTHON_LM_AVAILABLE = True
except ImportError:
    PYTHON_LM_AVAILABLE = False


# ============================================================================
# RULE-BASED NORMALIZATION (from postprocessing-for-normalised.py)
# ============================================================================

def load_abbreviation_dictionary(dict_path):
    """Load abbreviation dictionary from tab-separated file."""
    abbrev_dict = {}
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
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
                
                # Ensure both end with tsheg (།)
                if abbrev and not abbrev.endswith('་'):
                    abbrev += '་'
                if expansion and not expansion.endswith('་'):
                    expansion += '་'
                
                abbrev_dict[abbrev] = expansion
    
    # Sort by length (longest first)
    sorted_dict = dict(sorted(abbrev_dict.items(), key=lambda x: len(x[0]), reverse=True))
    return sorted_dict


def expand_abbreviations(text, abbrev_dict):
    """Expand abbreviations using exact matches."""
    for abbrev, expansion in abbrev_dict.items():
        text = text.replace(abbrev, expansion)
    return text


def fix_punctuation_spacing(text):
    """Fix punctuation and spacing issues."""
    # Change ༑ and ༎ to །
    text = text.replace('༑', '།')
    text = text.replace('༎', '།')
    
    # Add space after । if not already present (except double shad ༎༎)
    text = re.sub(r'།(?![།\s])', '། ', text)
    
    # Remove double tsheg
    while '་་' in text:
        text = text.replace('་་', '་')
    
    return text


def rule_based_normalize(text, abbrev_dict):
    """Apply rule-based normalization."""
    # Step 1: Expand abbreviations
    text = expand_abbreviations(text, abbrev_dict)
    
    # Step 2: Fix punctuation and spacing
    text = fix_punctuation_spacing(text)
    
    return text


def generate_inference_report(report_path, args, normalizer, num_lines, elapsed_time, 
                              start_time_str):
    """Generate inference report with run information."""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TIBETAN TEXT NORMALIZATION - INFERENCE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Run Information
        f.write("RUN INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Date/Time:        {start_time_str}\n")
        f.write(f"Mode:             {args.mode}\n")
        f.write(f"Device:           {normalizer.device}\n")
        f.write(f"Random Seed:      {RANDOM_SEED}\n")
        f.write("\n")
        
        # Input/Output
        f.write("INPUT/OUTPUT\n")
        f.write("-" * 70 + "\n")
        if hasattr(args, 'input_file') and args.input_file:
            f.write(f"Input file:       {args.input_file}\n")
            f.write(f"Output file:      {args.output_file}\n")
            f.write(f"Number of lines:  {num_lines}\n")
        elif hasattr(args, 'text') and args.text:
            f.write(f"Input:            Single text\n")
            f.write(f"Text:             {args.text[:50]}{'...' if len(args.text) > 50 else ''}\n")
        f.write("\n")
        
        # Model Information (if neural modes)
        if args.mode in ['neural', 'neural+lm', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules']:
            f.write("MODEL INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Model path:       {args.model_path}\n")
            
            # Try to get training info from checkpoint
            try:
                checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
                if 'args' in checkpoint:
                    train_args = checkpoint['args']
                    f.write(f"Training source:  {train_args.train_src}\n")
                    f.write(f"Training target:  {train_args.train_tgt}\n")
                    f.write(f"Training epochs:  {train_args.epochs}\n")
                    f.write(f"Batch size:       {train_args.batch_size}\n")
            except:
                pass
            
            f.write(f"Architecture:     d_model={normalizer.model.d_model if hasattr(normalizer, 'model') else 'N/A'}\n")
            if hasattr(normalizer, 'src_vocab'):
                f.write(f"Vocabulary:       src={len(normalizer.src_vocab)}, tgt={len(normalizer.tgt_vocab)}\n")
            f.write("\n")
        
        # Language Model Information (if LM modes)
        if args.mode in ['neural+lm', 'rules+neural+lm', 'neural+lm+rules'] and args.kenlm_path:
            f.write("LANGUAGE MODEL\n")
            f.write("-" * 70 + "\n")
            f.write(f"KenLM path:       {args.kenlm_path}\n")
            f.write(f"Backend:          {args.lm_backend}\n")
            if normalizer.lm is not None:
                f.write(f"Order:            {normalizer.lm.order}-gram\n")
            f.write("\n")
        
        # Rules Information (if rules modes)
        if args.mode in ['rules', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules'] and args.rules_dict:
            f.write("RULES INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Dictionary path:  {args.rules_dict}\n")
            if normalizer.rules_dict is not None:
                f.write(f"Abbreviations:    {len(normalizer.rules_dict)}\n")
            f.write("\n")
        
        # Parameters
        f.write("PARAMETERS\n")
        f.write("-" * 70 + "\n")
        if args.mode in ['neural', 'neural+lm', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules']:
            f.write(f"Beam width:       {args.beam_width}\n")
            f.write(f"Length penalty:   {args.length_penalty}\n")
        if args.mode in ['neural+lm', 'rules+neural+lm', 'neural+lm+rules']:
            f.write(f"LM weight:        {args.lm_weight}\n")
        f.write("\n")
        
        # Performance
        f.write("PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total time:       {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)\n")
        if num_lines > 0:
            f.write(f"Speed:            {num_lines/elapsed_time:.2f} lines/second\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")


# ============================================================================
# NEURAL MODEL CLASSES (same as before)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, 
                 dropout=0.1, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# ============================================================================
# UNIFIED NORMALIZER CLASS
# ============================================================================

class TibetanNormalizer:
    """Unified normalizer supporting neural, rule-based, or combined approaches."""
    
    def __init__(self, mode='neural', model_path=None, kenlm_path=None, 
                 rules_dict_path=None, lm_backend='auto', device=None):
        """
        Initialize normalizer
        
        Args:
            mode: 'neural', 'neural+lm', 'rules', 'rules+neural'
            model_path: Path to neural model
            kenlm_path: Path to language model
            rules_dict_path: Path to abbreviation dictionary
            lm_backend: 'auto', 'kenlm', 'python', 'none'
            device: torch device
        """
        self.mode = mode
        self.rules_dict = None
        self.model = None
        self.lm = None
        self.lm_type = None
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Mode: {mode}")
        print(f"Device: {self.device}")
        
        # Load rules if needed
        if mode in ['rules', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules'] and rules_dict_path:
            print(f"Loading abbreviation dictionary from {rules_dict_path}...")
            self.rules_dict = load_abbreviation_dictionary(rules_dict_path)
            print(f"✓ Loaded {len(self.rules_dict)} abbreviations")
        
        # Load neural model if needed
        if mode in ['neural', 'neural+lm', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules'] and model_path:
            self._load_neural_model(model_path)
        
        # Load LM if needed
        if mode in ['neural+lm', 'rules+neural+lm', 'neural+lm+rules'] and kenlm_path:
            self._load_language_model(kenlm_path, lm_backend)
        
        print("✓ Initialization complete\n")
    
    def _load_neural_model(self, model_path):
        """Load neural seq2seq model."""
        print(f"Loading neural model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.src_vocab = checkpoint['src_vocab']
        self.tgt_vocab = checkpoint['tgt_vocab']
        model_config = checkpoint.get('model_config', {})
        state_dict = checkpoint['model_state_dict']
        
        # Infer architecture from weights
        if 'src_embedding.weight' in state_dict:
            d_model = state_dict['src_embedding.weight'].shape[1]
        else:
            d_model = model_config.get('d_model', 512)
        
        num_encoder_layers = 0
        for key in state_dict.keys():
            if key.startswith('transformer.encoder.layers.'):
                layer_num = int(key.split('.')[3])
                num_encoder_layers = max(num_encoder_layers, layer_num + 1)
        
        num_layers = num_encoder_layers if num_encoder_layers > 0 else model_config.get('num_layers', 4)
        nhead = model_config.get('nhead', 8)
        dropout = model_config.get('dropout', 0.1)
        
        print(f"Model architecture: d_model={d_model}, layers={num_layers}, heads={nhead}")
        
        self.model = TransformerModel(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.idx_to_char = {idx: char for char, idx in self.tgt_vocab.items()}
        self.pad_idx = self.tgt_vocab.get('<pad>', 0)
        self.bos_idx = self.tgt_vocab.get('<bos>', 1)
        self.eos_idx = self.tgt_vocab.get('<eos>', 2)
        
        print(f"Vocabulary: src={len(self.src_vocab)}, tgt={len(self.tgt_vocab)}")
    
    def _load_language_model(self, lm_path, backend):
        """Load language model."""
        if backend == 'none':
            return
        
        if backend == 'auto':
            backend = 'kenlm' if KENLM_AVAILABLE else ('python' if PYTHON_LM_AVAILABLE else None)
        
        if backend == 'kenlm' and KENLM_AVAILABLE:
            print(f"Loading KenLM from {lm_path}...")
            self.lm = kenlm.Model(lm_path)
            self.lm_type = 'kenlm'
            print(f"✓ KenLM loaded (order: {self.lm.order})")
        elif backend == 'python' and PYTHON_LM_AVAILABLE:
            print(f"Loading Python ARPA from {lm_path}...")
            self.lm = ArpaLM(lm_path)
            self.lm_type = 'python'
            print(f"✓ Python ARPA loaded (order: {self.lm.order})")
        else:
            print("Warning: No LM backend available")
    
    def encode_text(self, text):
        """Encode text to indices."""
        indices = [self.src_vocab.get(char, self.src_vocab.get('<unk>', 3)) for char in text]
        return torch.tensor([indices], dtype=torch.long).to(self.device)
    
    def decode_indices(self, indices):
        """Decode indices to text."""
        chars = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.bos_idx]:
                chars.append(self.idx_to_char.get(idx, ''))
        return ''.join(chars)
    
    def get_lm_score(self, text):
        """Get LM score."""
        if self.lm is None:
            return 0.0
        try:
            return self.lm.score(text, bos=True, eos=True)
        except:
            return 0.0
    
    def beam_search(self, src, beam_width=5, max_len=200, lm_weight=0.2, length_penalty=0.6):
        """Beam search with optional LM."""
        self.model.eval()
        
        with torch.no_grad():
            src_mask = None
            src_key_padding_mask = (src == self.pad_idx)
            
            src_emb = self.model.dropout(
                self.model.pos_encoder(
                    self.model.src_embedding(src) * math.sqrt(self.model.d_model)
                )
            )
            memory = self.model.transformer.encoder(src_emb, mask=src_mask, 
                                                   src_key_padding_mask=src_key_padding_mask)
            
            beams = [([self.bos_idx], 0.0, False)]
            
            for step in range(max_len):
                candidates = []
                
                for seq, score, finished in beams:
                    if finished:
                        candidates.append((seq, score, True))
                        continue
                    
                    tgt = torch.tensor([seq], dtype=torch.long).to(self.device)
                    tgt_mask = self.model.generate_square_subsequent_mask(len(seq)).to(self.device)
                    
                    tgt_emb = self.model.dropout(
                        self.model.pos_encoder(
                            self.model.tgt_embedding(tgt) * math.sqrt(self.model.d_model)
                        )
                    )
                    
                    output = self.model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    logits = self.model.fc_out(output[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Block <unk>
                    unk_idx = self.tgt_vocab.get('<unk>', 3)
                    log_probs[0, unk_idx] = float('-inf')
                    
                    topk_log_probs, topk_indices = torch.topk(log_probs[0], beam_width)
                    
                    for log_prob, idx in zip(topk_log_probs, topk_indices):
                        idx = idx.item()
                        new_seq = seq + [idx]
                        new_score = score + log_prob.item()
                        new_finished = (idx == self.eos_idx)
                        candidates.append((new_seq, new_score, new_finished))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]
                
                if all(finished for _, _, finished in beams):
                    break
            
            # Rerank with LM if available
            if self.lm is not None:
                scored_beams = []
                for seq, seq2seq_score, _ in beams:
                    text = self.decode_indices(seq)
                    lm_score = self.get_lm_score(text)
                    length = len([idx for idx in seq if idx not in [self.pad_idx, self.bos_idx, self.eos_idx]])
                    length_norm = length ** length_penalty if length > 0 else 1.0
                    combined_score = (seq2seq_score / length_norm) + (lm_weight * lm_score)
                    scored_beams.append((text, combined_score))
                scored_beams.sort(key=lambda x: x[1], reverse=True)
                return scored_beams[0][0]
            else:
                best_seq = beams[0][0]
                return self.decode_indices(best_seq)
    
    def normalize(self, text, beam_width=5, lm_weight=0.2, length_penalty=0.6):
        """Normalize text using specified mode."""
        if not text.strip():
            return ""
        
        # Mode: rules only
        if self.mode == 'rules':
            if self.rules_dict is None:
                raise ValueError("Rules mode requires rules_dict_path")
            return rule_based_normalize(text, self.rules_dict)
        
        # Mode: neural only
        elif self.mode == 'neural':
            src = self.encode_text(text)
            return self.beam_search(src, beam_width=beam_width, lm_weight=0, length_penalty=length_penalty)
        
        # Mode: neural + lm
        elif self.mode == 'neural+lm':
            src = self.encode_text(text)
            return self.beam_search(src, beam_width=beam_width, lm_weight=lm_weight, length_penalty=length_penalty)
        
        # Mode: rules then neural
        elif self.mode == 'rules+neural':
            # Apply rules first
            preprocessed = rule_based_normalize(text, self.rules_dict)
            # Then neural
            src = self.encode_text(preprocessed)
            return self.beam_search(src, beam_width=beam_width, lm_weight=0, length_penalty=length_penalty)
        
        # Mode: rules then neural + lm
        elif self.mode == 'rules+neural+lm':
            # Apply rules first
            preprocessed = rule_based_normalize(text, self.rules_dict)
            # Then neural + LM
            src = self.encode_text(preprocessed)
            return self.beam_search(src, beam_width=beam_width, lm_weight=lm_weight, length_penalty=length_penalty)
        
        # Mode: neural + lm, then rules (postprocessing)
        elif self.mode == 'neural+lm+rules':
            # Neural + LM first
            src = self.encode_text(text)
            neural_output = self.beam_search(src, beam_width=beam_width, lm_weight=lm_weight, length_penalty=length_penalty)
            # Then apply rules as postprocessing
            return rule_based_normalize(neural_output, self.rules_dict)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def normalize_batch(self, texts, beam_width=5, lm_weight=0.2, length_penalty=0.6, show_progress=True):
        """Normalize batch of texts."""
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i % 100 == 0 or i == total - 1):
                print(f"Processing: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
            
            result = self.normalize(text, beam_width=beam_width, lm_weight=lm_weight, length_penalty=length_penalty)
            results.append(result)
        
        if show_progress:
            print()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Unified Tibetan text normalization')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['neural', 'neural+lm', 'rules', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules'],
                       help='Normalization mode')
    
    # Model paths
    parser.add_argument('--model_path', type=str, help='Neural model path (for neural modes)')
    parser.add_argument('--kenlm_path', type=str, help='KenLM path (for neural+lm mode)')
    parser.add_argument('--rules_dict', type=str, help='Abbreviation dictionary (for rules modes)')
    
    # LM backend
    parser.add_argument('--lm_backend', type=str, default='auto',
                       choices=['auto', 'kenlm', 'python', 'none'])
    
    # Input/output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str)
    input_group.add_argument('--input_file', type=str)
    input_group.add_argument('--interactive', action='store_true')
    
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file (default: auto-generated as inputfile-MODE_NUMBER-MODE_NAME.txt)')
    parser.add_argument('--report_file', type=str, default=None,
                       help='Inference report file (default: based on output_file with _report.txt)')
    
    # Neural parameters
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--lm_weight', type=float, default=0.2)
    parser.add_argument('--length_penalty', type=float, default=0.6)
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['neural', 'neural+lm', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules'] and not args.model_path:
        parser.error(f"--model_path required for mode '{args.mode}'")
    
    if args.mode in ['neural+lm', 'rules+neural+lm', 'neural+lm+rules'] and not args.kenlm_path:
        parser.error(f"--kenlm_path required for mode '{args.mode}'")
    
    if args.mode in ['rules', 'rules+neural', 'rules+neural+lm', 'neural+lm+rules'] and not args.rules_dict:
        parser.error(f"--rules_dict required for mode '{args.mode}'")
    
    # Auto-generate output filename if not provided (based on mode)
    if args.input_file and not args.output_file:
        import os
        base = os.path.splitext(args.input_file)[0]
        
        # Map modes to numbers
        mode_map = {
            'rules': '1',
            'neural': '2',
            'neural+lm': '3',
            'neural+lm+rules': '4',
            'rules+neural+lm': '5',
            'rules+neural': '6'
        }
        
        mode_num = mode_map.get(args.mode, 'X')
        
        # Replace last occurrence of 'source' with 'predictions-MODE'
        if base.endswith('_source') or base.endswith('-source'):
            # Remove the '_source' or '-source' suffix
            base_without_source = base.rsplit('_source', 1)[0] if '_source' in base else base.rsplit('-source', 1)[0]
            args.output_file = f"{base_without_source}_predictions-{mode_num}-{args.mode}.txt"
        else:
            # If no 'source' suffix, just append to the base
            args.output_file = f"{base}_predictions-{mode_num}-{args.mode}.txt"
        
        print(f"Output file not specified, using: {args.output_file}")
    
    # Auto-generate report filename based on output file
    if args.input_file and not args.report_file:
        import os
        base = os.path.splitext(args.output_file)[0]
        args.report_file = f"{base}_report.txt"
        print(f"Report file not specified, using: {args.report_file}")
    
    # Record start time
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize normalizer
    normalizer = TibetanNormalizer(
        mode=args.mode,
        model_path=args.model_path,
        kenlm_path=args.kenlm_path,
        rules_dict_path=args.rules_dict,
        lm_backend=args.lm_backend
    )
    
    # Single text
    if args.text:
        print("Input:", args.text)
        start = time.time()
        result = normalizer.normalize(args.text, beam_width=args.beam_width,
                                      lm_weight=args.lm_weight, length_penalty=args.length_penalty)
        elapsed = time.time() - start
        print("Output:", result)
        print(f"Time: {elapsed:.3f}s")
    
    # File processing
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(texts)} lines...")
        start = time.time()
        results = normalizer.normalize_batch(texts, beam_width=args.beam_width,
                                            lm_weight=args.lm_weight, length_penalty=args.length_penalty)
        elapsed = time.time() - start
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')
        
        print(f"✓ Done: {len(texts)} lines in {elapsed:.2f}s ({len(texts)/elapsed:.1f} lines/s)")
        print(f"✓ Predictions saved to: {args.output_file}")
        
        # Generate report
        if args.report_file:
            generate_inference_report(
                args.report_file, args, normalizer, len(texts), elapsed, start_time_str
            )
            print(f"✓ Report saved to: {args.report_file}")
    
    # Interactive
    elif args.interactive:
        print("Interactive mode (Ctrl+C or 'quit' to exit)")
        print(f"Mode: {args.mode}")
        print("-" * 60)
        
        try:
            while True:
                text = input("\nInput: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                
                start = time.time()
                result = normalizer.normalize(text, beam_width=args.beam_width,
                                             lm_weight=args.lm_weight, length_penalty=args.length_penalty)
                elapsed = time.time() - start
                print(f"Output: {result} ({elapsed:.3f}s)")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
    
    print("\nDone!")


if __name__ == '__main__':
    main()