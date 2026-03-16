"""
Code to train a character-based encoder-decoder transformer based on source (i.e. diplomatic)
and target (i.e. normalised) txt files that have corresponding sentence pairs.
As such it's a traditional seq2seq model (unlike gemma or other causal language models), 
which takes the sentence pairs to train and as such should be able to do well in normalisation tasks.

Original Claude prompt was to create script with eval and train/val/test splits with
all specs from the SG paper.

Run in conda pagantibenv:
python3 tibtrainencdecoder_witheval.py --train_src train_source.txt --train_tgt train_target.txt

Vocabulary (i.e. characters in a character-based transformer like this)
can be added in the build_vocab function and called as follows:
python3 tibtrainencdecoder_witheval.py --train_src train.txt --train_tgt target.txt --use_normalized_vocab

Outputs are:
- reports called "tibetan_report_Claude.txt" and "training_results_Claude.json"
- character-based transformer model called "tibetan_model_Claude.pt"

Evaluation metrics are usual CER, precision and recall but also the SG-specific metrics:

- Correcting Recall (CR) = Ccorr / Etotal
Measures the proportion of actual errors that were correctly fixed
Shows how many of the true errors in the source text were successfully corrected

- Correcting Precision (CP) = Ccorr / Eident
Measures the proportion of identified errors that were correctly fixed
Shows how accurate the model is when it decides to make a correction

Ccorr (Correctly Corrected): Positions where source ≠ target AND hypothesis = target
Etotal (Total Errors): Positions where source ≠ target
Eident (Identified Errors): Positions where source ≠ hypothesis

GPU optimisation applied for 750k paired training sents with the following recommended settings:

✅ Sampled Vocabulary Building - Only samples 10k texts to build vocab (much faster)
✅ Gradient Accumulation - Train with larger effective batch sizes without OOM
✅ Periodic Checkpoints - Saves every N epochs (not just best model)
✅ Checkpoint Directory - Organized checkpoint storage

Ultrafast settings for 4060:
python3 tibtrainencdecoder_witheval.py \
  --train_src augmented_1m_src.txt \
  --train_tgt augmented_1m_tgt.txt \
  --d_model 256 \
  --num_layers 4 \
  --nhead 8 \
  --batch_size 160 \
  --gradient_accumulation_steps 2 \
  --lr 0.001 \
  --dropout 0.1 \
  --weight_decay 0.0001 \
  --early_stopping 3 \
  --epochs 12 \
  --save_every 4 \
  --test_split 0.005 \
  --val_split 0.01 \
  --use_normalized_vocab \
  --checkpoint_dir checkpoints

Ultrafast params for rtx6000:
python3 tibtrainencdecoder_witheval.py \
  --train_src augmented_1m_src.txt \
  --train_tgt augmented_1m_tgt.txt \
  --d_model 512 \
  --num_layers 6 \
  --nhead 8 \
  --batch_size 512 \
  --gradient_accumulation_steps 1 \
  --lr 0.001 \
  --dropout 0.1 \
  --weight_decay 0.0001 \
  --early_stopping 3 \
  --epochs 12 \
  --save_every 4 \
  --test_split 0.005 \
  --val_split 0.01 \
  --use_normalized_vocab \
  --checkpoint_dir checkpoints

Expected training time: ~3-4 hours on RTX 4090 (i.e. the Cambridge LTL GPU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from collections import Counter
import argparse
import time
from datetime import timedelta
import json
import platform
import sys
import random

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
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.2):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(2, target.unsqueeze(2), self.confidence)
            true_dist[:, :, self.padding_idx] = 0
            mask = (target == self.padding_idx)
            true_dist.masked_fill_(mask.unsqueeze(2), 0)
        
        return self.criterion(pred, true_dist)

class TibetanDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=100):
        # Ensure src and tgt have the same length
        if len(src_texts) != len(tgt_texts):
            raise ValueError(f"Source and target must have same length. Got src={len(src_texts)}, tgt={len(tgt_texts)}")
        
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src = self.encode(self.src_texts[idx], self.src_vocab)
        tgt = self.encode(self.tgt_texts[idx], self.tgt_vocab)
        return torch.tensor(src), torch.tensor(tgt)
    
    def encode(self, text, vocab):
        tokens = ['<sos>'] + list(text[:self.max_len-2]) + ['<eos>']
        return [vocab.get(c, vocab['<unk>']) for c in tokens]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

def build_vocab(texts, use_normalized_vocab=False, sample_size=10000):
    """Build vocabulary with complete Tibetan Unicode range"""
    
    chars = set()
    
    # CRITICAL: Space character
    chars.add(' ')
    
    # Include ENTIRE Tibetan Unicode block (U+0F00-U+0FFF)
    # This ensures NO <unk> tokens in output
    print("Building vocabulary with complete Tibetan Unicode range...")
    for code in range(0x0F00, 0x1000):  # U+0F00 to U+0FFF
        chars.add(chr(code))
    
    # Also include common punctuation and symbols that might appear
    chars.update('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    chars.update('0123456789')
    
    # For source vocabulary, also scan actual data to catch any additional characters
    # (in case source has non-Tibetan characters like Latin, numbers, etc.)
    if not use_normalized_vocab:
        sample_texts = texts if len(texts) <= sample_size else random.sample(texts, sample_size)
        data_chars = set()
        for text in sample_texts:
            data_chars.update(text)
        
        additional = data_chars - chars
        if additional:
            print(f"Found {len(additional)} additional non-Tibetan characters in source: {sorted(additional)[:30]}")
            chars.update(additional)
    
    # Build vocabulary
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # Add all characters in sorted order
    for i, char in enumerate(sorted(chars), start=4):
        vocab[char] = i
    
    # Verification
    if ' ' not in vocab:
        raise RuntimeError("CRITICAL: Space character missing from vocabulary!")
    
    print(f"Vocabulary built with {len(vocab)} total tokens (including special tokens)")
    print(f"  - Tibetan Unicode range: U+0F00-U+0FFF")
    print(f"  - Space character: ✓ (ID: {vocab[' ']})")
    
    return vocab

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    # Levenshtein distance
    d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(reference)][len(hypothesis)] / len(reference)

def calculate_metrics(references, hypotheses, sources=None):
    """Calculate precision, recall, CER, Correcting Recall, and Correcting Precision"""
    total_cer = 0.0
    total_ref_chars = 0
    total_hyp_chars = 0
    total_correct_chars = 0
    
    # For Correcting Recall and Correcting Precision
    total_errors = 0  # Etotal: total actual errors (differences between source and target)
    total_identified_errors = 0  # Eident: errors identified by model (changes made)
    total_correctly_corrected = 0  # Ccorr: errors correctly corrected
    
    for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
        # CER
        total_cer += calculate_cer(ref, hyp)
        
        # Character-level precision and recall
        matches = 0
        for i, char in enumerate(ref):
            if i < len(hyp) and hyp[i] == char:
                matches += 1
        
        total_correct_chars += matches
        total_ref_chars += len(ref)
        total_hyp_chars += len(hyp)
        
        # Correcting Recall and Correcting Precision
        if sources is not None and idx < len(sources):
            src = sources[idx]
            
            # Count actual errors (Etotal): positions where source differs from target
            for i in range(max(len(src), len(ref))):
                src_char = src[i] if i < len(src) else ''
                ref_char = ref[i] if i < len(ref) else ''
                if src_char != ref_char:
                    total_errors += 1
            
            # Count identified errors (Eident): positions where model changed from source
            for i in range(max(len(src), len(hyp))):
                src_char = src[i] if i < len(src) else ''
                hyp_char = hyp[i] if i < len(hyp) else ''
                if src_char != hyp_char:
                    total_identified_errors += 1
            
            # Count correctly corrected errors (Ccorr): positions where:
            # 1. Source differs from target (there was an error)
            # 2. Hypothesis matches target (model corrected it)
            max_len = max(len(src), len(ref), len(hyp))
            for i in range(max_len):
                src_char = src[i] if i < len(src) else ''
                ref_char = ref[i] if i < len(ref) else ''
                hyp_char = hyp[i] if i < len(hyp) else ''
                
                # Error existed and was corrected
                if src_char != ref_char and hyp_char == ref_char:
                    total_correctly_corrected += 1
    
    avg_cer = total_cer / len(references) if references else 0.0
    precision = total_correct_chars / total_hyp_chars if total_hyp_chars > 0 else 0.0
    recall = total_correct_chars / total_ref_chars if total_ref_chars > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Correcting metrics
    correcting_recall = total_correctly_corrected / total_errors if total_errors > 0 else 0.0
    correcting_precision = total_correctly_corrected / total_identified_errors if total_identified_errors > 0 else 0.0
    
    return {
        'cer': avg_cer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correcting_recall': correcting_recall,
        'correcting_precision': correcting_precision,
        'total_errors': total_errors,
        'correctly_corrected': total_correctly_corrected,
        'identified_errors': total_identified_errors
    }

def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=100, device='cpu'):
    """Fast greedy decoding for evaluation - much faster than beam search"""
    model.eval()
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    
    with torch.no_grad():
        src = src.to(device)
        
        # Encode source
        src_emb = model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb)
        
        # Start with <sos> token
        ys = torch.ones(1, 1).fill_(tgt_vocab['<sos>']).type(torch.long).to(device)
        
        for i in range(max_len):
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
            tgt_emb = model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model))
            
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.fc_out(out[:, -1, :])
            
            # Greedy selection
            next_token = logits.argmax(dim=-1).item()
            
            if next_token == tgt_vocab['<eos>']:
                break
            
            ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_token).to(device)], dim=1)
        
        decoded_tokens = ys[0].cpu().numpy()
        decoded = ''.join([inv_tgt_vocab.get(idx, '') for idx in decoded_tokens 
                          if idx not in [tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>'], tgt_vocab['<unk>']]])
        
        return decoded

def beam_search(model, src, src_vocab, tgt_vocab, beam_width=5, max_len=100, device='cpu'):
    model.eval()
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    
    with torch.no_grad():
        src = src.to(device)
        src_mask = None
        
        # Encode source
        src_emb = model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb, src_mask)
        
        # Initialize beams
        beams = [(torch.tensor([[tgt_vocab['<sos>']]]).to(device), 0.0)]
        
        for _ in range(max_len):
            new_beams = []
            
            for seq, score in beams:
                if seq[0, -1].item() == tgt_vocab['<eos>']:
                    new_beams.append((seq, score))
                    continue
                
                tgt_mask = model.generate_square_subsequent_mask(seq.size(1)).to(device)
                tgt_emb = model.pos_encoder(model.tgt_embedding(seq) * math.sqrt(model.d_model))
                
                output = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)
                
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, token], dim=1)
                    new_score = score + topk_probs[0, i].item()
                    new_beams.append((new_seq, new_score))
            
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if all(seq[0, -1].item() == tgt_vocab['<eos>'] for seq, _ in beams):
                break
        
        best_seq = beams[0][0][0].cpu().numpy()
        decoded = ''.join([inv_tgt_vocab.get(idx, '') for idx in best_seq 
                          if idx not in [tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>'], tgt_vocab['<unk>']]])
        
        return decoded

def evaluate_model(model, dataloader, src_vocab, tgt_vocab, device, show_progress=True, use_beam_search=False, beam_width=5):
    """Evaluate model and return predictions with progress indicator
    
    Args:
        use_beam_search: If False, uses fast greedy decoding. If True, uses slower beam search.
    """
    model.eval()
    references = []
    hypotheses = []
    sources = []
    
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    inv_src_vocab = {v: k for k, v in src_vocab.items()}
    
    total_samples = sum(src.size(0) for src, _ in dataloader)
    processed = 0
    start_time = time.time()
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            
            for i in range(src.size(0)):
                # Get source
                src_indices = src[i].cpu().numpy()
                src_text = ''.join([inv_src_vocab.get(idx, '') for idx in src_indices 
                                   if idx not in [src_vocab['<pad>'], src_vocab['<sos>'], src_vocab['<eos>'], src_vocab['<unk>']]])
                
                # Get reference
                ref_indices = tgt[i].cpu().numpy()
                ref = ''.join([inv_tgt_vocab.get(idx, '') for idx in ref_indices 
                              if idx not in [tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>'], tgt_vocab['<unk>']]])
                
                # Get hypothesis using greedy decode (fast) or beam search (slow but better quality)
                if use_beam_search:
                    hyp = beam_search(model, src[i:i+1], src_vocab, tgt_vocab, 
                                     beam_width=beam_width, max_len=100, device=device)
                else:
                    hyp = greedy_decode(model, src[i:i+1], src_vocab, tgt_vocab, 
                                       max_len=100, device=device)
                
                sources.append(src_text)
                references.append(ref)
                hypotheses.append(hyp)
                
                processed += 1
                
                if show_progress:
                    elapsed = time.time() - start_time
                    progress = processed / total_samples
                    eta = (elapsed / processed) * (total_samples - processed) if processed > 0 else 0
                    
                    # Progress bar
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    sys.stdout.write(f'\r  Progress: [{bar}] {processed}/{total_samples} | '
                                   f'Elapsed: {timedelta(seconds=int(elapsed))} | '
                                   f'ETA: {timedelta(seconds=int(eta))}')
                    sys.stdout.flush()
    
    if show_progress:
        print()  # New line after progress bar
    
    return sources, references, hypotheses

def train_epoch(model, dataloader, criterion, optimizer, device, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt_input == 0)
        
        output = model(src, tgt_input, tgt_mask=tgt_mask, 
                      src_key_padding_mask=src_padding_mask,
                      tgt_key_padding_mask=tgt_padding_mask)
        
        loss = criterion(output, tgt_output)
        loss = loss / gradient_accumulation_steps  # Scale loss
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
    
    # Final step if there are leftover gradients
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_padding_mask = (src == 0)
            tgt_padding_mask = (tgt_input == 0)
            
            output = model(src, tgt_input, tgt_mask=tgt_mask,
                          src_key_padding_mask=src_padding_mask,
                          tgt_key_padding_mask=tgt_padding_mask)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
            num_batches += 1
    
    if num_batches == 0:
        print("WARNING: No batches in validation dataloader!")
        return float('inf')
    
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Tibetan Text Normalization')
    parser.add_argument('--train_src', type=str, required=True, help='Training source file')
    parser.add_argument('--train_tgt', type=str, required=True, help='Training target file')
    parser.add_argument('--val_src', type=str, default=None, help='Validation source file')
    parser.add_argument('--val_tgt', type=str, default=None, help='Validation target file')
    parser.add_argument('--test_src', type=str, default=None, help='Test source file')
    parser.add_argument('--test_tgt', type=str, default=None, help='Test target file')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split ratio (default: 0.15)')
    parser.add_argument('--no_auto_split', action='store_true', help='Disable automatic train/val/test split')
    parser.add_argument('--use_normalized_vocab', action='store_true', help='Use normalized Tibetan vocab')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of encoder/decoder layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization, default: 0.0001)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam search width')
    parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping patience (0 = disabled)')
    parser.add_argument('--save_model', type=str, default='tibetan_model.pt', help='Model save path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for periodic checkpoints')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--results_file', type=str, default='training_results.json', help='Results output file')
    parser.add_argument('--report_file', type=str, default='tibetan_report_Claude.txt', help='Text report file')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get system information
    system_info = {
        'processor': platform.processor(),
        'machine': platform.machine(),
        'system': platform.system(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'device': str(device)
    }
    
    if torch.cuda.is_available():
        system_info['cuda_device'] = torch.cuda.get_device_name(0)
        system_info['cuda_version'] = torch.version.cuda
    
    print(f"Processor: {system_info['processor']}")
    print(f"System: {system_info['system']} {system_info['machine']}")
    
    # Load data
    print("Loading training data...")
    with open(args.train_src, 'r', encoding='utf-8') as f:
        train_src = [line.strip() for line in f]
    with open(args.train_tgt, 'r', encoding='utf-8') as f:
        train_tgt = [line.strip() for line in f]
    
    # Validate that source and target have the same number of lines
    if len(train_src) != len(train_tgt):
        print(f"ERROR: Mismatch in training data!")
        print(f"  Source file has {len(train_src)} lines")
        print(f"  Target file has {len(train_tgt)} lines")
        print("  Source and target files must have the same number of lines.")
        sys.exit(1)
    
    print(f"Loaded {len(train_src)} training examples")
    
    # Auto-split into train/val/test if no validation/test files provided
    val_src, val_tgt = None, None
    test_src, test_tgt = None, None
    
    if (args.val_src is None or args.val_tgt is None or 
        args.test_src is None or args.test_tgt is None):
        if not args.no_auto_split and len(train_src) > 2:
            # Calculate split indices (ensure at least 1 sample in each split)
            total_size = len(train_src)
            test_size = max(1, int(total_size * args.test_split))
            val_size = max(1, int(total_size * args.val_split))
            train_size = total_size - test_size - val_size
            
            if train_size < 1:
                print("Warning: Dataset too small for 3-way split, using 2-way split instead")
                val_size = max(1, int(total_size * 0.2))
                train_size = total_size - val_size
                test_size = 0
            
            # Split the data
            test_src = train_src[-test_size:] if test_size > 0 else []
            test_tgt = train_tgt[-test_size:] if test_size > 0 else []
            val_src = train_src[train_size:train_size+val_size]
            val_tgt = train_tgt[train_size:train_size+val_size]
            train_src = train_src[:train_size]
            train_tgt = train_tgt[:train_size]
            
            print(f"Auto-split: {len(train_src)} training, {len(val_src)} validation, {len(test_src)} test")
        else:
            print("No automatic split (use --val_split/--test_split or provide separate files)")
    else:
        print(f"Using provided validation/test files")
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = build_vocab(train_src, args.use_normalized_vocab)
    tgt_vocab = build_vocab(train_tgt, args.use_normalized_vocab)
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # CRITICAL DEBUG: Check if space is in vocabulary
    space_in_src = ' ' in src_vocab
    space_in_tgt = ' ' in tgt_vocab
    print(f"Space ' ' in source vocab: {space_in_src}")
    print(f"Space ' ' in target vocab: {space_in_tgt}")
    
    # If space is missing, this is a critical error
    if not space_in_src or not space_in_tgt:
        print("\nCRITICAL ERROR: Space character not in vocabulary!")
        print("This will cause <unk> tokens in your data.")
        print("\nChecking if spaces exist in your data...")
        
        # Check first few lines for spaces
        has_space_in_src = any(' ' in line for line in train_src[:100])
        has_space_in_tgt = any(' ' in line for line in train_tgt[:100])
        
        print(f"  Spaces found in source data (first 100 lines): {has_space_in_src}")
        print(f"  Spaces found in target data (first 100 lines): {has_space_in_tgt}")
        
        if has_space_in_src or has_space_in_tgt:
            print("\n  >>> Your data HAS spaces but vocabulary doesn't include them!")
            print("  >>> Rebuilding vocabulary with space included...")
            
            # Force add space to vocab
            src_vocab[' '] = len(src_vocab)
            tgt_vocab[' '] = len(tgt_vocab)
            
            print(f"  >>> Fixed! Space now in source vocab: {' ' in src_vocab}")
            print(f"  >>> Fixed! Space now in target vocab: {' ' in tgt_vocab}")
    
    # Check vocabulary overlap
    src_chars = set(src_vocab.keys()) - {'<pad>', '<sos>', '<eos>', '<unk>'}
    tgt_chars = set(tgt_vocab.keys()) - {'<pad>', '<sos>', '<eos>', '<unk>'}
    overlap = src_chars & tgt_chars
    print(f"Character overlap between source and target: {len(overlap)}/{len(src_chars)} ({len(overlap)/len(src_chars)*100:.1f}%)")
    
    # Create datasets
    print(f"\nCreating training dataset with {len(train_src)} source and {len(train_tgt)} target examples...")
    train_dataset = TibetanDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    test_loader = None
    
    # Load validation data from files if provided
    if args.val_src is not None and args.val_tgt is not None:
        print(f"Loading validation data from {args.val_src} and {args.val_tgt}...")
        try:
            with open(args.val_src, 'r', encoding='utf-8') as f:
                val_src = [line.strip() for line in f]
            with open(args.val_tgt, 'r', encoding='utf-8') as f:
                val_tgt = [line.strip() for line in f]
            print(f"Read {len(val_src)} validation examples from files")
        except Exception as e:
            print(f"⨯ Error loading validation data: {e}")
            val_src, val_tgt = None, None
    
    # Load test data from files if provided
    if args.test_src is not None and args.test_tgt is not None:
        print(f"Loading test data from {args.test_src} and {args.test_tgt}...")
        try:
            with open(args.test_src, 'r', encoding='utf-8') as f:
                test_src = [line.strip() for line in f]
            with open(args.test_tgt, 'r', encoding='utf-8') as f:
                test_tgt = [line.strip() for line in f]
            print(f"Read {len(test_src)} test examples from files")
        except Exception as e:
            print(f"⨯ Error loading test data: {e}")
            test_src, test_tgt = None, None
    
    # Create validation loader if we have validation data
    if val_src is not None and val_tgt is not None and len(val_src) > 0:
        val_dataset = TibetanDataset(val_src, val_tgt, src_vocab, tgt_vocab)
        val_batch_size = min(args.batch_size, len(val_dataset))
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, 
                               shuffle=False, collate_fn=collate_fn)
        print(f"✓ Validation loader created: {len(val_dataset)} samples (batch size: {val_batch_size})")
    
    # Create test loader if we have test data
    if test_src is not None and test_tgt is not None and len(test_src) > 0:
        test_dataset = TibetanDataset(test_src, test_tgt, src_vocab, tgt_vocab)
        test_batch_size = min(args.batch_size, len(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, 
                                shuffle=False, collate_fn=collate_fn)
        print(f"✓ Test loader created: {len(test_dataset)} samples (batch size: {test_batch_size})")
    
    print()
    
    # Initialize model
    model = TransformerModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    criterion = LabelSmoothingLoss(len(tgt_vocab), padding_idx=0, smoothing=0.1)  # Reduced from 0.2
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.997), eps=1e-9, weight_decay=args.weight_decay)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create checkpoint directory
    import os
    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    total_start_time = time.time()
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Total Epochs: {args.epochs}")
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Batches per Epoch: {len(train_loader)}")
    
    # Initial time estimate based on a calibration run
    print("\nRunning calibration batches to estimate training time...")
    calibration_start = time.time()
    model.train()
    
    # Run 3 batches for better calibration
    calibration_batches = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt_input == 0)
        
        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask=tgt_mask, 
                      src_key_padding_mask=src_padding_mask,
                      tgt_key_padding_mask=tgt_padding_mask)
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        
        calibration_batches += 1
        if calibration_batches >= 3:
            break
    
    calibration_time = (time.time() - calibration_start) / calibration_batches
    estimated_epoch_time = calibration_time * len(train_loader)
    
    # Add validation time estimate if applicable
    if val_loader is not None:
        estimated_epoch_time *= 1.1  # Add 10% for validation (it's fast, just loss calculation)
    
    total_estimated_time = estimated_epoch_time * args.epochs
    
    print(f"Estimated time per epoch: {timedelta(seconds=int(estimated_epoch_time))}")
    print(f"Estimated total training time: {timedelta(seconds=int(total_estimated_time))}")
    
    # Provide recommendations if training will take very long
    if total_estimated_time > 86400:  # More than 1 day
        days = total_estimated_time / 86400
        print(f"\n⚠️  Training will take approximately {days:.1f} days")
        print("💡 Recommendations to speed up training:")
        print("   1. Reduce --epochs (try 20-30 instead of 50)")
        print("   2. Increase --batch_size (try 128 or 256)")
        print("   3. Use --early_stopping 5 to stop if not improving")
        print("   4. Increase --gradient_accumulation_steps for larger effective batch size")
        if args.test_split > 0.01:
            print(f"   5. Reduce --test_split (currently {args.test_split}, try 0.005)")
        if args.val_split > 0.02:
            print(f"   6. Reduce --val_split (currently {args.val_split}, try 0.01)")
    
    print("=" * 60 + "\n")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.gradient_accumulation_steps)
        output_str = f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}"
        
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            output_str += f" | Val Loss: {val_loss:.4f}"
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'src_vocab': src_vocab,
                    'tgt_vocab': tgt_vocab,
                    'args': args,
                    'epoch': epoch
                }, args.save_model)
                output_str += " | Model saved"
            else:
                epochs_without_improvement += 1
            
            # Save periodic checkpoints
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'src_vocab': src_vocab,
                    'tgt_vocab': tgt_vocab,
                    'args': args,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, checkpoint_path)
                output_str += f" | Checkpoint saved"
            
            # Early stopping
            if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
                print(f"\n\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
            
            # Warning if loss isn't decreasing
            if epoch > 10 and val_loss > best_val_loss * 1.5:
                output_str += " | WARNING: Loss increasing significantly"
        
        print(output_str)
    
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Total Time: {timedelta(seconds=int(total_time))}")
    print("="*60)
    
    # Evaluate best model on test set
    if test_loader is not None:
        print("\nEvaluating best model on test set using beam search...")
        print("(This will take longer but provides better quality predictions)")
        eval_start_time = time.time()
        
        # Load best model
        checkpoint = torch.load(args.save_model, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get predictions and calculate metrics using beam search for best quality
        sources, references, hypotheses = evaluate_model(
            model, test_loader, src_vocab, tgt_vocab, device, 
            use_beam_search=True, beam_width=args.beam_width
        )
        metrics = calculate_metrics(references, hypotheses, sources)
        
        eval_time = time.time() - eval_start_time
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS (Best Model on Test Set)")
        print("="*60)
        print(f"Character Error Rate (CER): {metrics['cer']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall: {metrics['recall']*100:.2f}%")
        print(f"F1 Score: {metrics['f1']*100:.2f}%")
        print(f"Correcting Recall (CR): {metrics['correcting_recall']*100:.2f}%")
        print(f"Correcting Precision (CP): {metrics['correcting_precision']*100:.2f}%")
        print("="*60)
        print(f"Evaluation time: {timedelta(seconds=int(eval_time))}")
        print("="*60)
        
        # Save results to JSON file
        results = {
            'system_info': system_info,
            'training_info': {
                'num_training_pairs': len(train_dataset),
                'num_validation_pairs': len(val_loader.dataset) if val_loader else 0,
                'num_test_pairs': len(test_loader.dataset) if test_loader else 0,
                'total_epochs': args.epochs,
                'best_val_loss': best_val_loss,
                'total_training_time_seconds': int(total_time),
                'total_training_time_formatted': str(timedelta(seconds=int(total_time))),
                'evaluation_time_seconds': int(eval_time),
                'evaluation_time_formatted': str(timedelta(seconds=int(eval_time)))
            },
            'evaluation_metrics': {
                'character_error_rate': float(metrics['cer']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1']),
                'correcting_recall': float(metrics['correcting_recall']),
                'correcting_precision': float(metrics['correcting_precision']),
                'total_errors': metrics['total_errors'],
                'correctly_corrected': metrics['correctly_corrected'],
                'identified_errors': metrics['identified_errors']
            },
            'model_config': {
                'd_model': args.d_model,
                'nhead': args.nhead,
                'num_layers': args.num_layers,
                'batch_size': args.batch_size,
                'learning_rate': args.lr
            }
        }
        
        with open(args.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to {args.results_file}")
        
        # Save text report
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("TIBETAN TEXT NORMALIZATION - TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("SYSTEM INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Processor:        {system_info['processor']}\n")
            f.write(f"Machine:          {system_info['machine']}\n")
            f.write(f"Operating System: {system_info['system']}\n")
            f.write(f"Python Version:   {system_info['python_version']}\n")
            f.write(f"PyTorch Version:  {system_info['pytorch_version']}\n")
            f.write(f"Device:           {system_info['device']}\n")
            if 'cuda_device' in system_info:
                f.write(f"CUDA Device:      {system_info['cuda_device']}\n")
                f.write(f"CUDA Version:     {system_info['cuda_version']}\n")
            f.write("\n")
            
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Model Dimension (d_model):     {args.d_model}\n")
            f.write(f"Attention Heads:               {args.nhead}\n")
            f.write(f"Encoder/Decoder Layers:        {args.num_layers}\n")
            f.write(f"Batch Size:                    {args.batch_size}\n")
            f.write(f"Learning Rate:                 {args.lr}\n")
            f.write(f"Optimizer:                     Adam (β1=0.9, β2=0.997)\n")
            f.write(f"Label Smoothing:               0.2\n")
            f.write(f"Beam Search Width:             {args.beam_width}\n")
            f.write("\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Training Sentence Pairs:       {len(train_dataset)}\n")
            f.write(f"Validation Sentence Pairs:     {len(val_loader.dataset) if val_loader else 0}\n")
            f.write(f"Test Sentence Pairs:           {len(test_loader.dataset)}\n")
            f.write(f"Source Vocabulary Size:        {len(src_vocab)}\n")
            f.write(f"Target Vocabulary Size:        {len(tgt_vocab)}\n")
            f.write("\n")
            
            f.write("TRAINING RESULTS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Epochs:                  {args.epochs}\n")
            f.write(f"Best Validation Loss:          {best_val_loss:.4f}\n")
            f.write(f"Total Training Time:           {timedelta(seconds=int(total_time))}\n")
            f.write(f"Evaluation Time:               {timedelta(seconds=int(eval_time))}\n")
            f.write("\n")
            
            f.write("EVALUATION METRICS (Best Model on Test Set)\n")
            f.write("-"*70 + "\n")
            f.write(f"Character Error Rate (CER):    {metrics['cer']*100:.2f}%\n")
            f.write(f"Precision:                     {metrics['precision']*100:.2f}%\n")
            f.write(f"Recall:                        {metrics['recall']*100:.2f}%\n")
            f.write(f"F1 Score:                      {metrics['f1']*100:.2f}%\n")
            f.write(f"Correcting Recall (CR):        {metrics['correcting_recall']*100:.2f}%\n")
            f.write(f"Correcting Precision (CP):     {metrics['correcting_precision']*100:.2f}%\n")
            f.write("\n")
            f.write(f"Error Correction Statistics:\n")
            f.write(f"  Total Errors (Etotal):       {metrics['total_errors']}\n")
            f.write(f"  Correctly Corrected (Ccorr): {metrics['correctly_corrected']}\n")
            f.write(f"  Identified Errors (Eident):  {metrics['identified_errors']}\n")
            f.write("\n")
            
            f.write("EXAMPLE PREDICTIONS (10 Random Examples from Test Set)\n")
            f.write("-"*70 + "\n")
            num_examples = min(10, len(sources))
            
            # Select random indices
            if num_examples > 0:
                indices = random.sample(range(len(sources)), num_examples)
                for idx, i in enumerate(indices, 1):
                    f.write(f"\nExample {idx}:\n")
                    f.write(f"  Source:     {sources[i]}\n")
                    f.write(f"  Target:     {references[i]}\n")
                    f.write(f"  Predicted:  {hypotheses[i]}\n")
            else:
                f.write("  No examples available.\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Text report saved to {args.report_file}")
    else:
        print("\nNo test set - skipping evaluation")
    
    print()
    
    # Example inference with beam search
    if test_loader is not None and len(test_loader.dataset) > 0:
        print("\n=== Example Inference ===")
        # Get first test example
        for src_batch, tgt_batch in test_loader:
            test_src_text = test_src[0]
            test_tgt_text = test_tgt[0]
            encoded = src_batch[0:1].to(device)
            result = beam_search(model, encoded, src_vocab, tgt_vocab, 
                                beam_width=args.beam_width, device=device)
            print(f"Source: {test_src_text}")
            print(f"Target: {test_tgt_text}")
            print(f"Predicted: {result}")
            break
    elif len(train_src) > 0:
        print("\n=== Example Inference ===")
        test_text = train_src[0]
        train_dataset_temp = TibetanDataset([train_src[0]], [train_tgt[0]], src_vocab, tgt_vocab)
        encoded = torch.tensor([train_dataset_temp.encode(test_text, src_vocab)]).to(device)
        result = beam_search(model, encoded, src_vocab, tgt_vocab, 
                            beam_width=args.beam_width, device=device)
        print(f"Input: {test_text}")
        print(f"Output: {result}")

if __name__ == '__main__':
    main()