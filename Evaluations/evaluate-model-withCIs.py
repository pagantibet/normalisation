"""
Standalone evaluation script for trained models and inference outputs
Supports multiple inference modes:
- seq2seq neural model only
- seq2seq with KenLM ranking
- rule-based preprocessing/postprocessing
- any combination or purely rule-based approaches
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import time
from datetime import timedelta
import json
import random
import sys
import math
import os
import numpy as np

# Import all the classes from the training script
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

class TibetanDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=100):
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

def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=100, device='cpu'):
    """Fast greedy decoding"""
    model.eval()
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    
    with torch.no_grad():
        src = src.to(device)
        src_emb = model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb)
        
        ys = torch.ones(1, 1).fill_(tgt_vocab['<sos>']).type(torch.long).to(device)
        
        for i in range(max_len):
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
            tgt_emb = model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model))
            
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.fc_out(out[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            
            if next_token == tgt_vocab['<eos>']:
                break
            
            ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_token).to(device)], dim=1)
        
        decoded_tokens = ys[0].cpu().numpy()
        decoded = ''.join([inv_tgt_vocab.get(idx, '') for idx in decoded_tokens 
                          if idx not in [tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>'], tgt_vocab['<unk>']]])
        
        return decoded

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
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
    """Calculate all metrics"""
    total_cer = 0.0
    total_ref_chars = 0
    total_hyp_chars = 0
    total_correct_chars = 0
    
    total_errors = 0
    total_identified_errors = 0
    total_correctly_corrected = 0
    
    for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
        total_cer += calculate_cer(ref, hyp)
        
        matches = 0
        for i, char in enumerate(ref):
            if i < len(hyp) and hyp[i] == char:
                matches += 1
        
        total_correct_chars += matches
        total_ref_chars += len(ref)
        total_hyp_chars += len(hyp)
        
        if sources is not None and idx < len(sources):
            src = sources[idx]
            
            for i in range(max(len(src), len(ref))):
                src_char = src[i] if i < len(src) else ''
                ref_char = ref[i] if i < len(ref) else ''
                if src_char != ref_char:
                    total_errors += 1
            
            for i in range(max(len(src), len(hyp))):
                src_char = src[i] if i < len(src) else ''
                hyp_char = hyp[i] if i < len(hyp) else ''
                if src_char != hyp_char:
                    total_identified_errors += 1
            
            for i in range(max(len(src), len(ref), len(hyp))):
                src_char = src[i] if i < len(src) else ''
                ref_char = ref[i] if i < len(ref) else ''
                hyp_char = hyp[i] if i < len(hyp) else ''
                
                if src_char != ref_char and hyp_char == ref_char:
                    total_correctly_corrected += 1
    
    avg_cer = total_cer / len(references) if len(references) > 0 else 0.0
    precision = total_correct_chars / total_hyp_chars if total_hyp_chars > 0 else 0.0
    recall = total_correct_chars / total_ref_chars if total_ref_chars > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
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

def bootstrap_ci(references, hypotheses, sources=None, n_iterations=1000, ci=95, seed=42):
    """
    Compute bootstrap confidence intervals for all metrics.
    Resamples (with replacement) from the available test lines n_iterations times
    and returns the lower/upper percentile bounds for each metric.
    """
    rng = np.random.default_rng(seed)
    n = len(references)
    alpha = (100 - ci) / 2  # e.g. 2.5 for a 95% CI

    metric_samples = {
        'cer': [], 'precision': [], 'recall': [], 'f1': [],
        'correcting_recall': [], 'correcting_precision': []
    }

    for _ in range(n_iterations):
        indices = rng.integers(0, n, size=n)  # sample with replacement
        refs_b  = [references[i] for i in indices]
        hyps_b  = [hypotheses[i] for i in indices]
        srcs_b  = [sources[i] for i in indices] if sources is not None else None

        m = calculate_metrics(refs_b, hyps_b, srcs_b)
        for key in metric_samples:
            metric_samples[key].append(m[key])

    cis = {}
    for key, values in metric_samples.items():
        lower = float(np.percentile(values, alpha))
        upper = float(np.percentile(values, 100 - alpha))
        cis[key] = {'lower': lower, 'upper': upper}

    return cis


def evaluate_model(model, test_loader, src_vocab, tgt_vocab, device, max_samples=None):
    """Evaluate model and return sources, references, hypotheses"""
    model.eval()
    all_sources = []
    all_references = []
    all_hypotheses = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_loader):
            src = src.to(device)
            batch_size = src.size(0)
            
            for i in range(batch_size):
                src_seq = src[i:i+1]
                src_text = decode_sequence(src_seq[0], src_vocab)
                tgt_text = decode_sequence(tgt[i], tgt_vocab)
                pred_text = greedy_decode(model, src_seq, src_vocab, tgt_vocab, device=device)
                
                all_sources.append(src_text)
                all_references.append(tgt_text)
                all_hypotheses.append(pred_text)
                
                if max_samples and len(all_sources) >= max_samples:
                    return all_sources, all_references, all_hypotheses
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {len(all_sources)} samples...")
    
    return all_sources, all_references, all_hypotheses

def decode_sequence(seq, vocab):
    """Decode a sequence back to text"""
    inv_vocab = {v: k for k, v in vocab.items()}
    special_tokens = {'<pad>', '<sos>', '<eos>', '<unk>'}
    return ''.join([inv_vocab.get(idx.item() if torch.is_tensor(idx) else idx, '') 
                   for idx in seq if inv_vocab.get(idx.item() if torch.is_tensor(idx) else idx, '') not in special_tokens])

def load_predictions_from_file(prediction_file):
    """Load predictions from a text file (one per line)"""
    with open(prediction_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def main():
    parser = argparse.ArgumentParser(description='Evaluate normalisation models or inference outputs')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='model', 
                       choices=['model', 'predictions'],
                       help='Evaluation mode: "model" (load and run model) or "predictions" (use pre-generated predictions)')
    
    # Model evaluation arguments (used when mode='model')
    parser.add_argument('--model', type=str, help='Path to saved model checkpoint (.pth)')
    
    # Prediction evaluation arguments (used when mode='predictions')
    parser.add_argument('--predictions', type=str, help='Path to predictions file (one prediction per line)')
    
    # Common arguments for both modes
    parser.add_argument('--test_src', type=str, required=True, help='Test source file')
    parser.add_argument('--test_tgt', type=str, required=True, help='Test target file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation (model mode only)')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to evaluate (None = all)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file (if not specified, auto-generated from predictions file)')
    parser.add_argument('--output_dir', type=str, default='evaluation-results', help='Output directory for results (default: evaluation-results)')
    
    # Inference method metadata (optional, for documentation purposes)
    parser.add_argument('--inference_method', type=str, default='unknown', 
                       help='Description of inference method (e.g., "seq2seq", "seq2seq+kenlm", "rules_only", etc.)')
    parser.add_argument('--uses_neural_model', action='store_true', 
                       help='Flag indicating if a neural model was used')
    parser.add_argument('--uses_kenlm', action='store_true',
                       help='Flag indicating if KenLM was used for ranking')
    parser.add_argument('--uses_preprocessing', action='store_true',
                       help='Flag indicating if rule-based preprocessing was used')
    parser.add_argument('--uses_postprocessing', action='store_true',
                       help='Flag indicating if rule-based postprocessing was used')
    parser.add_argument('--kenlm_path', type=str, default=None,
                       help='Path to KenLM model if used')
    parser.add_argument('--description', type=str, default='',
                       help='Free-text description of the inference approach')
    parser.add_argument('--bootstrap_n', type=int, default=1000,
                       help='Number of bootstrap iterations for confidence intervals (default: 1000, set 0 to disable)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'model' and not args.model:
        parser.error("--model is required when --mode=model")
    if args.mode == 'predictions' and not args.predictions:
        parser.error("--predictions is required when --mode=predictions")
    
    # Auto-generate output filename if not provided
    if args.output is None:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.mode == 'predictions' and args.predictions:
            # Extract base filename from predictions file
            # e.g., "predictions_neural.txt" -> "predictions_neural"
            pred_basename = os.path.basename(args.predictions)
            pred_name = os.path.splitext(pred_basename)[0]  # Remove extension
            
            # Generate output path: evaluation-results/predictions_neural_eval.json
            args.output = os.path.join(args.output_dir, f"{pred_name}_eval.json")
        else:
            # For model mode, use a default name
            args.output = os.path.join(args.output_dir, 'evaluation_results.json')
        
        print(f"Output file auto-generated: {args.output}")
    else:
        # If output path is provided but doesn't include directory, add output_dir
        if not os.path.dirname(args.output):
            os.makedirs(args.output_dir, exist_ok=True)
            args.output = os.path.join(args.output_dir, args.output)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluation mode: {args.mode}")
    
    # Load test data
    print(f"\nLoading test data...")
    with open(args.test_src, 'r', encoding='utf-8') as f:
        test_src = [line.strip() for line in f]
    with open(args.test_tgt, 'r', encoding='utf-8') as f:
        test_tgt = [line.strip() for line in f]
    
    if len(test_src) != len(test_tgt):
        print(f"ERROR: Source ({len(test_src)}) and target ({len(test_tgt)}) have different lengths!")
        sys.exit(1)
    
    print(f"Loaded {len(test_src)} test examples")
    
    if args.max_samples:
        test_src = test_src[:args.max_samples]
        test_tgt = test_tgt[:args.max_samples]
        print(f"Limited to {len(test_src)} samples")
    
    # Initialize variables for model info
    model_info = {}
    checkpoint = None
    
    # Evaluate based on mode
    eval_start = time.time()
    
    if args.mode == 'model':
        # Load and evaluate model
        print(f"\nLoading model from {args.model}...")
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        
        print(f"Source vocab size: {len(src_vocab)}")
        print(f"Target vocab size: {len(tgt_vocab)}")
        
        # Reconstruct model
        model = TransformerModel(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=checkpoint['args'].d_model,
            nhead=checkpoint['args'].nhead,
            num_encoder_layers=checkpoint['args'].num_layers,
            num_decoder_layers=checkpoint['args'].num_layers,
            dropout=checkpoint['args'].dropout
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create dataset
        test_dataset = TibetanDataset(test_src, test_tgt, src_vocab, tgt_vocab)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Evaluate
        sources, references, hypotheses = evaluate_model(model, test_loader, src_vocab, tgt_vocab, device, args.max_samples)
        
        # Store model info
        model_info = {
            'model_path': args.model,
            'source_vocab_size': len(src_vocab),
            'target_vocab_size': len(tgt_vocab),
            'uses_neural_model': True
        }
        
    else:  # mode == 'predictions'
        # Load pre-generated predictions
        print(f"\nLoading predictions from {args.predictions}...")
        hypotheses = load_predictions_from_file(args.predictions)
        
        if len(hypotheses) != len(test_src):
            print(f"ERROR: Predictions ({len(hypotheses)}) and source ({len(test_src)}) have different lengths!")
            sys.exit(1)
        
        sources = test_src
        references = test_tgt
        
        print(f"Loaded {len(hypotheses)} predictions")
        
        # Store inference method info
        model_info = {
            'predictions_file': args.predictions,
            'inference_method': args.inference_method,
            'uses_neural_model': args.uses_neural_model,
            'uses_kenlm': args.uses_kenlm,
            'uses_preprocessing': args.uses_preprocessing,
            'uses_postprocessing': args.uses_postprocessing,
        }
        
        if args.uses_neural_model and args.model:
            model_info['model_path'] = args.model
            # Try to load model info if available
            if os.path.exists(args.model):
                try:
                    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
                    model_info['source_vocab_size'] = len(checkpoint.get('src_vocab', {}))
                    model_info['target_vocab_size'] = len(checkpoint.get('tgt_vocab', {}))
                except Exception as e:
                    print(f"Warning: Could not load model info: {e}")
        
        if args.uses_kenlm and args.kenlm_path:
            model_info['kenlm_path'] = args.kenlm_path
        
        if args.description:
            model_info['description'] = args.description
    
    # Calculate metrics
    metrics = calculate_metrics(references, hypotheses, sources)
    eval_time = time.time() - eval_start

    # Compute bootstrap confidence intervals
    if args.bootstrap_n > 0:
        print(f"\nComputing 95% bootstrap CIs ({args.bootstrap_n} iterations)...")
        cis = bootstrap_ci(references, hypotheses, sources, n_iterations=args.bootstrap_n)
        print("Done.")
    else:
        cis = None
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    if cis:
        print(f"Character Error Rate (CER): {metrics['cer']*100:.2f}%  (95% CI: {cis['cer']['lower']*100:.2f}–{cis['cer']['upper']*100:.2f}%)")
        print(f"Precision: {metrics['precision']*100:.2f}%  (95% CI: {cis['precision']['lower']*100:.2f}–{cis['precision']['upper']*100:.2f}%)")
        print(f"Recall: {metrics['recall']*100:.2f}%  (95% CI: {cis['recall']['lower']*100:.2f}–{cis['recall']['upper']*100:.2f}%)")
        print(f"F1 Score: {metrics['f1']*100:.2f}%  (95% CI: {cis['f1']['lower']*100:.2f}–{cis['f1']['upper']*100:.2f}%)")
        print(f"Correcting Recall (CR): {metrics['correcting_recall']*100:.2f}%  (95% CI: {cis['correcting_recall']['lower']*100:.2f}–{cis['correcting_recall']['upper']*100:.2f}%)")
        print(f"Correcting Precision (CP): {metrics['correcting_precision']*100:.2f}%  (95% CI: {cis['correcting_precision']['lower']*100:.2f}–{cis['correcting_precision']['upper']*100:.2f}%)")
    else:
        print(f"Character Error Rate (CER): {metrics['cer']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall: {metrics['recall']*100:.2f}%")
        print(f"F1 Score: {metrics['f1']*100:.2f}%")
        print(f"Correcting Recall (CR): {metrics['correcting_recall']*100:.2f}%")
        print(f"Correcting Precision (CP): {metrics['correcting_precision']*100:.2f}%")
    print("="*60)
    print(f"Evaluation time: {timedelta(seconds=int(eval_time))}")
    print(f"Samples evaluated: {len(sources)}")
    print("="*60)
    
    # Save results
    results = {
        'evaluation_mode': args.mode,
        'test_src': args.test_src,
        'test_tgt': args.test_tgt,
        'samples_evaluated': len(sources),
        'evaluation_time_seconds': eval_time,
        'metrics': {
            'character_error_rate': float(metrics['cer']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'correcting_recall': float(metrics['correcting_recall']),
            'correcting_precision': float(metrics['correcting_precision']),
            'total_errors': metrics['total_errors'],
            'correctly_corrected': metrics['correctly_corrected'],
            'identified_errors': metrics['identified_errors']
        }
    }

    if cis:
        results['bootstrap_cis_95'] = {
            'n_iterations': args.bootstrap_n,
            'character_error_rate': cis['cer'],
            'precision': cis['precision'],
            'recall': cis['recall'],
            'f1_score': cis['f1'],
            'correcting_recall': cis['correcting_recall'],
            'correcting_precision': cis['correcting_precision'],
        }
    
    # Add model/inference info
    results.update(model_info)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Generate text report
    report_file = args.output.replace('.json', '_CIreport.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TIBETAN TEXT NORMALIZATION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("EVALUATION INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Evaluation Mode:      {args.mode}\n")
        f.write(f"Test Source:          {args.test_src}\n")
        f.write(f"Test Target:          {args.test_tgt}\n")
        f.write(f"Samples Evaluated:    {len(sources)}\n")
        f.write("\n")
        
        f.write("INFERENCE METHOD\n")
        f.write("-"*70 + "\n")
        if args.mode == 'model':
            f.write(f"Method:               Neural seq2seq model\n")
            f.write(f"Model Path:           {args.model}\n")
            f.write(f"Source Vocabulary:    {model_info.get('source_vocab_size', 'N/A')}\n")
            f.write(f"Target Vocabulary:    {model_info.get('target_vocab_size', 'N/A')}\n")
        else:
            f.write(f"Method:               {args.inference_method}\n")
            f.write(f"Predictions File:     {args.predictions}\n")
            f.write(f"Uses Neural Model:    {'Yes' if args.uses_neural_model else 'No'}\n")
            f.write(f"Uses KenLM:           {'Yes' if args.uses_kenlm else 'No'}\n")
            f.write(f"Uses Preprocessing:   {'Yes' if args.uses_preprocessing else 'No'}\n")
            f.write(f"Uses Postprocessing:  {'Yes' if args.uses_postprocessing else 'No'}\n")
            
            if args.uses_neural_model and args.model:
                f.write(f"Neural Model Path:    {args.model}\n")
                if 'source_vocab_size' in model_info:
                    f.write(f"Source Vocabulary:    {model_info['source_vocab_size']}\n")
                    f.write(f"Target Vocabulary:    {model_info['target_vocab_size']}\n")
            
            if args.uses_kenlm and args.kenlm_path:
                f.write(f"KenLM Model Path:     {args.kenlm_path}\n")
            
            if args.description:
                f.write(f"\nDescription:\n")
                f.write(f"{args.description}\n")
        f.write("\n")
        
        # Add training parameters if available (for model mode or if checkpoint loaded)
        if checkpoint is not None and 'args' in checkpoint:
            f.write("NEURAL MODEL PARAMETERS\n")
            f.write("-"*70 + "\n")
            try:
                train_args = checkpoint['args']
                f.write(f"Model Dimension (d_model):     {train_args.d_model}\n")
                f.write(f"Attention Heads (nhead):       {train_args.nhead}\n")
                f.write(f"Encoder/Decoder Layers:        {train_args.num_layers}\n")
                f.write(f"Dropout:                       {train_args.dropout}\n")
                f.write(f"Batch Size:                    {train_args.batch_size}\n")
                f.write(f"Learning Rate:                 {train_args.lr}\n")
                f.write(f"Epochs:                        {train_args.epochs}\n")
                f.write(f"Early Stopping:                {train_args.early_stopping}\n")
                f.write(f"Gradient Accumulation:         {train_args.gradient_accumulation_steps}\n")
                f.write(f"Weight Decay:                  {train_args.weight_decay}\n")
                if hasattr(train_args, 'val_split'):
                    f.write(f"Validation Split:              {train_args.val_split}\n")
                if hasattr(train_args, 'test_split'):
                    f.write(f"Test Split:                    {train_args.test_split}\n")
            except Exception as e:
                f.write(f"Could not load training parameters: {e}\n")
            f.write("\n")
        
        f.write("EVALUATION METRICS\n")
        if cis:
            f.write(f"(± values are 95% bootstrap CIs, n={args.bootstrap_n} iterations)\n")
        f.write("-"*70 + "\n")

        def fmt(label, val, ci_key):
            point = val * 100
            if cis:
                half = (cis[ci_key]['upper'] - cis[ci_key]['lower']) * 100 / 2
                return f"{label}{point:.2f}% (±{half:.2f}%)\n"
            return f"{label}{point:.2f}%\n"

        f.write(fmt("Character Error Rate (CER):    ", metrics['cer'],                'cer'))
        f.write(fmt("Precision:                     ", metrics['precision'],           'precision'))
        f.write(fmt("Recall:                        ", metrics['recall'],              'recall'))
        f.write(fmt("F1 Score:                      ", metrics['f1'],                 'f1'))
        f.write(fmt("Correcting Recall (CR):        ", metrics['correcting_recall'],   'correcting_recall'))
        f.write(fmt("Correcting Precision (CP):     ", metrics['correcting_precision'],'correcting_precision'))
        f.write("\n")
        f.write(f"Error Correction Statistics:\n")
        f.write(f"  Total Errors (Etotal):       {metrics['total_errors']}\n")
        f.write(f"  Correctly Corrected (Ccorr): {metrics['correctly_corrected']}\n")
        f.write(f"  Identified Errors (Eident):  {metrics['identified_errors']}\n")
        f.write("\n")
        
        f.write("TIMING\n")
        f.write("-"*70 + "\n")
        f.write(f"Evaluation Time:               {timedelta(seconds=int(eval_time))}\n")
        f.write(f"Samples per Second:            {len(sources)/eval_time:.1f}\n")
        f.write("\n")
        
        f.write("EXAMPLE PREDICTIONS (10 Random Examples)\n")
        f.write("-"*70 + "\n")
        num_examples = min(10, len(sources))
        
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
    
    print(f"✓ Text report saved to {report_file}")
    
    # Show 5 random examples in terminal
    print("\n" + "="*60)
    print("5 RANDOM EXAMPLES")
    print("="*60)
    indices = random.sample(range(len(sources)), min(5, len(sources)))
    for idx, i in enumerate(indices, 1):
        print(f"\nExample {idx}:")
        print(f"  Source:    {sources[i][:80]}")
        print(f"  Target:    {references[i][:80]}")
        print(f"  Predicted: {hypotheses[i][:80]}")

if __name__ == '__main__':
    main()