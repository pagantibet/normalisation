#!/bin/bash
# Name of the job
#SBATCH -J tibtrain_6000
# time: 15 hours
#SBATCH --time=15:0:0
# Request RTX 6000 Ada
#SBATCH --gres=gpu:rtx_6000_ada:1
# Number of cpus
#SBATCH --cpus-per-task=16
# Log output
#SBATCH -e ./slurm-err-%j.txt
#SBATCH -o ./slurm-out-%j.txt
#SBATCH --open-mode=append

# Start your application
eval "$(conda shell.bash hook)"

conda activate pagantibenv

# OPTIMIZED FOR QUALITY + SPEED ON RTX 6000 ADA
# Focuses on improving those CER/Precision/Recall metrics

python3 tibtrainencdecoder_witheval.py \
  --train_src train_source-tok.txt \
  --train_tgt train_target-tok.txt \
  --d_model 512 \
  --num_layers 4 \
  --nhead 8 \
  --batch_size 128 \
  --gradient_accumulation_steps 1 \
  --lr 0.0005 \
  --dropout 0.15 \
  --weight_decay 0.0001 \
  --early_stopping 5 \
  --epochs 15 \
  --save_every 3 \
  --test_split 0.005 \
  --val_split 0.01 \
  --use_normalized_vocab \
  --checkpoint_dir checkpoints-tok \
  --beam_width 5 \
  --save_model tibetan_model_tokenised_allchars.pt \
  --checkpoint_dir checkpoints_tokenised_allchars \
  --results_file training_results_tokenised_allchars.json \
  --report_file tibetan_report_tokenised_allchars.txt

