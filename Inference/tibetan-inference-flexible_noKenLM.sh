#!/bin/bash
# Name of the job
#SBATCH -J tibtrain_6000
# time: 6 hours
#SBATCH --time=6:0:0
# Number of GPU
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

PYTHONPATH=$(pwd) python3 tibetan-inference-flexible.py \
    --model_path ../tibetan_model_6000.pt \
    --lm_backend python \
    --input_file GoldTest_source.txt \
    --output_file GoldTest_predictions_noKenLM.txt \
    --beam_width 5 \
    --lm_weight 0.2 \
    --length_penalty 0.6

