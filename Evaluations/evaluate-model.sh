#!/bin/bash
# Name of the job
#SBATCH -J tibtrain_6000
# time: 6 hours
#SBATCH --time=6:0:0
# Number of GPU
#SBATCH --gres=gpu:rtx_4090:1
# Number of cpus
#SBATCH --cpus-per-task=16
# Log output
#SBATCH -e ./slurm-err-%j.txt
#SBATCH -o ./slurm-out-%j.txt
#SBATCH --open-mode=append

# Start your application
eval "$(conda shell.bash hook)"

conda activate pagantibenv

# Can do -predictions or -models evaluation
# Focuses on improving those CER/Precision/Recall metrics

#for predictions evaluation (no gpu needed):

PYTHONPATH=$(pwd) python3 evaluate_model.py \
  --mode predictions \
  --predictions inference/ACTibOCRnoiseTest_source-tok_predictions-6-rules+neural.txt \
  --test_src inference/ACTibOCRnoiseTest_source-tok.txt \
  --test_tgt inference/ACTibOCRnoiseTest_target-tok.txt \
  --inference_method "rules+neural" \
  --uses_neural_model \
  # --uses_kenlm \

# list of inference methods to choose from:
# 1 rules
# 2 neural
# 3 neural+lm
# 4 neural+lm+rules
# 5 rules+neural+lm
# 6 rules+neural

# #for model evaluations:

# python3 evaluate_model.py \
#   --model tibetan_model_tokenized.pt \
#   --test_src test_src-tok_5k.txt \
#   --test_tgt test_tgt-tok_5k.txt \
#   --output evaluation_rtx6000.json