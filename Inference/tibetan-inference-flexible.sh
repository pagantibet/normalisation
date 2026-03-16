#!/bin/bash
# Name of the job
#SBATCH -J tibtrain_inference
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

# OPTIMIZED FOR QUALITY + SPEED ON RTX rtx_4090

PYTHONPATH=$(pwd) python3 tibetan-inference-flexible.py \
    --mode rules+neural \
    --model_path ../tibetan_model_tokenised_allchars.pt \
    --kenlm_path model_5gram_char-tok.arpa \
    --lm_backend python \
    --input_file ACTibOCRnoiseTest_source-tok.txt \
    --rules_dict abbreviations.txt

# ============================================================================
# AVAILABLE MODES & MODELS
# ============================================================================
#

#  --model_path ../tibetan_model_tokenised_allchars.pt \
#  --model_path ../tibetan_model_nontokenized_allchars.pt \

#  --kenlm_path model_5gram_char-tok.arpa \
#  --kenlm_path model_5gram_char.arpa \

# Mode 1: rules
#   --mode rules --rules_dict abbreviations.txt
#   (Rule-based only: dictionary + punctuation fixes, very fast)
#
# Mode 2: neural
#   --mode neural --model_path ../tibetan_model_nontokenized_allchars.pt
#   (Seq2seq only: fast neural normalization)
#
# Mode 3: neural+lm
#   --mode neural+lm --model_path ../tibetan_model_nontokenized_allchars.pt --kenlm_path model.arpa --lm_backend python
#   (Seq2seq + KenLM: neural with language model reranking)
#
# Mode 4: neural+lm+rules
#   --mode neural+lm+rules --model_path ../tibetan_model_nontokenized_allchars.pt --kenlm_path model.arpa --lm_backend python --rules_dict abbreviations.txt
#   (Neural + LM → Rules: full pipeline with postprocessing)
#
# Mode 5: rules+neural+lm
#   --mode rules+neural+lm --model_path ../tibetan_model_nontokenized_allchars.pt --kenlm_path model.arpa --lm_backend python --rules_dict abbreviations.txt
#   (Rules → Neural + LM: preprocessing with rules first)
#
# Mode 6: rules+neural
#   --mode rules+neural --model_path ../tibetan_model_nontokenized_allchars.pt --rules_dict abbreviations.txt
#   (Rules → Neural: simple combination with preprocessing)
#
# ============================================================================
