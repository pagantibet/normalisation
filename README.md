# Normalisation

This repo contains the code for the Normalisation part of PaganTibet described in Meelen & Griffiths (2026). Please cite the repo and the following article when using any part of this code:

Meelen, M. & Griffiths, R.M. (2026) 'Historical Tibetan Normalisation: rule-based vs neural & n-gram LM methods for extremely low-resource languages' in _Proceedings of the AI4CHIEF conference_, Springer.

**Abstract**. _Historical Tibetan manuscripts present significant normalisation challenges due to difficulties including extensive abbreviations, non-standard orthography, and a complete lack of established gold-standard data. This paper presents a hybrid approach combining rule-based methods with character-level encoder-decoder transformer models enhanced with n-gram-based language models to normalise extremely difficult diplomatic Tibetan texts into Standard Classical Tibetan. We address the scarcity of parallel training data through data augmentation, compare tokenised and non-tokenised approaches, and evaluate performance on different types of test sets. This work contributes to the understudied task of historical text normalisation, with implications beyond Tibetan, for digital humanities and no/low-resource language work._

---

## Table of Contents

- [Datasets & Preparation](#datasets--preparation)
  - [Tokenisation](#tokenisation)
  - [Creating lines & cleaning text](#creating-lines--cleaning-text)
- [Data Augmentation](#data-augmentation)
  - [Random noise injection](#random-noise-injection)
  - [OCR-based Noise Simulation](#ocr-based-noise-simulation)
  - [Rule-Based Diplomatic Transformations](#rule-based-diplomatic-transformations)
  - [Dictionary-based Augmentation](#dictionary-based-augmentation)
- [Model Architecture & Training](#model-architecture--training)
  - [Seq-2-Seq neural encoder-decoder transformer](#seq-2-seq-neural-encoder-decoder-transformer)
  - [KenLM N-gram model for ranking](#kenlm-n-gram-model-for-ranking)
- [Inference](#inference)
- [Evaluations](#evaluations)

---

# Datasets & Preparation

Datasets for the experiments in Meelen & Griffiths (2026) had to be prepared in various ways before they could be used for training, validation, and testing. We used three datasets in our experiments:

- a 'Gold-standard' collection of manually-normalised 7421 paired sentences from the PaganTibet corpus
- the Standard Classical Tibetan ACTib (>180m words), available from [Zenodo (Meelen & Roux 2020)](https://zenodo.org/records/3951503)
- a custom-made abbreviation dictionary for rule-based replacements (around 10k abbreviations with expansions)

Datasets and models can be found on the [PaganTibet Huggingface](https://huggingface.co/pagantibet).

## Tokenisation

To test the effect of tokenisation, we prepared both tokenised and non-tokenised versions of each dataset using a customised version of the [Botok Tibetan tokeniser](www.github.com/OpenPecha/botok). Note that our results show tokenisation is best left until after Normalisation in the pipeline. To tokenise source and target data:

```
python3 botokenise_src-tgt.py
```

The full ReadMe of this script can be found in [Data_Preparation/botokenise_src-tgt_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Data_Preparation/botokenise_ReadMe.md).

## Creating lines & cleaning text

Since Normalisation is in essence a sequence-2-sequence task that ideally requires some context, manuscript lines were chosen as sequence units since this is how they appear in the manually-normalised 'Gold' data. Since the ACTib does not contain linebreaks and generally contains some non-Tibetan materials (e.g. page numbers) that should be cleaned before Normalisation, cleaning and artificial linebreaks of reasonably-varying lengths can be created in the following way:

```
python3 createTiblines.py <input_file> <output_file> [options]
```

The full ReadMe of this script can be found in [Data_Augmentation/createTiblines_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Data_Preparation/createTiblines_ReadMe.md).

# Data Augmentation

To solve the issue of data scarcity, we offer four data augmentation methods: random noise injection, OCR-based Noise Simulation, Rule-Based Diplomatic Transformations and Dictionary-based Augmentation.

## Random noise injection

We developed a custom noise injection script to simulate naturally-occurring scribal variations in diplomatic texts, following Huang et al's (2023) random noise formula. The noise injection follows a probabilistic model calibrated to reflect realistic manuscript variation frequencies, including character substitutions, diacritic variations, and orthographic inconsistencies common in Classical Tibetan documents.

```
python3 Tibrandomnoiseaugmentation.py my_corpus.txt
```

The full ReadMe of this script can be found in [Data_Augmentation/Tibrandomnoiseaugmentation_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Data_Augmentation/Tibrandomnoiseaugmentation_ReadMe.md).

## OCR-based Noise Simulation

Similar to the random-noise insertion, to model errors introduced during the OCR of Tibetan manuscripts, we employed the [nlpaug python library](github.com/makcedward/nlpaug) to generate OCR-realistic noise patterns specifically for Tibetan texts:

```
python3 nlpaugtib.py --input <input_file.txt> --type <segmented|nonsegmented> [--aug_prob FLOAT]
```

The full ReadMe of this script can be found in [Data_Augmentation/nlpaugtib_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Data_Augmentation/nlpaugtib_ReadMe.py).

## Rule-Based Diplomatic Transformations

For small Gold datasets, we recommend implementing a more-targeted rule-based augmentation strategy using a custom script to generate additional diplomatic variants from normalised text. This script applied rule-based character replacements reflecting common scribal conventions and variations often found in historical Tibetan manuscripts. The script applies these transformations stochastically, with adjustable ratios.

```
python3 tibrule_augmentation.py input.txt --char-ratio 0.1 --syllable-ratio 0.05
```

The full ReadMe of this script can be found in [Data_Augmentation/tibrule_augmentation_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Data_Augmentation/tibrule-augmentation_ReadMe.md).

## Dictionary-based Augmentation

We finally introduce a dictionary-based data augmentation method, adding abbreviation dictionary entries to random lines to help the model recognise and learn these. It can be applied to tokenised (default) or non-tokenised text:

```
python3 dictionary-augmentation.py input.txt abbreviation-dictionary.txt
```

The full ReadMe of this script can be found in [Data_Augmentation/dictionaryaugmentation_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Data_Augmentation/dictionaryaugmentation_ReadMe.md).

# Model Architecture & Training

## Seq-2-Seq neural encoder-decoder transformer

The Seq-2-Seq model has a character-based encoder-decoder transformer architecture with 4 layers and 8 attention heads, using the Adam optimiser (lr = 0.0005, β1 = 0.9, β2 = 0.997) and 0.1 label smoothing. The models were trained on an RTX ADA 6000 GPU, for 5-6 hours each (tokenised and non-tokenised, including beam search evaluation), implemented in PyTorch - full parameter settings can be found in the Appendix of Meelen & Griffiths (2026). To train a similar Seq-2-Seq model:

```
sbatch tibtrain.sh
```

```
python3 tibtrainencdecoder_witheval.py
```

The full ReadMe of this script can be found in [Training/tibtrainencdecoder_witheval_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Training/tibtrainencdecoder_witheval_ReadMe.md).

## KenLM N-gram model for ranking

The character-based tokenised and non-tokenised 5-gram KenLMs for ranking S2S inferences were trained in <5 minutes using Google Colab (host a290405c6e16 using Linux 6.6.105+) on the 8m-line split of the ACTib (pruning thresholds: 0 1 1 2 2; with modified Kneser-Ney discount values) - full parameter settings can be found in the Appendix of Meelen & Griffiths (2026).

All models are available open access on the [PaganTibet Huggingface](https://huggingface.co/pagantibet).

# Inference

To normalise Tibetan text, we make 6 different inference modes available:

1. rule-based only (i.e. only using the replacement rules from the Abbreviation dictionary)
2. neural only (i.e. only using the Seq-2-Seq model)
3. neural+lm (the Seq-2-Seq model with KenLM ranking)
4. neural+lm+rules (the Seq-2-Seq model with KenLM ranking and rule-based replacements as postprocessing)
5. rules+neural+lm (the Seq-2-Seq model with KenLM ranking and rule-based replacements as preprocessing)
6. rules+neural (the Seq-2-Seq model and rule-based replacements as preprocessing)

Note that modes without rule-based pre- and/or postprocessing are likely to yield poorer results for challenging corpora like the PaganTibet ones. For more standard Buddhist texts, neural Seq-2-Seq models only perform reasonably well for non-tokenised text. For further details and full results, see Meelen & Griffiths (2026).

The flexible inference script can be run using slurm on GPU clusters as well or directly using python to create prediciton files and reports:

```
sbatch tibetan-inference-flexible.sh 
```
```
python3 tibetan-inference-flexible.py
```

The full ReadMe of this script can be found in [Inference/tibetan-inference-flexible_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Inference/tibetan-inference-flexible_ReadMe.md).

# Evaluations

The Seq-2-Seq training script has a built-in evaluation model using beam search. All six inference modes can also be evaluated separately using the evaluation script, which outputs standard measures like CER, precision, recall and F1-scores, but also - following Huang et al (2021) - Correction Precision and Recall to get a more accurate picture of how effectively Normalisation was done. For very small datasets, bootstrapping over 1000 iterations to gauge Confidence Intervals (CIs) is recommended. It can be run on prediction files (i.e. the output of the inference scrips) a cluster or directly using python.

```
sbatch evaluate-model.sh
```

```
python3 evaluate_model.py
```

The full ReadMe of this script can be found in [Evaluations/evaluate_model_ReadMe](https://github.com/pagantibet/normalisation/blob/main/Evaluations/evaluate_model_ReadMe.md).

Full details of evaluation results including confidence intervals and example predictions reported in Meelen & Griffiths (2026) can be found in the [tokenised](https://github.com/pagantibet/normalisation/tree/main/Evaluations/Gold-tokenised-CI) and [non-tokenised](https://github.com/pagantibet/normalisation/tree/main/Evaluations/Gold-nontokenised-CI) Evaluation directories.

