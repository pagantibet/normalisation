# Normalisation

This repo contains the code for the Normalisation part of PaganTibet described in Meelen & Griffiths (2026). Please cite the repo and the following article when using any part of this code:

Meelen, M. & Griffiths, R.M. (2026) 'Historical Tibetan Normalisation: rule-based vs neural & n-gram LM methods for extremely low-resource languages' in _Proceedings of the AI4CHIEF conference_, Springer.

**Abstract**. _Historical Tibetan manuscripts present significant normalisation challenges due to difficulties including extensive abbreviations, non-standard orthography, and a complete lack of established gold-standard data. This paper presents a hybrid approach combining rule-based methods with character-level encoder-decoder transformer models enhanced with n-gram-based language models to normalise extremely difficult diplomatic Tibetan texts into Standard Classical Tibetan. We address the scarcity of parallel training data through data augmentation, compare
tokenised and non-tokenised approaches, and evaluate performance on different types of test sets. This work contributes to the understudied task of historical text normalisation, with implications beyond Tibetan, for digital humanities and no/low-resource language work._

# Datasets & Preparation

Datasets for the experiments in Meelen & Griffiths (2026) had to be prepared in various ways before they could be used for training, validation, and testing. We used three datasets in our experiments:

- a 'Gold-standard' collection of manually-normalised 7421 paired sentences from the PaganTibet corpus
- the Standard Classical Tibetan ACTib (>180m words), available from [Zenodo (Meelen & Roux 2020)](https://zenodo.org/records/3951503)
- a custom-made abbreviation dictionary for rule-based replacements (around 10k abbreviations
with expansions)

Datasets and models can be found on the [PaganTibet Huggingface](https://huggingface.co/pagantibet).

## Tokenisation

To test the effect of tokenisation, we prepared both tokenised and non-tokenised versions of each dataset using a customised version of the [Botok Tibetan tokeniser](www.github.com/OpenPecha/botok). Note that our results show tokenisation is best lest until after Normalisation in the pipeline. To tokenise source and target data:

```
python3 botokenise_src-tgt.py
```

The full ReadMe of this script can be found in [Data_Preparation/botokenise_src-tgt_ReadMe]().

## Creating lines & cleaning text

Since Normalisation is in essence a sequence-2-sequence task that ideally requires some context, manuscript lines were chosen as sequence units since this is how they appear in the manually-normalised 'Gold' data. Since the ACTib does not contain linebreaks and generally contains some non-Tibetan materials (e.g. page numbers) that should be cleaned before Normalisation, cleaning and artificial linebreaks of reasonably-varying lengths can be created in the following way:

```
python3 createTiblines.py <input_file> <output_file> [options]
```

The full ReadMe of this script can be found in [Data_Augmentation/createTiblines_ReadMe]().

# Data Augmentation

To solve the issue of data scarcity, we offer four data augmentation methods: random noise injection, OCR-based Noise Simulation, Rule-Based Diplomatic Transformations and Dictionary-based Augmentation.

## Random noise injection

We developed a custom noise injection script to simulate naturally-occurring scribal variations in diplomatic texts, following Huang et al’s (2023) random noise formula. The noise injection follows a probabilistic model calibrated to reflect realistic manuscript variation frequencies, including character substitutions, diacritic variations, and orthographic inconsistencies common in Classical Tibetan documents.

```
python3 Tibrandomnoiseaugmentation.py my_corpus.txt
```

## OCR-based Noise Simulation

## Rule-Based Diplomatic Transformations

## Dictionary-based Augmentation

# Model Architecture

# Evaluations

# Inference
