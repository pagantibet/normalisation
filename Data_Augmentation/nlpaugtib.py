import argparse
import os
import random

# all of this is based on Claude's interpretation of the nlpaug code on github: https://github.com/makcedward/nlpaug
# It's meant to create new sentence pairs from the original (i.e. normalised Classical tibetan) 
# to the noisy unnormalised data which is supposed to be more like the diplomatic text
# although it doesn't have all the abbreviations.

# To run data augmentation, assuming all files are in the same folder:
# python3 nlpaugtib.py  --input unsegACTib-tok_lines_ocr_noise_500k.txt --type segmented
# python3 nlpaugtib.py  --input unsegACTib_lines_ocr_noise_500k.txt --type nonsegmented

TSHEG = "་"


def split_syllables(text):
    syllables = []
    buf = ""
    for ch in text:
        buf += ch
        if ch == TSHEG:
            syllables.append(buf)
            buf = ""
    if buf.strip():
        syllables.append(buf)
    return syllables


def swap_units(units, aug_prob):
    out = units[:]
    i = 0
    while i < len(out) - 1:
        if random.random() < aug_prob:
            out[i], out[i + 1] = out[i + 1], out[i]
            i += 2
        else:
            i += 1
    return out


def augment_segmented_line(line, aug_prob):
    utt = ""
    if "<utt>" in line:
        line = line.replace("<utt>", "").strip()
        utt = " <utt>"

    words = line.split()
    if len(words) < 2:
        return line + utt

    # swap whole words
    augmented_words = swap_units(words, aug_prob)
    return " ".join(augmented_words) + utt


def augment_nonsegmented_line(line, aug_prob):
    utt = ""
    if "<utt>" in line:
        line = line.replace("<utt>", "").strip()
        utt = " <utt>"

    syllables = split_syllables(line)
    if len(syllables) < 2:
        return line + utt

    augmented_syllables = swap_units(syllables, aug_prob)
    return "".join(augmented_syllables) + utt


def main():
    parser = argparse.ArgumentParser(
        description="Tibetan data augmentation (word-aware, syllable-safe)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--type",
        choices=["segmented", "nonsegmented"],
        required=True
    )
    parser.add_argument("--aug_prob", type=float, default=0.05)

    args = parser.parse_args()

    base, _ = os.path.splitext(args.input)
    output_path = base + "_augmented.txt"

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    augmented = []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            augmented.append("")
            continue

        if args.type == "segmented":
            augmented.append(
                augment_segmented_line(line, args.aug_prob)
            )
        else:
            augmented.append(
                augment_nonsegmented_line(line, args.aug_prob)
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for line in augmented:
            f.write(line + "\n")

    print(f"Augmented file written to: {output_path}")


if __name__ == "__main__":
    main()
