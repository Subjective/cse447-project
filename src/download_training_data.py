#!/usr/bin/env python
"""
Gather diverse multilingual text from the Hugging Face "oscar" corpus
and write them to training_data.txt for subsequent model training.
"""

import os
from datasets import load_dataset

def gather_multilingual_data(
    languages=None,
    num_lines_per_lang=1000,
    output_file="src/training_data.txt"
):
    """
    Gathers text from the OSCAR corpus for the specified languages,
    then writes them to `training_data.txt`.

    :param languages: list of language codes (e.g., ["en", "fr", "de", "zh"]).
    :param num_lines_per_lang: how many lines to gather for each language.
    :param output_file: path to the training data file.
    """

    if languages is None:
        languages = [
            "en", "fr", "es", "de", "it", "ru", "ar", "zh"
        ]

    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    total_lines_collected = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for lang in languages:
            print(f"[INFO] Gathering data for language: {lang}")

            # Check for available configurations
            possible_configs = [f"deduplicated_{lang}", f"original_{lang}"]
            dataset_found = None

            for config in possible_configs:
                try:
                    dataset = load_dataset(
                        "oscar-corpus/OSCAR-2109",
                        config,
                        split="train",
                        streaming=True
                    )
                    dataset_found = config
                    break
                except:
                    continue  # Try the next config

            if dataset_found is None:
                print(f"[WARN] Couldn't find valid dataset config for '{lang}'. Skipping.")
                continue

            count = 0
            for sample in dataset:
                text = sample.get("text", "")
                lines = [line.strip() for line in text.split("\n") if line.strip()]

                for line in lines:
                    f_out.write(line + "\n")
                    count += 1
                    total_lines_collected += 1

                    if count >= num_lines_per_lang:
                        break
                if count >= num_lines_per_lang:
                    break

            print(f"[INFO] Collected {count} lines for '{lang}' from {dataset_found}.")

    print(f"[INFO] Total lines collected: {total_lines_collected}")
    print(f"[INFO] Final data written to: {output_file}")


if __name__ == "__main__":
    gather_multilingual_data(
        languages=None,
        num_lines_per_lang=1000,
        output_file="src/training_data.txt"
    )
