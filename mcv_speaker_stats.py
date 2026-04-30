#!/usr/bin/env python3
"""
Quick scan of a local Mozilla Common Voice TSV file to count
how many utterances are available at various speaker-per-sentence thresholds.

Usage:
    python mcv_speaker_stats.py --data_dir /opt/modeling/dpluth/data/mcv/en
    python mcv_speaker_stats.py --data_dir /path/to/mcv/en --tsv validated.tsv
"""

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

THRESHOLDS = [10, 50, 100, 250, 500, 1000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Directory containing MCV TSV files and clips/")
    parser.add_argument("--tsv", default="validated.tsv",
                        help="Which TSV file to use (default: validated.tsv)")
    args = parser.parse_args()

    tsv_path = args.data_dir / args.tsv
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    print(f"Reading {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t", usecols=["client_id", "sentence"],
                     low_memory=False)
    print(f"  {len(df):,} rows, {df['client_id'].nunique():,} unique speakers, "
          f"{df['sentence'].nunique():,} unique sentences\n")

    # Count unique speakers per sentence
    speakers_per_sentence = df.groupby("sentence")["client_id"].nunique()

    df["n_words"] = df["sentence"].str.split().str.len()

    print(f"{'min_speakers':>14}  {'qualifying_sentences':>22}  {'total_utterances':>18}  {'total_words':>14}  {'avg_spk/sentence':>18}  {'med_spk/sentence':>18}")
    print("-" * 112)
    for threshold in THRESHOLDS:
        qualifying_sentences = speakers_per_sentence[speakers_per_sentence >= threshold]
        subset       = df[df["sentence"].isin(qualifying_sentences.index)]
        n_sentences  = len(qualifying_sentences)
        n_utterances = int(subset.shape[0])
        n_words      = int(subset["n_words"].sum())
        avg_spk      = qualifying_sentences.mean()
        med_spk      = qualifying_sentences.median()
        print(f"{threshold:>14,}  {n_sentences:>22,}  {n_utterances:>18,}  {n_words:>14,}  {avg_spk:>18.1f}  {med_spk:>18.1f}")

    # Top-10 sentences by speaker count
    top = speakers_per_sentence.sort_values(ascending=False).head(10)
    print("\nTop 10 sentences by speaker count:")
    for sentence, n_spk in top.items():
        preview = sentence[:60] + ("..." if len(sentence) > 60 else "")
        print(f"  {n_spk:>5,} speakers — {preview!r}")


if __name__ == "__main__":
    main()
