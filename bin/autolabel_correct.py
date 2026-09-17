#!/usr/bin/env python3
"""
Trim leading/trailing open-water chunks from auto-labeled ICESat-2 track CSVs.

Source: Notebook 1 — del_last_first_chunk() and column cleanup logic.

Notebook 1 applied this only to raw auto-labelled tracks that begin and/or end in open water
(label 2), before manual correction. Its scan found the *last* open-water chunk anywhere in the
track, which silently deletes everything after the last interior lead when the track does not end
in open water. This version only trims when the track actually starts/ends with open water, and is
a no-op otherwise. Do not run it on already corrected ``*_done.csv`` files (the workflow generator
skips it for them by default).
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPEN_WATER = 2.0


def del_last_first_chunk(df_file, corner_offset=100):
    """Remove a leading and/or trailing continuous open-water (label==2) run plus an offset.

    Parameters
    ----------
    df_file : pd.DataFrame
        Labeled track data with a 'label' column, sorted along track.
    corner_offset : int
        Extra rows to trim after/before the open-water run. Capped at 10% of the data length.

    Returns
    -------
    pd.DataFrame
        Trimmed DataFrame with reset index (unchanged if no edge open water).
    """
    df_file = df_file.reset_index(drop=True)
    n = len(df_file)
    if n == 0:
        return df_file
    corner_offset = min(corner_offset, max(0, n // 10))
    labels = df_file["label"].to_numpy()

    # Trailing open-water run
    if labels[-1] == OPEN_WATER:
        i = n - 1
        while i >= 0 and labels[i] == OPEN_WATER:
            i -= 1
        run_start = i + 1
        drop_start = max(0, run_start - corner_offset)
        logger.info("Trailing open-water run of %d rows (+%d offset) removed",
                    n - run_start, run_start - drop_start)
        df_file = df_file.iloc[:drop_start].reset_index(drop=True)
        labels = labels[:drop_start]
        n = len(df_file)

    # Leading open-water run
    if n > 0 and labels[0] == OPEN_WATER:
        i = 0
        while i < n and labels[i] == OPEN_WATER:
            i += 1
        drop_end = min(n, i + corner_offset)
        logger.info("Leading open-water run of %d rows (+%d offset) removed", i, drop_end - i)
        df_file = df_file.iloc[drop_end:].reset_index(drop=True)

    return df_file


def clean_columns(df):
    """Remove Unnamed index columns."""
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def main():
    parser = argparse.ArgumentParser(
        description="Trim leading/trailing open-water chunks from a labeled IS2 track CSV"
    )
    parser.add_argument("--input", required=True, help="Input labeled CSV")
    parser.add_argument("--output", required=True, help="Output corrected CSV")
    parser.add_argument("--corner-offset", type=int, default=100,
                        help="Extra rows to trim beyond the open-water run (default: 100)")
    args = parser.parse_args()

    logger.info("Loading %s", args.input)
    df = pd.read_csv(args.input, index_col=0)
    df = clean_columns(df.reset_index(drop=True))
    logger.info("Input rows: %d", len(df))

    if "label" not in df.columns:
        raise SystemExit("Input has no 'label' column")

    if df["label"].iloc[0] != OPEN_WATER and df["label"].iloc[-1] != OPEN_WATER:
        logger.info("Track does not start or end with open water — nothing to trim")
    else:
        df = del_last_first_chunk(df, corner_offset=args.corner_offset)

    logger.info("Output rows: %d", len(df))
    df.to_csv(args.output, index=False)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
