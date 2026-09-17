#!/usr/bin/env python3
"""
Prepare sliding-window feature vectors for LSTM training from enriched track CSVs.

Source: Notebook 3 — cells 4 (make_input_with_track), 6 (per-file loop), 11 (accumulation).

Deliberate differences from the notebook:
- ``track`` is the unique granule+beam id (e.g. ``20191104195311_05940510_gt1r``), not just the
  beam, so tracks from different days never collide downstream.
- Files that belong to the same granule+beam (e.g. two Sentinel-2 tiles) are merged before
  windowing, so the along-track series is contiguous.
- Positional metadata (x_atc, year, month, day, lon, lat) is carried next to the features so that
  inference/freeboard can recover location without the notebook's float-equality merge.
- Feature vectors containing NaN are dropped (a track without any open water has NaN
  ``rel_height_*`` everywhere); the notebook let them through.
- The 2 m spacing check uses a tolerance instead of exact float equality.
"""

import argparse
import glob
import logging
import os
import re
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "h_cor_mean",
    "h_diff",
    "rel_height_min_elev",
    "height_sd",
    "pcnth_mean",
    "pcnt_mean",
    "bcnt_mean",
    "brate_mean",
]

# Metadata carried alongside the features (never fed to the model).
META_COLS = ["x_atc", "year", "month", "day", "lon", "lat"]

TRACK_RE = re.compile(r"ATL03_(\d{14})_(\d{8})_.*?(gt\d[lr])", re.IGNORECASE)
BEAM_RE = re.compile(r"(gt\d[lr])", re.IGNORECASE)


def extract_track_id(filename):
    """Return a unique track id ``<datetime>_<rgtcycle>_<beam>`` from an ATL03-style filename.

    Falls back to the bare beam (legacy behaviour) and finally to the file stem.
    """
    base = os.path.basename(filename)
    m = TRACK_RE.search(base)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3).lower()}"
    m = BEAM_RE.search(base)
    if m:
        return m.group(1).lower()
    return os.path.splitext(base)[0]


def feature_columns(features, nearby):
    """Column names in the order written by make_input_with_track (timestep-major)."""
    return [f"{f}{t}" for t in range(-nearby, nearby + 1) for f in features]


def make_input_with_track(data, fields, nearby, spacing=2.0, tol=1e-3):
    """Construct sliding-window feature vectors for the LSTM (vectorised).

    For every centre point whose ±nearby neighbours are exactly ``spacing`` metres apart
    (within ``tol``), emit ``label``, ``track``, metadata and (2*nearby+1) x len(fields)
    feature columns named ``<field><offset>``.
    """
    start = time.time()
    n = len(data)
    if n < 2 * nearby + 1:
        return pd.DataFrame()

    x = data["x_atc"].to_numpy(dtype=float)
    centers = np.arange(nearby, n - nearby)
    ok = np.ones(len(centers), dtype=bool)
    for off in range(-nearby, nearby + 1):
        if off == 0:
            continue
        ok &= np.abs((x[centers + off] - x[centers]) - off * spacing) <= tol
    valid = centers[ok]

    cols = {
        "label": data["label"].to_numpy()[valid],
        "track": data["track"].to_numpy()[valid],
    }
    for m in META_COLS:
        if m in data.columns:
            cols[m] = data[m].to_numpy()[valid]
    for off in range(-nearby, nearby + 1):
        for f in fields:
            cols[f"{f}{off}"] = data[f].to_numpy()[valid + off]

    out = pd.DataFrame(cols)
    logger.info("  %d / %d centre points are contiguous (%.1fs)", len(out), n, time.time() - start)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sliding-window feature vectors for LSTM from enriched track CSVs"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing enriched track CSVs")
    parser.add_argument("--output", required=True, help="Output prepared CSV")
    parser.add_argument("--nearby", type=int, default=2,
                        help="Neighbouring 2m segments on each side (default: 2)")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES),
                        help="Comma-separated feature column names")
    parser.add_argument("--pattern", type=str, default="*.csv",
                        help="Glob pattern for input CSV files (default: *.csv)")
    parser.add_argument("--spacing", type=float, default=2.0,
                        help="Expected along-track spacing in metres (default: 2)")
    parser.add_argument("--keep-nan", action="store_true",
                        help="Keep feature vectors that contain NaN (notebook behaviour)")
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",")]
    logger.info("Features (%d): %s", len(features), features)
    logger.info("Nearby: %d (timesteps: %d)", args.nearby, 2 * args.nearby + 1)

    filelist = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    out_abs = os.path.abspath(args.output)
    filelist = [f for f in filelist if os.path.abspath(f) != out_abs]
    logger.info("Found %d input files", len(filelist))
    if not filelist:
        logger.error("No files found in %s matching %s", args.input_dir, args.pattern)
        sys.exit(1)

    # Group files by unique track id (granule + beam); tiles of one track are merged.
    groups = {}
    for filepath in filelist:
        groups.setdefault(extract_track_id(filepath), []).append(filepath)
    logger.info("Unique tracks: %d", len(groups))

    all_datasets = []
    for track_id, files in sorted(groups.items()):
        logger.info("Track %s (%d file(s))", track_id, len(files))
        frames = []
        for filepath in files:
            logger.info("  Reading %s", filepath)
            d = pd.read_csv(filepath)
            d = d.loc[:, ~d.columns.str.contains("^Unnamed")]
            frames.append(d)
        data = pd.concat(frames, ignore_index=True)

        missing = [f for f in features if f not in data.columns]
        if missing:
            logger.warning("  Skipping track %s — missing columns: %s", track_id, missing)
            continue

        # Deduplicate by (x_atc, label) and sort along track (Notebook 3, cell 6)
        data = data.drop_duplicates(subset=["x_atc", "label"])
        data = data.sort_values(by=["x_atc"]).reset_index(drop=True)
        data["track"] = track_id

        if "rel_height_min_elev" in data.columns and data["rel_height_min_elev"].isna().all():
            logger.warning("  Track %s has no open water: rel_height_* is NaN for every row", track_id)

        dataset = make_input_with_track(data, features, args.nearby, spacing=args.spacing)
        if len(dataset) == 0:
            logger.warning("  Track %s produced no feature vectors", track_id)
            continue

        if not args.keep_nan:
            fcols = feature_columns(features, args.nearby)
            nan_mask = dataset[fcols].isna().any(axis=1) | dataset["label"].isna()
            if nan_mask.any():
                logger.warning("  Dropping %d / %d vectors with NaN features/labels",
                               int(nan_mask.sum()), len(dataset))
                dataset = dataset[~nan_mask].reset_index(drop=True)

        logger.info("  Generated %d feature vectors for track %s", len(dataset), track_id)
        all_datasets.append(dataset)

    if not all_datasets:
        logger.error("No datasets generated — no files produced feature vectors")
        sys.exit(1)

    result = pd.concat(all_datasets, ignore_index=True)
    if len(result) == 0:
        logger.error("All files processed but 0 total feature vectors generated")
        sys.exit(1)

    n_feat = len(features) * (2 * args.nearby + 1)
    n_meta = sum(1 for m in META_COLS if m in result.columns)
    logger.info("Total feature vectors: %d", len(result))
    logger.info("Columns: label + track + %d metadata + %d features = %d (actual %d)",
                n_meta, n_feat, 2 + n_meta + n_feat, len(result.columns))
    logger.info("Label counts: %s", result["label"].value_counts().to_dict())
    logger.info("Vectors per track: %s", result["track"].value_counts().to_dict())

    result.to_csv(args.output, index=False)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
