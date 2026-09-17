#!/usr/bin/env python3

"""
Extract ICESat-2 ATL10 freeboard data from raw HDF5 granules into CSV files.

Reads raw ATL10 HDF5 files downloaded from NASA Earthdata, extracts freeboard
and height segment data from all six beams, and parses timestamps from
delta_time + ATLAS epoch.

Handles path differences across ATL10 versions: tries /{beam}/freeboard_segment/
first, then falls back to /{beam}/freeboard_beam_segment/ (older versions).

Output CSVs match the column format expected by validate_freeboard.py and
paper_figures.py (columns: beam, lat, lon, x, seg_id, height, label, h_ref,
freeboard, fb_confidence, fb_quality, h_norm, error, ssh_flag, day, hour,
minute, second).

Usage:
    python bin/extract_atl10.py --input-dir ./data/atl10/ --output-dir ./data/atl10_csv/
    python bin/extract_atl10.py --input data/atl10/ATL10*.h5 --output-dir ./data/atl10_csv/
"""

import argparse
import glob
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_BEAMS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

# ATLAS SDP GPS epoch: 2018-01-01 00:00:00 UTC
ATLAS_EPOCH = datetime(2018, 1, 1, 0, 0, 0)

# Freeboard segment root path variants (try in order).
# ATL10 v3+ uses "freeboard_segment"; older versions use "freeboard_beam_segment".
FB_ROOTS = ["freeboard_segment", "freeboard_beam_segment"]

# HDF5 dataset paths for ATL10 (relative to /{beam}/{fb_root}/).
# Keys: (output column name, list of sub-path variants to try).
ATL10_PATHS = {
    "freeboard": [
        "beam_fb_height",
    ],
    "fb_confidence": [
        "beam_fb_confidence",
    ],
    "fb_quality": [
        "beam_fb_quality_flag",
    ],
    "lat": [
        "geophysical/latitude",
        "latitude",
    ],
    "lon": [
        "geophysical/longitude",
        "longitude",
    ],
    "delta_time": [
        "geophysical/delta_time",
        "delta_time",
    ],
    "x": [
        "geophysical/seg_dist_x",
        "seg_dist_x",
    ],
    "height": [
        "height_segments/height_segment_height",
    ],
    "label": [
        "height_segments/height_segment_type",
    ],
    "seg_id": [
        "height_segments/height_segment_id",
    ],
    "error": [
        "height_segments/height_segment_w_gaussian",
    ],
    "ssh_flag": [
        "height_segments/height_segment_ssh_flag",
    ],
    "h_norm": [
        "height_segments/height_segment_htcorr_skew",
    ],
    "h_ref": [
        "geophysical/height_segment_lpe",
    ],
}


def _find_fb_root(h5, beam):
    """Find the freeboard segment root path for a beam.

    Returns the full path prefix (e.g., 'gt1l/freeboard_segment') or None.
    """
    for root in FB_ROOTS:
        path = f"{beam}/{root}"
        if path in h5:
            return path
    return None


def _try_read(h5, fb_root, sub_paths):
    """Try to read a dataset from multiple sub-path variants under fb_root.

    Returns numpy array or None if all paths fail.
    """
    for sub_path in sub_paths:
        full_path = f"{fb_root}/{sub_path}"
        if full_path in h5:
            return h5[full_path][:]
    return None


def extract_granule(h5_path, output_dir, beams=None):
    """Extract all beams from a single ATL10 HDF5 granule.

    Produces one CSV per granule (all beams combined, distinguished by
    'beam' column) matching the csv_Iqrah/ convention.

    Returns path to output CSV or None on failure.
    """
    import numpy as np
    import pandas as pd

    beams = beams or ALL_BEAMS
    granule_name = Path(h5_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import h5py
    except ImportError:
        logger.error("h5py is required: pip install h5py")
        return None

    all_rows = []

    with h5py.File(h5_path, "r") as h5:
        for beam in beams:
            if beam not in h5:
                logger.debug("  Beam %s not in file — skipping", beam)
                continue

            fb_root = _find_fb_root(h5, beam)
            if fb_root is None:
                logger.warning("  %s: no freeboard_segment group found — skipping", beam)
                continue

            # Read required fields
            freeboard = _try_read(h5, fb_root, ATL10_PATHS["freeboard"])
            lat = _try_read(h5, fb_root, ATL10_PATHS["lat"])
            lon = _try_read(h5, fb_root, ATL10_PATHS["lon"])

            if freeboard is None or lat is None or lon is None:
                logger.warning("  %s: missing freeboard/lat/lon — skipping", beam)
                continue

            n_segs = len(freeboard)
            logger.info("  %s: %d segments (root: %s)", beam, n_segs, fb_root)

            # Read optional fields
            delta_time = _try_read(h5, fb_root, ATL10_PATHS["delta_time"])
            fb_confidence = _try_read(h5, fb_root, ATL10_PATHS["fb_confidence"])
            fb_quality = _try_read(h5, fb_root, ATL10_PATHS["fb_quality"])
            x = _try_read(h5, fb_root, ATL10_PATHS["x"])
            height = _try_read(h5, fb_root, ATL10_PATHS["height"])
            label = _try_read(h5, fb_root, ATL10_PATHS["label"])
            seg_id = _try_read(h5, fb_root, ATL10_PATHS["seg_id"])
            error = _try_read(h5, fb_root, ATL10_PATHS["error"])
            ssh_flag = _try_read(h5, fb_root, ATL10_PATHS["ssh_flag"])
            h_norm = _try_read(h5, fb_root, ATL10_PATHS["h_norm"])
            h_ref = _try_read(h5, fb_root, ATL10_PATHS["h_ref"])

            # Parse timestamps from delta_time
            if delta_time is not None and len(delta_time) > 0:
                days = np.zeros(n_segs, dtype=int)
                hours = np.zeros(n_segs, dtype=int)
                minutes = np.zeros(n_segs, dtype=int)
                seconds = np.zeros(n_segs, dtype=float)
                for i in range(n_segs):
                    ts = ATLAS_EPOCH + timedelta(seconds=float(delta_time[i]))
                    days[i] = ts.day
                    hours[i] = ts.hour
                    minutes[i] = ts.minute
                    seconds[i] = ts.second + ts.microsecond / 1e6
            else:
                days = np.zeros(n_segs, dtype=int)
                hours = np.zeros(n_segs, dtype=int)
                minutes = np.zeros(n_segs, dtype=int)
                seconds = np.zeros(n_segs, dtype=float)

            # Handle length mismatches for height_segments sub-group
            # (height_segments may have different length than freeboard_segment)
            def _align(arr, target_len):
                if arr is None:
                    return np.full(target_len, np.nan)
                if len(arr) == target_len:
                    return arr
                if len(arr) > target_len:
                    return arr[:target_len]
                # Pad with NaN
                padded = np.full(target_len, np.nan)
                padded[:len(arr)] = arr
                return padded

            beam_df = pd.DataFrame({
                "beam": beam,
                "lat": lat,
                "lon": lon,
                "x": x if x is not None else np.nan,
                "seg_id": _align(seg_id, n_segs) if seg_id is not None else np.arange(n_segs),
                "height": _align(height, n_segs),
                "label": _align(label, n_segs),
                "h_ref": _align(h_ref, n_segs),
                "freeboard": freeboard,
                "fb_confidence": fb_confidence if fb_confidence is not None else np.nan,
                "fb_quality": fb_quality if fb_quality is not None else np.nan,
                "h_norm": _align(h_norm, n_segs),
                "error": _align(error, n_segs),
                "ssh_flag": _align(ssh_flag, n_segs),
                "day": days,
                "hour": hours,
                "minute": minutes,
                "second": seconds,
            })
            all_rows.append(beam_df)

    if not all_rows:
        logger.warning("  No beam data extracted from %s", granule_name)
        return None

    df = pd.concat(all_rows, ignore_index=True)

    # Column order
    col_order = [
        "beam", "lat", "lon", "x", "seg_id", "height", "label", "h_ref",
        "freeboard", "fb_confidence", "fb_quality", "h_norm", "error",
        "ssh_flag", "day", "hour", "minute", "second",
    ]
    df = df[col_order]

    out_path = output_dir / f"{granule_name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("  Wrote %d rows → %s", len(df), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract ATL10 HDF5 granules into CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all granules in a directory
  %(prog)s --input-dir ./data/atl10/ --output-dir ./data/atl10_csv/

  # Extract specific files
  %(prog)s --input data/atl10/ATL10_20191103*.h5 --output-dir ./data/atl10_csv/

  # Use only strong beams
  %(prog)s --input-dir ./data/atl10/ --output-dir ./data/atl10_csv/ --beams gt1l,gt2l,gt3l
        """,
    )

    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory containing ATL10 HDF5 files",
    )
    parser.add_argument(
        "--input", type=str, nargs="*", default=None,
        help="Specific ATL10 HDF5 file(s) (supports glob patterns)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--beams", type=str, default=",".join(ALL_BEAMS),
        help="Comma-separated beam names (default: all 6 beams)",
    )

    args = parser.parse_args()

    # Collect input files
    h5_files = []
    if args.input:
        for pattern in args.input:
            h5_files.extend(glob.glob(pattern))
    if args.input_dir:
        h5_files.extend(sorted(glob.glob(os.path.join(args.input_dir, "*.h5"))))

    h5_files = sorted(set(h5_files))
    if not h5_files:
        logger.error("No HDF5 files found")
        sys.exit(1)

    beams = [b.strip() for b in args.beams.split(",")]
    logger.info("Processing %d HDF5 files, beams: %s", len(h5_files), beams)

    all_outputs = []
    for i, h5_path in enumerate(h5_files):
        logger.info("[%d/%d] %s", i + 1, len(h5_files), Path(h5_path).name)
        try:
            out = extract_granule(h5_path, args.output_dir, beams=beams)
            if out:
                all_outputs.append(out)
        except Exception as e:
            logger.error("  Failed: %s", e)
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("ATL10 EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Granules processed: {len(h5_files)}")
    print(f"CSV files created:  {len(all_outputs)}")
    print(f"Output directory:   {args.output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
