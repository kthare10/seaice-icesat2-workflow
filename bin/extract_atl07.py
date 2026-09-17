#!/usr/bin/env python3

"""
Extract ICESat-2 ATL07 sea ice height data from raw HDF5 granules into CSV files.

Reads raw ATL07 HDF5 files downloaded from NASA Earthdata, extracts segment-level
sea ice height data from all six beams, computes freeboard as max(0, height - h_ref),
and parses timestamps from delta_time + ATLAS epoch.

Output CSVs match the column format expected by validate_freeboard.py and
paper_figures.py (columns: beam, lat, lon, x, seg_id, height, h_ref, label,
error, quality, confidence, ssh_flag, mss, dac, geoid, tide, freeboard,
day, hour, minute, second).

Usage:
    python bin/extract_atl07.py --input-dir ./data/atl07/ --output-dir ./data/atl07_csv/
    python bin/extract_atl07.py --input data/atl07/ATL07*.h5 --output-dir ./data/atl07_csv/
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

# HDF5 dataset paths for ATL07 (beam-relative).
# Keys: (output column name, list of path variants to try).
# Variants handle path changes across ATL07 versions (v2–v7).
ATL07_PATHS = {
    "lat": [
        "{beam}/sea_ice_segments/latitude",
    ],
    "lon": [
        "{beam}/sea_ice_segments/longitude",
    ],
    "delta_time": [
        "{beam}/sea_ice_segments/delta_time",
    ],
    "height": [
        "{beam}/sea_ice_segments/heights/height_segment_height",
    ],
    "label": [
        "{beam}/sea_ice_segments/heights/height_segment_type",
    ],
    "ssh_flag": [
        "{beam}/sea_ice_segments/heights/height_segment_ssh_flag",
    ],
    "quality": [
        "{beam}/sea_ice_segments/heights/height_segment_quality",
    ],
    "confidence": [
        "{beam}/sea_ice_segments/heights/height_segment_confidence",
    ],
    "error": [
        "{beam}/sea_ice_segments/heights/height_segment_w_gaussian",
    ],
    "x": [
        "{beam}/sea_ice_segments/geolocation/seg_dist_x",
    ],
    "seg_id": [
        "{beam}/sea_ice_segments/geolocation/segment_id_beg",
        "{beam}/sea_ice_segments/geolocation/height_segment_id",
    ],
    "h_ref": [
        "{beam}/sea_ice_segments/geophysical/height_segment_lpe",
    ],
    "mss": [
        "{beam}/sea_ice_segments/geophysical/height_segment_mss",
    ],
    "dac": [
        "{beam}/sea_ice_segments/geophysical/height_segment_dac",
    ],
    "geoid": [
        "{beam}/sea_ice_segments/geophysical/height_segment_geoid",
    ],
    "tide": [
        "{beam}/sea_ice_segments/geophysical/height_segment_ocean_tide",
    ],
}


def _try_read(h5, paths, beam):
    """Try to read a dataset from multiple path variants.

    Returns numpy array or None if all paths fail.
    """
    for path_template in paths:
        path = path_template.format(beam=beam)
        if path in h5:
            return h5[path][:]
    return None


def extract_granule(h5_path, output_dir, beams=None):
    """Extract all beams from a single ATL07 HDF5 granule.

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
            # Check if beam exists at all
            if beam not in h5:
                logger.debug("  Beam %s not in file — skipping", beam)
                continue

            # Read required fields
            lat = _try_read(h5, ATL07_PATHS["lat"], beam)
            lon = _try_read(h5, ATL07_PATHS["lon"], beam)
            height = _try_read(h5, ATL07_PATHS["height"], beam)

            if lat is None or lon is None or height is None:
                logger.warning("  %s: missing lat/lon/height — skipping", beam)
                continue

            n_segs = len(height)
            logger.info("  %s: %d segments", beam, n_segs)

            # Read optional fields
            delta_time = _try_read(h5, ATL07_PATHS["delta_time"], beam)
            label = _try_read(h5, ATL07_PATHS["label"], beam)
            ssh_flag = _try_read(h5, ATL07_PATHS["ssh_flag"], beam)
            quality = _try_read(h5, ATL07_PATHS["quality"], beam)
            confidence = _try_read(h5, ATL07_PATHS["confidence"], beam)
            error = _try_read(h5, ATL07_PATHS["error"], beam)
            x = _try_read(h5, ATL07_PATHS["x"], beam)
            seg_id = _try_read(h5, ATL07_PATHS["seg_id"], beam)
            h_ref = _try_read(h5, ATL07_PATHS["h_ref"], beam)
            mss = _try_read(h5, ATL07_PATHS["mss"], beam)
            dac = _try_read(h5, ATL07_PATHS["dac"], beam)
            geoid = _try_read(h5, ATL07_PATHS["geoid"], beam)
            tide = _try_read(h5, ATL07_PATHS["tide"], beam)

            # Compute freeboard
            if h_ref is not None:
                freeboard = np.maximum(0.0, height - h_ref)
            else:
                freeboard = np.full(n_segs, np.nan)

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

            # Build rows
            beam_df = pd.DataFrame({
                "beam": beam,
                "lat": lat,
                "lon": lon,
                "x": x if x is not None else np.nan,
                "seg_id": seg_id if seg_id is not None else np.arange(n_segs),
                "height": height,
                "h_ref": h_ref if h_ref is not None else np.nan,
                "label": label if label is not None else np.nan,
                "error": error if error is not None else np.nan,
                "quality": quality if quality is not None else np.nan,
                "confidence": confidence if confidence is not None else np.nan,
                "ssh_flag": ssh_flag if ssh_flag is not None else np.nan,
                "mss": mss if mss is not None else np.nan,
                "dac": dac if dac is not None else np.nan,
                "geoid": geoid if geoid is not None else np.nan,
                "tide": tide if tide is not None else np.nan,
                "freeboard": freeboard,
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
        "beam", "lat", "lon", "x", "seg_id", "height", "h_ref", "label",
        "error", "quality", "confidence", "ssh_flag", "mss", "dac", "geoid",
        "tide", "freeboard", "day", "hour", "minute", "second",
    ]
    df = df[col_order]

    out_path = output_dir / f"{granule_name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("  Wrote %d rows → %s", len(df), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract ATL07 HDF5 granules into CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all granules in a directory
  %(prog)s --input-dir ./data/atl07/ --output-dir ./data/atl07_csv/

  # Extract specific files
  %(prog)s --input data/atl07/ATL07_20191103*.h5 --output-dir ./data/atl07_csv/

  # Use only strong beams
  %(prog)s --input-dir ./data/atl07/ --output-dir ./data/atl07_csv/ --beams gt1l,gt2l,gt3l
        """,
    )

    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory containing ATL07 HDF5 files",
    )
    parser.add_argument(
        "--input", type=str, nargs="*", default=None,
        help="Specific ATL07 HDF5 file(s) (supports glob patterns)",
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
    print("ATL07 EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Granules processed: {len(h5_files)}")
    print(f"CSV files created:  {len(all_outputs)}")
    print(f"Output directory:   {args.output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
