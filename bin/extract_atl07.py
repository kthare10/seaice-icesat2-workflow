#!/usr/bin/env python3

"""
Extract ICESat-2 ATL07 sea ice height data from raw HDF5 granules into CSV files.

Reads raw ATL07 HDF5 files downloaded from NASA Earthdata, extracts segment-level
sea ice height data from all six beams, and parses timestamps from delta_time + ATLAS
epoch. NASA fill values (3.4028235e38) become NaN.

ATL07 carries no per-segment sea surface; NASA's lives in ATL10. When the matching
ATL10 granule is given with ``--atl10``, ``h_ref`` is the ATL10 reference surface
(``height_segment_height - beam_fb_height``, joined on ``height_segment_id``) linearly
interpolated along track onto every ATL07 segment, ``freeboard = height - h_ref``, and
``freeboard_10`` is ATL10's ``beam_fb_height`` on the segments ATL10 reports. Without
ATL10 those three columns are NaN.

Output CSVs match the column format expected by validate_freeboard.py and
paper_figures.py (columns: beam, lat, lon, x, seg_id, height, h_ref, label,
error, quality, confidence, ssh_flag, mss, dac, geoid, tide, freeboard,
freeboard_10, day, hour, minute, second).

Usage:
    python bin/extract_atl07.py --input-dir ./data/atl07/ --output-dir ./data/atl07_csv/
    python bin/extract_atl07.py --input data/atl07/ATL07*.h5 --output-dir ./data/atl07_csv/ \
        --atl10 data/atl10/ATL10*.h5
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
        "{beam}/sea_ice_segments/seg_dist_x",
        "{beam}/sea_ice_segments/geolocation/seg_dist_x",
    ],
    "seg_id": [
        "{beam}/sea_ice_segments/height_segment_id",
        "{beam}/sea_ice_segments/geolocation/height_segment_id",
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
        "{beam}/sea_ice_segments/geophysical/height_segment_ocean",
        "{beam}/sea_ice_segments/geophysical/height_segment_ocean_tide",
    ],
}

# NASA ICESat-2 float fill value (3.4028235e38); anything this large is missing data.
FILL_THRESHOLD = 1e38


def _try_read(h5, paths, beam):
    """Try to read a dataset from multiple path variants.

    Returns numpy array or None if all paths fail.
    """
    for path_template in paths:
        path = path_template.format(beam=beam)
        if path in h5:
            return _mask_fill(h5[path][:])
    return None


def _mask_fill(arr):
    """Return float arrays with NASA fill values replaced by NaN; others unchanged."""
    import numpy as np

    if arr.dtype.kind != "f":
        return arr
    arr = arr.astype(np.float64)
    arr[np.abs(arr) > FILL_THRESHOLD] = np.nan
    return arr


def read_atl10_surface(atl10_path, beam):
    """Read ATL10's reference surface and freeboard for one beam.

    Returns (height_segment_id, h_ref, freeboard) arrays, or None if the beam or a
    dataset is missing. ``h_ref`` is ``height_segment_height - beam_fb_height``: the
    surface ATL10's freeboard is measured from.
    """
    import h5py

    root = f"{beam}/freeboard_segment"
    with h5py.File(atl10_path, "r") as h5:
        paths = [f"{root}/height_segment_id", f"{root}/heights/height_segment_height",
                 f"{root}/beam_fb_height"]
        if not all(p in h5 for p in paths):
            return None
        seg_id, height, fb = (_mask_fill(h5[p][:]) for p in paths)
    return seg_id, height - fb, fb


def attach_atl10_surface(seg_id, x, height, atl10):
    """Map ATL10's surface onto ATL07 segments.

    Returns (h_ref, freeboard, freeboard_10). ``h_ref`` is joined on height_segment_id,
    then linearly interpolated along ``x`` to segments ATL10 omits (inside ATL10's
    coverage only; no extrapolation). ``freeboard_10`` is ATL10's own value where it
    exists.
    """
    import numpy as np

    n = len(height)
    nan = np.full(n, np.nan)
    if atl10 is None:
        return nan, nan.copy(), nan.copy()
    id10, h_ref10, fb10 = atl10
    order = np.argsort(id10)
    id10, h_ref10, fb10 = id10[order], h_ref10[order], fb10[order]
    pos = np.clip(np.searchsorted(id10, seg_id), 0, len(id10) - 1)
    hit = id10[pos] == seg_id

    h_ref = nan.copy()
    h_ref[hit] = h_ref10[pos[hit]]
    freeboard_10 = nan.copy()
    freeboard_10[hit] = fb10[pos[hit]]

    known = np.isfinite(h_ref) & np.isfinite(x)
    if known.sum() >= 2:
        xs = x[known]
        srt = np.argsort(xs)
        fill = ~np.isfinite(h_ref) & np.isfinite(x) & (x >= xs.min()) & (x <= xs.max())
        h_ref[fill] = np.interp(x[fill], xs[srt], h_ref[known][srt])
    return h_ref, height - h_ref, freeboard_10


def find_matching_atl10(h5_path, atl10_files):
    """Return the ATL10 granule with the same granule id (timestamp, RGT, version)."""
    key = Path(h5_path).stem.split("_", 1)[-1]
    for path in atl10_files or []:
        if Path(path).stem.split("_", 1)[-1] == key:
            return path
    return None


def extract_granule(h5_path, output_dir, beams=None, atl10_path=None):
    """Extract all beams from a single ATL07 HDF5 granule.

    Produces one CSV per granule (all beams combined, distinguished by
    'beam' column) matching the csv_Iqrah/ convention.

    ``atl10_path`` (optional) is the matching ATL10 granule supplying ``h_ref``,
    ``freeboard`` and ``freeboard_10``.

    Returns path to output CSV or None on failure.
    """
    import numpy as np
    import pandas as pd

    beams = beams or ALL_BEAMS
    if atl10_path:
        logger.info("  Sea surface from ATL10: %s", Path(atl10_path).name)
    else:
        logger.warning("  No matching ATL10 granule: h_ref/freeboard/freeboard_10 will be NaN")
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
            mss = _try_read(h5, ATL07_PATHS["mss"], beam)
            dac = _try_read(h5, ATL07_PATHS["dac"], beam)
            geoid = _try_read(h5, ATL07_PATHS["geoid"], beam)
            tide = _try_read(h5, ATL07_PATHS["tide"], beam)

            if seg_id is None:
                logger.warning("  %s: no height_segment_id — cannot join ATL10", beam)
            atl10 = (read_atl10_surface(atl10_path, beam)
                     if atl10_path and seg_id is not None else None)
            if atl10_path and atl10 is None:
                logger.warning("  %s: ATL10 surface unavailable for this beam", beam)
            h_ref, freeboard, freeboard_10 = attach_atl10_surface(
                seg_id if seg_id is not None else np.arange(n_segs),
                x if x is not None else np.full(n_segs, np.nan), height, atl10)
            logger.info("  %s: %d fill heights masked, h_ref on %d segments "
                        "(%d from ATL10 directly)", beam, int(np.isnan(height).sum()),
                        int(np.isfinite(h_ref).sum()), int(np.isfinite(freeboard_10).sum()))

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
                "h_ref": h_ref,
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
                "freeboard_10": freeboard_10,
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
        "tide", "freeboard", "freeboard_10", "day", "hour", "minute", "second",
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

  # Take the sea surface from the matching ATL10 granules
  %(prog)s --input-dir ./data/atl07/ --output-dir ./data/atl07_csv/ --atl10 data/atl10/*.h5

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
    parser.add_argument(
        "--atl10", type=str, nargs="*", default=None,
        help="ATL10 HDF5 granule(s); each ATL07 granule uses the one with its granule id "
             "for h_ref/freeboard/freeboard_10",
    )

    args = parser.parse_args()
    atl10_files = []
    for pattern in args.atl10 or []:
        atl10_files.extend(glob.glob(pattern))

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
            out = extract_granule(h5_path, args.output_dir, beams=beams,
                                  atl10_path=find_matching_atl10(h5_path, atl10_files))
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
