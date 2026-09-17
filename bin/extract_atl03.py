#!/usr/bin/env python3

"""
Extract ICESat-2 ATL03 photon data from raw HDF5 granules into per-track CSVs.

Reads raw ATL03 HDF5 files downloaded from NASA Earthdata, extracts photon-level
data from the strong beams (chosen from sc_orient), filters for high-confidence sea-ice signal photons,
resamples to 2-meter along-track segments, computes statistical features, and
applies geophysical corrections (geoid, DAC, tide, mean sea surface).

Output CSVs match the column format expected by the auto-labeling and workflow
pipeline (41 columns including Ori_Id, timestamps, coordinates, height statistics,
geophysical corrections, and derived h_cor_mean/h_cor_med).

Usage:
    python bin/extract_atl03.py --input-dir ./data/atl03/ --output-dir ./data/extracted/
    python bin/extract_atl03.py --input data/atl03/ATL03_2019*.h5 --output-dir ./data/extracted/
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

# Strong/weak beam assignment depends on spacecraft orientation (orbit_info/sc_orient), see the
# ATL03 ATBD / read-ICESat-2 docs: in the forward orientation the weak beams lead and a weak beam
# is on the left edge (gt1l), so the strong beams are gt1r/gt2r/gt3r; backward is the reverse.
#   0 = backward -> strong beams gt1l, gt2l, gt3l
#   1 = forward  -> strong beams gt1r, gt2r, gt3r (the paper's Nov 2019 tracks use gt*r)
STRONG_BEAMS_BY_ORIENT = {0: ["gt1l", "gt2l", "gt3l"], 1: ["gt1r", "gt2r", "gt3r"]}
STRONG_BEAMS = STRONG_BEAMS_BY_ORIENT[1]  # fallback when sc_orient is unavailable


def strong_beams_for(h5):
    """Return the strong beams of an open ATL03 HDF5 file based on orbit_info/sc_orient."""
    try:
        orient = int(h5["orbit_info/sc_orient"][0])
    except Exception:
        logger.warning("  orbit_info/sc_orient not found — assuming forward orientation (%s)",
                       STRONG_BEAMS)
        return STRONG_BEAMS
    if orient not in STRONG_BEAMS_BY_ORIENT:
        logger.warning("  sc_orient=%d (transition) — using %s", orient, STRONG_BEAMS)
        return STRONG_BEAMS
    beams = STRONG_BEAMS_BY_ORIENT[orient]
    logger.info("  sc_orient=%d -> strong beams %s", orient, beams)
    return beams
BIN_SIZE_M = 2.0
MIN_SIGNAL_CONF = 3  # High-confidence photons on sea-ice surface type

# ATLAS SDP GPS epoch: 2018-01-01 00:00:00 UTC (GPS seconds)
ATLAS_EPOCH = datetime(2018, 1, 1, 0, 0, 0)


# ------------------------------------------------------------------
# Photon extraction
# ------------------------------------------------------------------


def _read_beam(h5, beam):
    """Read photon-level arrays from a single beam.

    Returns dict of numpy arrays or None if the beam is missing/empty.
    """
    import numpy as np

    heights_path = f"{beam}/heights"
    if heights_path not in h5:
        # Fallback: datasets may be directly under the beam group
        if beam not in h5:
            return None
        grp = h5[beam]
        if "h_ph" not in grp:
            return None
        heights_path = beam

    grp = h5[heights_path]

    required = ["h_ph", "lat_ph", "lon_ph", "signal_conf_ph"]
    for key in required:
        if key not in grp:
            logger.warning("  Missing %s/%s — skipping beam", heights_path, key)
            return None

    h_ph = grp["h_ph"][:]
    lat_ph = grp["lat_ph"][:]
    lon_ph = grp["lon_ph"][:]
    signal_conf = grp["signal_conf_ph"][:]
    delta_time = grp["delta_time"][:] if "delta_time" in grp else np.zeros_like(h_ph)

    # Along-track distance
    if "dist_ph_along" in grp:
        dist_ph_along = grp["dist_ph_along"][:]
    else:
        dist_ph_along = None  # Compute later

    # Signal confidence may be (N,) or (N, 5). Use sea-ice column (index 2).
    if signal_conf.ndim == 2:
        signal_conf = signal_conf[:, 2]

    return {
        "h_ph": h_ph,
        "lat_ph": lat_ph,
        "lon_ph": lon_ph,
        "signal_conf": signal_conf,
        "delta_time": delta_time,
        "dist_ph_along": dist_ph_along,
    }


def _read_geolocation(h5, beam):
    """Read segment-level geolocation data for x_atc computation."""
    import numpy as np

    geo_path = f"{beam}/geolocation"
    if geo_path not in h5:
        return None

    geo = h5[geo_path]
    result = {}
    for key in ["segment_dist_x", "segment_id", "segment_ph_cnt",
                "reference_photon_lat", "reference_photon_lon"]:
        if key in geo:
            result[key] = geo[key][:]

    return result if result else None


def _read_geophys_corr(h5, beam):
    """Read segment-level geophysical corrections."""
    corr_path = f"{beam}/geophys_corr"
    if corr_path not in h5:
        return None

    corr = h5[corr_path]
    result = {}
    for key, alias in [("geoid", "geoid"), ("dac", "dac"),
                       ("tide_ocean", "tide"), ("dem_h", "dem_h"),
                       ("geoid_free2mean", "geoid_free2mean")]:
        if key in corr:
            result[alias] = corr[key][:]

    return result if result else None


def _read_mss(h5, beam):
    """Read mean sea surface height (per-segment)."""
    import numpy as np

    # MSS can be in different locations depending on ATL03 version
    for path in [f"{beam}/geophys_corr/geoid",
                 f"{beam}/geolocation/ref_elev"]:
        if path in h5:
            pass  # geoid is already handled

    # Try ancillary_data or compute from geoid + geoid_free2mean
    return None


# ------------------------------------------------------------------
# Segment binning
# ------------------------------------------------------------------


def _compute_along_track(lat_ph, lon_ph, geoloc):
    """Compute along-track distance for each photon.

    Uses segment_dist_x from geolocation if available, otherwise
    approximates from lat/lon.
    """
    import numpy as np

    if geoloc is not None and "segment_dist_x" in geoloc and "segment_ph_cnt" in geoloc:
        seg_dist = geoloc["segment_dist_x"]
        seg_cnt = geoloc["segment_ph_cnt"]

        # Expand segment-level x_atc to photon-level
        total_ph = int(np.sum(seg_cnt))
        if total_ph == len(lat_ph):
            x_atc = np.empty(total_ph)
            idx = 0
            for s_dist, s_cnt in zip(seg_dist, seg_cnt):
                cnt = int(s_cnt)
                if cnt > 0:
                    # Spread photons evenly within the segment
                    x_atc[idx:idx + cnt] = s_dist + np.linspace(0, BIN_SIZE_M, cnt, endpoint=False)
                    idx += cnt
            return x_atc

    # Fallback: approximate from lat/lon using haversine increments
    dlat = np.diff(lat_ph, prepend=lat_ph[0])
    dlon = np.diff(lon_ph, prepend=lon_ph[0])
    step_dist = np.sqrt(
        (dlat * 111_000) ** 2 + (dlon * 111_000 * np.cos(np.radians(lat_ph))) ** 2
    )
    return np.cumsum(step_dist)


def _interpolate_corrections(seg_dist, corr_values, photon_x_atc):
    """Interpolate segment-level corrections to photon level."""
    import numpy as np

    if seg_dist is None or corr_values is None:
        return np.zeros(len(photon_x_atc))

    # Handle length mismatches and NaN
    mask = np.isfinite(corr_values) & np.isfinite(seg_dist)
    if np.sum(mask) < 2:
        return np.full(len(photon_x_atc), np.nanmean(corr_values) if len(corr_values) > 0 else 0.0)

    return np.interp(photon_x_atc, seg_dist[mask], corr_values[mask])


def _estimate_background(h_values):
    """Estimate background count and rate for a segment using MAD."""
    import numpy as np

    if len(h_values) < 2:
        return 0.0, 0.0

    h_range = np.ptp(h_values)
    if h_range == 0:
        return 0.0, 0.0

    median_h = np.median(h_values)
    mad = np.median(np.abs(h_values - median_h))
    sigma = 1.4826 * mad

    if sigma == 0:
        return 0.0, 0.0

    bg_mask = np.abs(h_values - median_h) > 2 * sigma
    bg_count = float(np.sum(bg_mask))
    bg_rate = bg_count / h_range

    return bg_count, bg_rate


def bin_photons_to_segments(photon_df, bin_size=BIN_SIZE_M):
    """Bin photon data into along-track segments and compute statistics.

    Returns a list of segment dicts.
    """
    import numpy as np

    x_atc = photon_df["x_atc"]
    bins = np.arange(x_atc.min(), x_atc.max() + bin_size, bin_size)
    bin_idx = np.digitize(x_atc, bins) - 1

    segments = []
    for bi in np.unique(bin_idx):
        if bi < 0 or bi >= len(bins) - 1:
            continue
        mask = bin_idx == bi
        group = {k: v[mask] for k, v in photon_df.items()}

        h_raw = group["h_ph"]
        N = len(h_raw)
        if N == 0:
            continue

        # First-photon bias correction
        if N > 5:
            sorted_h = np.sort(h_raw)
            fpb_corr = float(sorted_h[0])
            h_values = sorted_h[1:]  # Remove lowest
        else:
            fpb_corr = 0.0
            h_values = h_raw

        height_mean = float(np.mean(h_values))
        height_med = float(np.median(h_values))
        height_sd = float(np.std(h_values)) if len(h_values) > 1 else 0.0

        # Percentile metrics
        if N > 1:
            ranks = np.arange(1, N + 1, dtype=float)
            pcnt = ranks / N * N  # Photon count-based percentile
            pcnt_mean = float(np.mean(pcnt))
            pcnt_sd = float(np.std(pcnt))
            pcnt_med = float(np.median(pcnt))
            # Height percentile
            sorted_h_raw = np.sort(h_raw)
            pcnth_mean = float(np.mean(sorted_h_raw[:max(1, N // 2)]))
            pcnth_sd = float(np.std(sorted_h_raw[:max(1, N // 2)]))
            pcnth_med = float(np.median(sorted_h_raw[:max(1, N // 2)]))
        else:
            pcnt_mean = pcnt_sd = pcnt_med = float(N)
            pcnth_mean = pcnth_sd = pcnth_med = float(h_raw[0])

        bcnt, brate = _estimate_background(h_raw)

        # Geophysical corrections (interpolated to segment center)
        seg_x_atc = float(bins[bi] + bin_size / 2)
        geoid = float(np.mean(group.get("geoid", np.array([0.0]))))
        dac = float(np.mean(group.get("dac", np.array([0.0]))))
        tide = float(np.mean(group.get("tide", np.array([0.0]))))
        mss = float(np.mean(group.get("mss", np.array([0.0]))))

        # Corrected heights: relative to mean sea surface, with FPB correction
        # h_cor = height - fpb_corr - mss  (standard ATL03 processing)
        # But fpb_corr in the sample CSV is the correction offset, not the removed value
        fpb_offset = height_mean - (float(np.mean(h_raw)) if N > 5 else 0.0)
        h_cor_mean = height_mean - mss
        h_cor_med = height_med - mss

        # Satellite geometry (constant per granule — filled later)
        # Timestamp from delta_time
        dt = float(np.mean(group.get("delta_time", np.array([0.0]))))

        segments.append({
            "Ori_Id": int(bins[bi] / bin_size),
            "delta_time": dt,
            "lat": float(np.mean(group["lat_ph"])),
            "lon": float(np.mean(group["lon_ph"])),
            "dac": dac,
            "geoid": geoid,
            "tide": tide,
            "N": N,
            "height_mean": height_mean,
            "pcnt_mean": pcnt_mean,
            "pcnth_mean": pcnth_mean,
            "bcnt_mean": bcnt,
            "brate_mean": brate,
            "height_sd": height_sd,
            "pcnt_sd": pcnt_sd,
            "pcnth_sd": pcnth_sd,
            "bcnt_sd": 0.0,
            "brate_sd": 0.0,
            "height_med": height_med,
            "pcnt_med": pcnt_med,
            "pcnth_med": pcnth_med,
            "bcnt_med": bcnt,
            "brate_med": brate,
            "fpb_corr": abs(fpb_offset),
            "mss": mss,
            "h_cor_mean": h_cor_mean,
            "h_cor_med": h_cor_med,
            "x_atc": seg_x_atc,
        })

    return segments


# ------------------------------------------------------------------
# Main extraction
# ------------------------------------------------------------------


def extract_granule(h5_path, output_dir, beams=None):
    """Extract all beams from a single ATL03 HDF5 granule.

    Produces one CSV per beam: <output_dir>/<granule_base>_<beam>.csv

    Returns list of output CSV paths.
    """
    import h5py
    import numpy as np
    import pandas as pd

    granule_name = Path(h5_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []

    with h5py.File(h5_path, "r") as h5:
        if not beams:
            beams = strong_beams_for(h5)
        # Determine top-level structure
        # Raw ATL03: beams at top level (gt1l, gt2l, ...)
        # Merged: granule_XXXX/beam (from download_atl03.py merge)
        top_keys = list(h5.keys())
        is_merged = any(k.startswith("granule_") for k in top_keys)

        if is_merged:
            granule_beam_pairs = []
            for gk in sorted(k for k in top_keys if k.startswith("granule_")):
                for beam in beams:
                    if f"{gk}/{beam}" in h5:
                        granule_beam_pairs.append((gk, beam))
        else:
            granule_beam_pairs = [("", beam) for beam in beams if beam in h5]

        for granule_key, beam in granule_beam_pairs:
            prefix = f"{granule_key}/" if granule_key else ""
            beam_full = f"{prefix}{beam}"
            logger.info("  Processing %s/%s", granule_name, beam_full)

            # Read photon data
            if granule_key:
                # Merged format: datasets directly under granule/beam
                photon_data = _read_beam_flat(h5, f"{granule_key}/{beam}")
            else:
                # Raw format: datasets under beam/heights/
                photon_data = _read_beam(h5, beam)

            if photon_data is None:
                logger.warning("    No photon data — skipping")
                continue

            h_ph = photon_data["h_ph"]
            lat_ph = photon_data["lat_ph"]
            lon_ph = photon_data["lon_ph"]
            signal_conf = photon_data["signal_conf"]
            delta_time = photon_data["delta_time"]

            logger.info("    Raw photons: %d", len(h_ph))

            # Filter high-confidence sea-ice signal photons
            mask = signal_conf >= MIN_SIGNAL_CONF
            if np.sum(mask) == 0:
                logger.warning("    No signal photons after filtering — skipping")
                continue

            h_ph = h_ph[mask]
            lat_ph = lat_ph[mask]
            lon_ph = lon_ph[mask]
            delta_time = delta_time[mask]

            dist_along = photon_data.get("dist_ph_along")
            if dist_along is not None:
                dist_along = dist_along[mask]

            logger.info("    Signal photons (conf>=%d): %d", MIN_SIGNAL_CONF, len(h_ph))

            # Read geolocation for x_atc
            geoloc = _read_geolocation(h5, f"{prefix}{beam}" if not granule_key else None)

            # Compute along-track distance
            if dist_along is not None:
                # Use geolocation segment_dist_x as base offset if available
                if geoloc is not None and "segment_dist_x" in geoloc:
                    x_atc = dist_along + geoloc["segment_dist_x"][0]
                else:
                    x_atc = dist_along
            elif geoloc is not None:
                x_atc = _compute_along_track(lat_ph, lon_ph, geoloc)
            else:
                # Pure fallback
                dlat = np.diff(lat_ph, prepend=lat_ph[0])
                dlon = np.diff(lon_ph, prepend=lon_ph[0])
                step = np.sqrt((dlat * 111_000) ** 2 +
                               (dlon * 111_000 * np.cos(np.radians(lat_ph))) ** 2)
                x_atc = np.cumsum(step)

            # Read geophysical corrections and interpolate to photon level
            geophys = _read_geophys_corr(h5, f"{prefix}{beam}" if not granule_key else None)
            seg_dist = geoloc["segment_dist_x"] if geoloc and "segment_dist_x" in geoloc else None

            if geophys:
                geoid_ph = _interpolate_corrections(seg_dist, geophys.get("geoid"), x_atc)
                dac_ph = _interpolate_corrections(seg_dist, geophys.get("dac"), x_atc)
                tide_ph = _interpolate_corrections(seg_dist, geophys.get("tide"), x_atc)
            else:
                geoid_ph = np.zeros(len(h_ph))
                dac_ph = np.zeros(len(h_ph))
                tide_ph = np.zeros(len(h_ph))

            # MSS: approximate from geoid (MSS ≈ geoid for open ocean at segment scale)
            mss_ph = geoid_ph.copy()

            # Build photon dict for binning
            photon_dict = {
                "h_ph": h_ph,
                "lat_ph": lat_ph,
                "lon_ph": lon_ph,
                "delta_time": delta_time,
                "x_atc": x_atc,
                "geoid": geoid_ph,
                "dac": dac_ph,
                "tide": tide_ph,
                "mss": mss_ph,
            }

            # Bin to 2m segments
            segments = bin_photons_to_segments(photon_dict)
            if not segments:
                logger.warning("    No segments produced — skipping")
                continue

            df = pd.DataFrame(segments)

            # Add timestamp columns from delta_time
            if "delta_time" in df.columns and df["delta_time"].abs().sum() > 0:
                ref_dt = df["delta_time"].iloc[0]
                ts = ATLAS_EPOCH + timedelta(seconds=float(ref_dt))
                df["year"] = ts.year
                df["month"] = ts.month
                df["day"] = ts.day
                df["hour"] = ts.hour
                df["minute"] = ts.minute
                df["second"] = ts.second
            else:
                # Parse timestamp from filename: ATL03_YYYYMMDDHHMMSS_...
                ts_str = granule_name.split("_")[1] if "_" in granule_name else ""
                if len(ts_str) >= 14:
                    ts = datetime.strptime(ts_str[:14], "%Y%m%d%H%M%S")
                    df["year"] = ts.year
                    df["month"] = ts.month
                    df["day"] = ts.day
                    df["hour"] = ts.hour
                    df["minute"] = ts.minute
                    df["second"] = ts.second
                else:
                    for col in ["year", "month", "day", "hour", "minute", "second"]:
                        df[col] = 0

            # Projected coordinates (Antarctic Polar Stereographic EPSG:3976)
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3976", always_xy=True)
                df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
            except ImportError:
                # Approximate projection (good enough for pixel lookup)
                df["x"] = df["lon"] * 111_000 * np.cos(np.radians(df["lat"]))
                df["y"] = df["lat"] * 111_000

            # Satellite geometry placeholders (not in standard ATL03 photon data)
            df["s_azi"] = 0.0
            df["s_ele"] = 0.0

            # Geometry column (WKT point in projected coords)
            df["geometry"] = df.apply(
                lambda r: f"POINT ({r['x']} {r['y']})", axis=1
            )

            # Reorder columns to match expected format
            col_order = [
                "Ori_Id", "year", "month", "day", "hour", "minute", "second",
                "lat", "lon", "x", "y", "dac", "geoid", "tide", "s_azi", "s_ele",
                "N", "height_mean", "pcnt_mean", "pcnth_mean", "bcnt_mean", "brate_mean",
                "height_sd", "pcnt_sd", "pcnth_sd", "bcnt_sd", "brate_sd",
                "height_med", "pcnt_med", "pcnth_med", "bcnt_med", "brate_med",
                "fpb_corr", "mss", "h_cor_mean", "h_cor_med", "x_atc", "geometry",
            ]
            for c in col_order:
                if c not in df.columns:
                    df[c] = 0.0
            df = df[col_order]

            # Output filename
            out_name = f"{granule_name}_{beam}.csv"
            out_path = output_dir / out_name
            df.to_csv(out_path, index=True)
            logger.info("    Wrote %d segments → %s", len(df), out_path)
            outputs.append(out_path)

    return outputs


def _read_beam_flat(h5, beam_path):
    """Read photon data from a flat beam group (merged HDF5 format)."""
    import numpy as np

    if beam_path not in h5:
        return None
    grp = h5[beam_path]

    required = ["h_ph", "lat_ph", "lon_ph", "signal_conf_ph"]
    for key in required:
        if key not in grp:
            return None

    signal_conf = grp["signal_conf_ph"][:]
    if signal_conf.ndim == 2:
        signal_conf = signal_conf[:, 2]

    return {
        "h_ph": grp["h_ph"][:],
        "lat_ph": grp["lat_ph"][:],
        "lon_ph": grp["lon_ph"][:],
        "signal_conf": signal_conf,
        "delta_time": grp["delta_time"][:] if "delta_time" in grp else np.zeros_like(grp["h_ph"][:]),
        "dist_ph_along": grp["dist_ph_along"][:] if "dist_ph_along" in grp else None,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract ATL03 HDF5 granules into per-beam CSV segment files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all granules in a directory
  %(prog)s --input-dir ./data/atl03/ --output-dir ./data/extracted/

  # Extract specific files
  %(prog)s --input data/atl03/ATL03_20191103*.h5 --output-dir ./data/extracted/

  # Use only gt1l beam
  %(prog)s --input-dir ./data/atl03/ --output-dir ./data/extracted/ --beams gt1l
        """,
    )

    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory containing ATL03 HDF5 files",
    )
    parser.add_argument(
        "--input", type=str, nargs="*", default=None,
        help="Specific ATL03 HDF5 file(s) (supports glob patterns)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--beams", type=str, default="strong",
        help="Comma-separated beam names, or 'strong' to pick the strong beams from "
             "orbit_info/sc_orient (default: strong)",
    )
    parser.add_argument(
        "--bin-size", type=float, default=BIN_SIZE_M,
        help=f"Along-track bin size in meters (default: {BIN_SIZE_M})",
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

    beams = None if args.beams.strip().lower() == "strong" else [b.strip() for b in args.beams.split(",")]
    logger.info("Processing %d HDF5 files, beams: %s", len(h5_files), beams or "strong (from sc_orient)")

    all_outputs = []
    for i, h5_path in enumerate(h5_files):
        logger.info("[%d/%d] %s", i + 1, len(h5_files), Path(h5_path).name)
        try:
            outputs = extract_granule(h5_path, args.output_dir, beams=beams)
            all_outputs.extend(outputs)
        except Exception as e:
            logger.error("  Failed: %s", e)
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("ATL03 EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Granules processed: {len(h5_files)}")
    print(f"CSV files created:  {len(all_outputs)}")
    print(f"Output directory:   {args.output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
