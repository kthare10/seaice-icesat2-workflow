#!/usr/bin/env python3

"""
Auto-label ATL03 track segments using Sentinel-2 imagery.

Projects ATL03 photon segment coordinates onto co-located Sentinel-2 imagery
and classifies each segment as thick ice (0), thin ice (1), or open water (2)
using HSV color segmentation thresholds.

The classification is purely brightness-based (Value channel in HSV):
  - V >= 205  →  thick ice  (label 0, bright white)
  - 31 <= V < 205  →  thin ice  (label 1, medium brightness)
  - V < 31  →  open water  (label 2, dark)

Input:  Per-beam CSVs from extract_atl03.py (with lat, lon, x, y columns)
Output: Same CSVs with three new columns: pix_x, pix_y, label

Usage:
    python bin/autolabel_s2.py \
        --tracks-dir ./data/extracted/ \
        --sentinel2 ./data/sentinel2/sentinel2_scenes.tar.gz \
        --output-dir ./data/labeled/

    python bin/autolabel_s2.py \
        --tracks-dir ./data/extracted/ \
        --sentinel2-dir ./data/sentinel2/extracted_scenes/ \
        --output-dir ./data/labeled/
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HSV thresholds from parallel_segmentation.py
# OpenCV HSV ranges: H=[0,180], S=[0,255], V=[0,255]
# Classification is primarily driven by V (brightness).
# ---------------------------------------------------------------------------

# (H_min, S_min, V_min), (H_max, S_max, V_max)
HSV_THICK_ICE = ((0, 0, 205), (185, 255, 255))   # label 0 — bright
HSV_THIN_ICE = ((0, 0, 31), (185, 255, 204))      # label 1 — medium
HSV_OPEN_WATER = ((0, 0, 0), (185, 255, 30))       # label 2 — dark


# ---------------------------------------------------------------------------
# Sentinel-2 image handling
# ---------------------------------------------------------------------------


def load_sentinel2_rgb(scene_dir):
    """Load Sentinel-2 bands and compose an RGB image.

    Looks for B04 (Red), B03 (Green), B02 (Blue) GeoTIFFs.
    Returns (rgb_array, transform, crs) or None if bands are missing.
    """
    import rasterio

    scene_dir = Path(scene_dir)
    band_files = {}
    for band in ["B04", "B03", "B02"]:
        candidates = list(scene_dir.glob(f"*{band}*.tif")) + list(scene_dir.glob(f"{band}.tif"))
        if candidates:
            band_files[band] = candidates[0]

    if len(band_files) < 3:
        logger.warning("Missing RGB bands in %s (found: %s)", scene_dir, list(band_files.keys()))
        return None

    # Read bands and stack as RGB
    with rasterio.open(band_files["B04"]) as src:
        red = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        profile = src.profile

    with rasterio.open(band_files["B03"]) as src:
        green = src.read(1).astype(np.float32)

    with rasterio.open(band_files["B02"]) as src:
        blue = src.read(1).astype(np.float32)

    # Normalize to 0-255 uint8 for HSV conversion
    # Sentinel-2 L2A surface reflectance is typically 0-10000
    for arr in [red, green, blue]:
        arr[arr < 0] = 0

    max_val = max(red.max(), green.max(), blue.max())
    if max_val == 0:
        max_val = 1
    scale = 255.0 / max_val

    rgb = np.stack([
        (red * scale).clip(0, 255).astype(np.uint8),
        (green * scale).clip(0, 255).astype(np.uint8),
        (blue * scale).clip(0, 255).astype(np.uint8),
    ], axis=-1)  # (H, W, 3)

    return rgb, transform, crs


def segment_hsv(rgb):
    """Apply HSV color segmentation to an RGB image.

    Returns a label array (H, W) with values:
      0 = thick ice, 1 = thin ice, 2 = open water, 255 = unclassified
    """
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    # Apply masks in priority order (thick ice > thin ice > open water)
    # since V ranges overlap: thick [205,255], thin [31,204], water [0,30]
    mask_water = cv2.inRange(hsv, HSV_OPEN_WATER[0], HSV_OPEN_WATER[1])
    mask_tice = cv2.inRange(hsv, HSV_THIN_ICE[0], HSV_THIN_ICE[1])
    mask_ice = cv2.inRange(hsv, HSV_THICK_ICE[0], HSV_THICK_ICE[1])

    labels[mask_water == 255] = 2
    labels[mask_tice == 255] = 1
    labels[mask_ice == 255] = 0

    return labels


# ---------------------------------------------------------------------------
# Coordinate projection & label lookup
# ---------------------------------------------------------------------------


def project_tracks_to_pixels(lons, lats, transform, crs):
    """Convert lon/lat coordinates to pixel (col, row) in the raster.

    Returns (pix_x, pix_y) arrays.
    """
    from rasterio.transform import rowcol

    # Project lon/lat (EPSG:4326) to the raster's CRS
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
    except ImportError:
        # If pyproj not available, assume raster is in EPSG:4326
        logger.warning("pyproj not available — assuming raster CRS matches EPSG:4326")
        xs, ys = lons, lats

    # Convert projected coordinates to pixel indices
    rows, cols = rowcol(transform, xs, ys)
    return np.array(cols), np.array(rows)


def lookup_labels(label_img, pix_x, pix_y):
    """Look up label values for pixel coordinates, with bounds checking."""
    h, w = label_img.shape
    labels = np.full(len(pix_x), 255, dtype=np.uint8)

    for i in range(len(pix_x)):
        col = int(round(pix_x[i]))
        row = int(round(pix_y[i]))
        if 0 <= row < h and 0 <= col < w:
            labels[i] = label_img[row, col]

    return labels


# ---------------------------------------------------------------------------
# Scene matching
# ---------------------------------------------------------------------------


def find_best_scene(track_lon, track_lat, scenes_info):
    """Find the scene whose footprint best covers the track.

    Returns (scene_dir, rgb, transform, crs) or None.
    """
    from rasterio.transform import rowcol

    track_center_lon = np.mean(track_lon)
    track_center_lat = np.mean(track_lat)

    best = None
    best_coverage = 0

    for scene_dir, rgb, transform, crs in scenes_info:
        pix_x, pix_y = project_tracks_to_pixels(track_lon, track_lat, transform, crs)
        h, w = rgb.shape[:2]
        in_bounds = (pix_x >= 0) & (pix_x < w) & (pix_y >= 0) & (pix_y < h)
        coverage = np.mean(in_bounds)

        if coverage > best_coverage:
            best_coverage = coverage
            best = (scene_dir, rgb, transform, crs)

    if best_coverage < 0.01:
        return None

    logger.info("    Best scene: %s (%.1f%% coverage)", Path(best[0]).name, best_coverage * 100)
    return best


# ---------------------------------------------------------------------------
# Main labeling pipeline
# ---------------------------------------------------------------------------


def label_tracks(tracks_dir, scenes_dirs, output_dir):
    """Label all track CSVs using Sentinel-2 scenes.

    Returns list of output CSV paths.
    """
    import pandas as pd

    tracks_dir = Path(tracks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find track CSVs
    csv_files = sorted(tracks_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in %s", tracks_dir)
        return []

    logger.info("Found %d track CSVs", len(csv_files))

    # Load all S2 scenes
    logger.info("Loading Sentinel-2 scenes...")
    scenes_info = []
    for scene_dir in scenes_dirs:
        result = load_sentinel2_rgb(scene_dir)
        if result is not None:
            rgb, transform, crs = result
            scenes_info.append((str(scene_dir), rgb, transform, crs))
            logger.info("  Loaded %s (%dx%d)", Path(scene_dir).name, rgb.shape[1], rgb.shape[0])

    if not scenes_info:
        logger.error("No valid Sentinel-2 scenes loaded")
        return []

    # Pre-segment all scenes
    logger.info("Segmenting %d scenes with HSV thresholds...", len(scenes_info))
    scenes_labels = []
    for scene_dir, rgb, transform, crs in scenes_info:
        label_img = segment_hsv(rgb)
        scenes_labels.append((scene_dir, label_img, transform, crs))
        n_ice = np.sum(label_img == 0)
        n_thin = np.sum(label_img == 1)
        n_water = np.sum(label_img == 2)
        total = label_img.size
        logger.info("  %s: ice=%.1f%% thin=%.1f%% water=%.1f%%",
                     Path(scene_dir).name,
                     100 * n_ice / total, 100 * n_thin / total, 100 * n_water / total)

    # Label each track
    outputs = []
    for csv_path in csv_files:
        logger.info("  Labeling %s", csv_path.name)
        df = pd.read_csv(csv_path, index_col=0)

        if "lon" not in df.columns or "lat" not in df.columns:
            logger.warning("    Missing lon/lat columns — skipping")
            continue

        lons = df["lon"].values
        lats = df["lat"].values

        # Find best matching scene
        best_label = None
        best_pix_x = None
        best_pix_y = None
        best_coverage = 0

        for scene_dir, label_img, transform, crs in scenes_labels:
            pix_x, pix_y = project_tracks_to_pixels(lons, lats, transform, crs)
            h, w = label_img.shape
            in_bounds = (pix_x >= 0) & (pix_x < w) & (pix_y >= 0) & (pix_y < h)
            coverage = np.mean(in_bounds)

            if coverage > best_coverage:
                best_coverage = coverage
                best_label = lookup_labels(label_img, pix_x, pix_y)
                best_pix_x = pix_x
                best_pix_y = pix_y

        if best_coverage < 0.01:
            logger.warning("    No scene covers this track (best coverage: %.1f%%) — "
                           "labeling all as unclassified", best_coverage * 100)
            df["pix_x"] = 0
            df["pix_y"] = 0
            df["label"] = 255
        else:
            df["pix_x"] = best_pix_x.astype(int)
            df["pix_y"] = best_pix_y.astype(int)
            df["label"] = best_label.astype(int)

            # Replace unclassified (255) with nearest neighbor label
            unclassified = df["label"] == 255
            if unclassified.any() and not unclassified.all():
                classified_idx = np.where(~unclassified)[0]
                unclassified_idx = np.where(unclassified)[0]
                nearest = np.searchsorted(classified_idx, unclassified_idx).clip(0, len(classified_idx) - 1)
                df.loc[unclassified, "label"] = df.iloc[classified_idx[nearest]]["label"].values

            n_total = len(df)
            for lbl, name in [(0, "thick ice"), (1, "thin ice"), (2, "open water")]:
                n = (df["label"] == lbl).sum()
                logger.info("    %s: %d (%.1f%%)", name, n, 100 * n / n_total)

        # Rename output to match expected pattern: *_labeled_10m.csv
        stem = csv_path.stem
        out_name = f"{stem}_labeled_10m.csv"
        out_path = output_dir / out_name
        df.to_csv(out_path, index=True)
        outputs.append(out_path)
        logger.info("    → %s", out_path)

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label ATL03 track segments using Sentinel-2 HSV classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label using a tar.gz of S2 scenes
  %(prog)s --tracks-dir ./data/extracted/ \\
           --sentinel2 ./data/sentinel2/sentinel2_scenes.tar.gz \\
           --output-dir ./data/labeled/

  # Label using a directory of already-extracted S2 scenes
  %(prog)s --tracks-dir ./data/extracted/ \\
           --sentinel2-dir ./data/sentinel2/scenes/ \\
           --output-dir ./data/labeled/
        """,
    )

    parser.add_argument(
        "--tracks-dir", type=str, required=True,
        help="Directory containing extracted ATL03 track CSVs",
    )
    parser.add_argument(
        "--sentinel2", type=str, default=None,
        help="Path to sentinel2_scenes.tar.gz",
    )
    parser.add_argument(
        "--sentinel2-dir", type=str, default=None,
        help="Directory containing extracted S2 scene subdirectories "
             "(each with B02.tif, B03.tif, B04.tif)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for labeled CSVs",
    )

    args = parser.parse_args()

    if not args.sentinel2 and not args.sentinel2_dir:
        parser.error("Provide either --sentinel2 (tar.gz) or --sentinel2-dir")

    # Resolve S2 scene directories
    tmp_extract_dir = None
    scenes_dirs = []

    if args.sentinel2_dir:
        s2_base = Path(args.sentinel2_dir)
        # Each subdirectory should be a scene with band GeoTIFFs
        for d in sorted(s2_base.iterdir()):
            if d.is_dir():
                scenes_dirs.append(d)
        # Also check if GeoTIFFs are directly in the directory
        if not scenes_dirs and list(s2_base.glob("*.tif")):
            scenes_dirs.append(s2_base)

    elif args.sentinel2:
        # Extract tar.gz to a temp directory
        tar_path = Path(args.sentinel2)
        if not tar_path.exists():
            logger.error("File not found: %s", tar_path)
            sys.exit(1)

        tmp_extract_dir = tempfile.mkdtemp(prefix="s2_scenes_")
        logger.info("Extracting %s → %s", tar_path, tmp_extract_dir)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_extract_dir)

        # Find scene subdirectories
        extract_base = Path(tmp_extract_dir)
        for d in sorted(extract_base.rglob("*")):
            if d.is_dir() and list(d.glob("*.tif")):
                scenes_dirs.append(d)
        if not scenes_dirs and list(extract_base.glob("*.tif")):
            scenes_dirs.append(extract_base)

    if not scenes_dirs:
        logger.error("No Sentinel-2 scene directories found")
        sys.exit(1)

    logger.info("Found %d Sentinel-2 scene(s)", len(scenes_dirs))

    try:
        outputs = label_tracks(args.tracks_dir, scenes_dirs, args.output_dir)

        print(f"\n{'=' * 70}")
        print("AUTO-LABELING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Track CSVs labeled:  {len(outputs)}")
        print(f"Output directory:    {args.output_dir}")
        print(f"HSV thresholds:")
        print(f"  Thick ice (0): V >= 205")
        print(f"  Thin ice  (1): 31 <= V < 205")
        print(f"  Open water(2): V < 31")
        print(f"{'=' * 70}\n")

    finally:
        # Clean up temp directory
        if tmp_extract_dir and os.path.exists(tmp_extract_dir):
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
