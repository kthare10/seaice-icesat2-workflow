#!/usr/bin/env python3
"""
Preprocess ATL03 track CSV: z-score, relative sea surface, interpolation, h_diff.

Source: Notebook 2 — cells 10-14 (rel sea surface), 16 (interpolation), 19 (h_diff).
The z-score step (cells 5-6) is disabled in the author's final pipeline (the notebook breaks
before running it and nothing downstream reads its output); it is available via --zscore.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Z-score correction (Notebook 2, cell 5)
# ---------------------------------------------------------------------------

def set_elev_correction(window_data_slice, input_data):
    """Compute per-label z-scores within a sliding window."""
    thick_ice_slice = window_data_slice[window_data_slice["label"] == 0].copy()
    thin_ice_slice = window_data_slice[window_data_slice["label"] == 1].copy()
    open_water_slice = window_data_slice[window_data_slice["label"] == 2].copy()

    for s in [thick_ice_slice, thin_ice_slice, open_water_slice]:
        if len(s) > 1:
            s["z_score"] = zscore(s["h_cor_mean"], nan_policy="omit")
        elif len(s) == 1:
            s["z_score"] = 0.0

    combined = pd.concat(
        [thick_ice_slice, thin_ice_slice, open_water_slice]
    ).sort_values(by=["x_atc"])

    for idx, row in combined.iterrows():
        input_data.loc[idx, "z_score"] = row["z_score"]


def find_elev_correction(radius, input_data):
    """Iterate through sliding windows to compute z-scores."""
    index = 0
    while index < len(input_data):
        base_x = input_data.x_atc.iloc[index]
        start_x = base_x - radius
        end_x = base_x + radius
        window = input_data.loc[
            (input_data["x_atc"] > start_x) & (input_data["x_atc"] < end_x),
            ["x_atc", "h_cor_mean", "label"],
        ]
        set_elev_correction(window, input_data)
        index = window.index[-1] + 1


# ---------------------------------------------------------------------------
# Relative sea surface (Notebook 2, cell 10)
# ---------------------------------------------------------------------------

def set_sea_surface_ow(window_data_slice, input_data):
    """Compute min/avg/nearest-minimum sea surface elevations from open water."""
    df_label = window_data_slice.label.max()

    if df_label <= 1:
        min_elev = np.nan
        avg_elev = np.nan
        min_dist_low10 = pd.DataFrame()
    else:
        ow = window_data_slice[window_data_slice["label"].eq(df_label)]
        min_elev = ow["h_cor_mean"].min()
        avg_elev = ow["h_cor_mean"].mean()
        min_dist_low10 = (
            ow.sort_values(by="h_cor_mean", ignore_index=False)
            .iloc[:10]
            .sort_values(by="x_atc", ignore_index=False)
        )

    for df in range(len(window_data_slice)):
        idx = window_data_slice.index[df]
        input_data.loc[idx, "rel_sea_surf_min_elev"] = min_elev
        input_data.loc[idx, "rel_sea_surf_avg_elev"] = avg_elev

        x = window_data_slice.x_atc.iloc[df]
        min_dist_elev = np.nan

        if not min_dist_low10.empty:
            min_dist = sys.float_info.max
            min_dist_idx = -1
            for df2, row in min_dist_low10.iterrows():
                diff = abs(x - row["x_atc"])
                if diff < min_dist:
                    min_dist = diff
                    min_dist_idx = df2
            min_dist_elev = min_dist_low10.loc[min_dist_idx, "h_cor_mean"]

        input_data.loc[idx, "rel_sea_surf_min_dist"] = min_dist_elev


def find_sea_surface_ow(radius, input_data):
    """Iterate through sliding windows to compute relative sea surface."""
    index = 0
    while index < len(input_data):
        base_x = input_data.x_atc.iloc[index]
        start_x = base_x - radius
        end_x = base_x + radius
        window = input_data.loc[
            (input_data["x_atc"] > start_x) & (input_data["x_atc"] < end_x),
            ["x_atc", "h_cor_mean", "label"],
        ]
        set_sea_surface_ow(window, input_data)
        index = window.index[-1] + 1


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ATL03 track CSV: z-score, rel sea surface, interpolation, h_diff"
    )
    parser.add_argument("--input", required=True, help="Input corrected labeled CSV")
    parser.add_argument("--output", required=True, help="Output enriched CSV")
    parser.add_argument(
        "--radius", type=float, default=5000.0, help="Sliding window radius in meters (default: 5000)"
    )
    parser.add_argument(
        "--elev-threshold", type=float, default=10.0, help="Max elevation filter in meters (default: 10)"
    )
    parser.add_argument(
        "--zscore", action="store_true",
        help="Also compute per-class window z-scores and drop rows where it is NaN "
             "(Notebook 2 cells 5-6; not part of the author's final pipeline)",
    )
    args = parser.parse_args()

    logger.info("Loading %s", args.input)
    data = pd.read_csv(args.input, index_col=0)
    data = data.sort_values(by=["x_atc"])
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    data = data[data.h_cor_mean <= args.elev_threshold]
    data = data[data["label"].notna()]
    data = data.reset_index(drop=True)
    logger.info("Rows after filtering: %d", len(data))

    if len(data) == 0:
        logger.error("No rows remain after elevation/label filtering — writing header-only output")
        data.to_csv(args.output, index=False)
        sys.exit(0)

    if (data["label"] == 2).sum() == 0:
        logger.warning("Track has no open-water (label 2) segments: rel_sea_surf_* will be NaN "
                       "everywhere and the track cannot contribute rel_height features")

    # Step 1 (optional): Z-score correction
    if args.zscore:
        logger.info("Computing z-scores (radius=%.0f m)...", args.radius)
        data["z_score"] = np.nan
        find_elev_correction(args.radius, data)
        data = data[data["z_score"].notna()]
        data = data.reset_index(drop=True)
        logger.info("Rows after z-score filter: %d", len(data))

    # Step 2: Relative sea surface
    logger.info("Computing relative sea surface heights...")
    data["rel_sea_surf_min_elev"] = np.nan
    data["rel_sea_surf_avg_elev"] = np.nan
    data["rel_sea_surf_min_dist"] = np.nan
    find_sea_surface_ow(args.radius, data)

    # Step 3: Interpolation
    logger.info("Interpolating NaN sea surface values...")
    for col in ["rel_sea_surf_min_elev", "rel_sea_surf_avg_elev", "rel_sea_surf_min_dist"]:
        data[col] = data[col].interpolate(method="linear", limit_direction="both", axis=0)

    # Step 4: Relative heights
    data["rel_height_min_elev"] = data["h_cor_mean"] - data["rel_sea_surf_min_elev"]
    data["rel_height_avg_elev"] = data["h_cor_mean"] - data["rel_sea_surf_avg_elev"]
    data["rel_height_min_dist_elev"] = data["h_cor_mean"] - data["rel_sea_surf_min_dist"]

    # Step 5: h_diff
    logger.info("Computing h_diff = h_cor_mean - h_cor_med...")
    data["h_diff"] = data["h_cor_mean"] - data["h_cor_med"]

    data.to_csv(args.output, index=False)
    logger.info("Saved enriched CSV to %s (%d rows)", args.output, len(data))


if __name__ == "__main__":
    main()
