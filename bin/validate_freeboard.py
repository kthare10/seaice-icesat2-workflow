#!/usr/bin/env python3
"""
Validate ATL03-derived freeboard against ATL07/ATL10 products.

Source: Notebook 6 — per-track comparison plotting (classification, sea surface, freeboard
distribution, point density), parameterised into one function.

Reference-data semantics: the notebooks compare against Koo et al. (2023) ATL07 CSVs whose
``label`` is a classifier output (0 = water, 1 = thin ice, 2 = thick ice) and which carry
``h_ref``/``freeboard`` (ATL07) plus ``h_ref_10``/``freeboard_10`` (ATL10). CSVs produced by
``extract_atl07.py`` carry NASA's ``height_segment_type`` (0 = cloud, 1 = ice/snow, 2-9 = leads)
in ``label``; those are remapped to water/ice here (``--atl07-label-scheme``). Thin ice cannot be
recovered from the NASA codes.
"""

import argparse
import glob
import logging
import os
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BEAM_RE = re.compile(r"(gt\d[lr])", re.IGNORECASE)
DATE_RE = re.compile(r"(\d{8})\d{6}")

KOO_THICK, KOO_THIN, KOO_WATER = 2, 1, 0


def beam_of(track_id):
    m = BEAM_RE.search(str(track_id))
    return m.group(1).lower() if m else str(track_id).replace("_", "")


def date_of(track_data, track_id):
    """YYYYMMDD from year/month/day columns, else from the 14-digit granule timestamp."""
    if all(c in track_data.columns for c in ("year", "month", "day")):
        r = track_data.iloc[0]
        return f"{int(r.year):04d}{int(r.month):02d}{int(r.day):02d}"
    m = DATE_RE.search(str(track_id))
    if m:
        return m.group(1)
    if "day" in track_data.columns:  # legacy: November 2019 only
        return f"201911{int(track_data['day'].iloc[0]):02d}"
    return None


def remap_atl07_labels(atl07, scheme):
    """Return ATL07 frame with Koo-style labels (0 water, 1 thin, 2 thick)."""
    if "label" not in atl07.columns:
        return atl07
    if scheme == "auto":
        scheme = "nasa" if atl07["label"].max() > 2 else "koo"
    if scheme == "koo":
        return atl07
    lab = atl07["label"].astype(float)
    out = atl07.copy()
    water = (lab >= 2) & (lab <= 9)
    if "ssh_flag" in out.columns:
        water |= out["ssh_flag"].fillna(0).astype(float) == 1
    new = np.where(water, KOO_WATER, KOO_THICK).astype(float)
    new[lab == 0] = np.nan  # cloud covered
    out["label"] = new
    out = out[out["label"].notna()]
    logger.info("  Remapped NASA height_segment_type -> water/ice: %d water, %d ice, %d cloud dropped",
                int((new == KOO_WATER).sum()), int((new == KOO_THICK).sum()), int(np.isnan(new).sum()))
    return out


def load_reference(ref_dir, product, date_str, beam):
    """Load and beam-filter the ATL07/ATL10 CSV(s) for a date."""
    if not ref_dir or not date_str:
        return None
    files = sorted(glob.glob(os.path.join(ref_dir, f"{product}*{date_str}*.csv")))
    if not files:
        logger.info("  No %s CSV for %s in %s", product, date_str, ref_dir)
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d = d.loc[:, ~d.columns.str.contains("^Unnamed")]
        if "beam" in d.columns:
            d = d[d["beam"].astype(str).str.lower() == beam]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    sort_col = "x" if "x" in d.columns else ("lon" if "lon" in d.columns else None)
    if sort_col:
        d = d.sort_values(by=[sort_col]).reset_index(drop=True)
    logger.info("  Loaded %s: %d rows for beam %s", product, len(d), beam)
    return d


def validate_track(atl03_data, atl07_data, track_id, output_dir):
    """Generate comparison plots for a single track."""
    plots = []
    x_col = "lon" if "lon" in atl03_data.columns else "x_atc"
    x_label = "Longitude (deg)" if x_col == "lon" else "Along-track distance (m)"
    fb_col = next((c for c in ("freeboard_new_h_ref_smooth", "freeboard_new_h_ref", "freeboard_min_elev")
                   if c in atl03_data.columns), None)
    ss_col = next((c for c in ("new_h_ref_smooth", "new_h_ref", "sea_surf_min_elev")
                   if c in atl03_data.columns), None)

    thick_ice = atl03_data[atl03_data.pred_label == 0]
    thin_ice = atl03_data[atl03_data.pred_label == 1]
    water = atl03_data[atl03_data.pred_label == 2]

    # 1. Classification scatter
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.scatter(thick_ice[x_col], thick_ice.h_cor_mean, s=2, c="C0", label="thick ice")
    ax.scatter(thin_ice[x_col], thin_ice.h_cor_mean, s=2, c="C1", label="thin ice")
    ax.scatter(water[x_col], water.h_cor_mean, s=2, c="C2", label="open water")
    ax.set_xlabel(x_label)
    ax.set_ylabel("ATL03 elevation (m)")
    ax.set_ylim(-0.5, 3.5)
    ax.legend(loc="upper right")
    ax.set_title(f"ATL03 Classification — {track_id}")
    ax.grid(linestyle=":", linewidth=0.5)
    fname = os.path.join(output_dir, f"classification_{track_id}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    plots.append(fname)

    # 2. Sea surface methods
    if ss_col is not None:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        for col, color, name in [("sea_surf_min_elev", "C2", "Minimum Elevation"),
                                 ("sea_surf_avg_elev", "C1", "Average Elevation"),
                                 ("sea_surf_min_dist", "C3", "Nearest Minimum Elevation"),
                                 ("new_h_ref_smooth", "C0", "NASA Sea Surface Formula (smoothed)")]:
            if col in atl03_data.columns:
                ax.scatter(atl03_data.x_atc, atl03_data[col], s=2, c=color, label=name)
        if atl07_data is not None and "h_ref" in atl07_data.columns and "x" in atl07_data.columns:
            ax.scatter(atl07_data.x, atl07_data.h_ref, s=2, c="C4", label="ATL07 h_ref")
        ax.set_xlabel("Along-track distance (m)")
        ax.set_ylabel("Sea surface elevation (m)")
        ax.set_ylim(-0.4, 0.4)
        ax.legend(loc="upper right", markerscale=4)
        ax.set_title(f"Sea Surface — {track_id}")
        ax.grid(linestyle=":", linewidth=0.5)
        fname = os.path.join(output_dir, f"sea_surface_{track_id}.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        plots.append(fname)

    if atl07_data is not None and len(atl07_data) > 0 and "label" in atl07_data.columns:
        thick_ice7 = atl07_data[atl07_data.label == KOO_THICK]
        thin_ice7 = atl07_data[atl07_data.label == KOO_THIN]
        water7 = atl07_data[atl07_data.label == KOO_WATER]

        # 3. ATL07 classification
        if "height" in atl07_data.columns:
            fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
            ax.scatter(thick_ice7.lon, thick_ice7.height, s=2, c="C0", label="thick ice (ATL07)")
            ax.scatter(thin_ice7.lon, thin_ice7.height, s=2, c="C1", label="thin ice (ATL07)")
            ax.scatter(water7.lon, water7.height, s=2, c="C2", label="open water (ATL07)")
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("ATL07 elevation (m)")
            ax.set_ylim(-0.5, 3.5)
            ax.legend(loc="upper right")
            ax.set_title(f"ATL07 Classification — {track_id}")
            ax.grid(linestyle=":", linewidth=0.5)
            fname = os.path.join(output_dir, f"atl07_classification_{track_id}.png")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            plots.append(fname)

        # 4. Freeboard distribution
        if fb_col is not None and "freeboard" in atl07_data.columns:
            fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
            atl03_ice = atl03_data[atl03_data.pred_label <= 1]
            atl07_ice = atl07_data[atl07_data.label >= 1]
            if len(atl03_ice) > 0:
                ax.hist(atl03_ice[fb_col].dropna().clip(0, 3.5), bins=50, histtype="step",
                        ec="blue", linewidth=2, label="ATL03 freeboard")
            if len(atl07_ice) > 0:
                ax.hist(atl07_ice.freeboard.dropna().clip(0, 3.5), bins=50, histtype="step",
                        ec="red", linewidth=2, label="ATL07 freeboard")
                if "freeboard_10" in atl07_ice.columns:
                    ax.hist(atl07_ice.freeboard_10.dropna().clip(0, 3.5), bins=50, histtype="step",
                            ec="green", linewidth=2, label="ATL10 freeboard")
            ax.set_xlabel("Freeboard (m)")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 3.5)
            ax.legend()
            ax.set_title(f"Freeboard Distribution — {track_id}")
            fname = os.path.join(output_dir, f"freeboard_hist_{track_id}.png")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            plots.append(fname)

        # 5. Point density
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        atl03_ice = atl03_data[atl03_data.pred_label <= 1]
        atl07_ice = atl07_data[atl07_data.label >= 1]
        if len(atl03_ice) > 0:
            ax.hist(atl03_ice[x_col], bins=40, alpha=0.5, edgecolor="black", label="ATL03 ice")
        if len(atl07_ice) > 0:
            atl07_x = "lon" if "lon" in atl07_ice.columns else "x"
            ax.hist(atl07_ice[atl07_x], bins=40, alpha=0.5, color="green", edgecolor="black",
                    label="ATL07/ATL10 ice")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(f"Point Density — {track_id}")
        fname = os.path.join(output_dir, f"density_{track_id}.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        plots.append(fname)

    return plots


def main():
    parser = argparse.ArgumentParser(description="Validate ATL03 freeboard against ATL07/ATL10")
    parser.add_argument("--atl03-freeboard", required=True,
                        help="ATL03 freeboard CSV (output from compute_freeboard)")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--output-tar", default=None,
                        help="Output tar.gz filename (default: validation_report.tar.gz in output-dir)")
    parser.add_argument("--atl07-dir", default=None, help="Directory with ATL07 CSV files")
    parser.add_argument("--atl10-dir", default=None, help="Directory with ATL10 CSV files")
    parser.add_argument("--atl07-label-scheme", choices=["auto", "koo", "nasa"], default="auto",
                        help="Meaning of the ATL07 'label' column (default: auto-detect)")
    parser.add_argument("--track", type=str, default=None, help="Single track id to validate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading ATL03 freeboard from %s", args.atl03_freeboard)
    atl03 = pd.read_csv(args.atl03_freeboard)
    atl03 = atl03.loc[:, ~atl03.columns.str.contains("^Unnamed")]
    if len(atl03) and "h_cor_mean" in atl03.columns:
        atl03 = atl03[atl03.h_cor_mean <= 10]

    track_col = next((c for c in ("track", "track_x") if c in atl03.columns), None)
    if args.track:
        tracks = [args.track]
    elif track_col:
        tracks = atl03[track_col].astype(str).unique().tolist()
    else:
        tracks = ["all"]

    all_plots = []
    for track_id in tracks:
        logger.info("Validating track: %s", track_id)
        if track_col and track_id != "all":
            track_data = atl03[atl03[track_col].astype(str) == track_id].copy()
        else:
            track_data = atl03.copy()
        if len(track_data) == 0:
            logger.warning("No data for track %s, skipping", track_id)
            continue
        track_data = track_data.sort_values(by=["x_atc"]).reset_index(drop=True)

        beam = beam_of(track_id)
        date_str = date_of(track_data, track_id)
        atl07_data = load_reference(args.atl07_dir, "ATL07", date_str, beam)
        if atl07_data is not None:
            atl07_data = remap_atl07_labels(atl07_data, args.atl07_label_scheme)
            atl10_data = load_reference(args.atl10_dir, "ATL10", date_str, beam)
            if (atl10_data is not None and "freeboard" in atl10_data.columns
                    and "freeboard_10" not in atl07_data.columns and "lon" in atl10_data.columns):
                atl07_data = pd.merge_asof(
                    atl07_data.sort_values("lon"),
                    atl10_data[["lon", "freeboard"]].sort_values("lon").rename(
                        columns={"freeboard": "freeboard_10"}),
                    on="lon", direction="nearest")
                logger.info("  Merged ATL10 freeboard_10: %d non-null",
                            int(atl07_data["freeboard_10"].notna().sum()))

        all_plots.extend(validate_track(track_data, atl07_data, track_id, args.output_dir))

    tar_path = args.output_tar or os.path.join(args.output_dir, "validation_report.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for plot in all_plots:
            tar.add(plot, arcname=os.path.basename(plot))
    logger.info("Validation report: %s (%d plots)", tar_path, len(all_plots))


if __name__ == "__main__":
    main()
