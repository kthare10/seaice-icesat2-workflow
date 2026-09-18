#!/usr/bin/env python3
"""
Compute sea ice freeboard from LSTM classification results.

Source: Notebook 5 — cells 10 (simple sea surface), 14 (main loop), 16 (interpolation),
        24 (set_new_sea_surface_eqn_ow, NASA sea-surface equation), 25 (invocation);
        Notebook 6 — smoothed sea surface (nanmin window + nearest interpolation) that is the
        product plotted in the paper's Figs. 8-11.

Uses pred_label (not label) for open-water detection. Inputs must be in metres
(inference_lstm.py re-attaches the raw centre-point features).

Lead weight: the paper (Eq. 2) uses w_i = exp(-((h_i - h_min)/sigma_i)^2). The notebook wrote
``np.exp(-(hi-hmin)/si)**2`` (= exp(-2 (h-hmin)/sigma)); ``--weight-form notebook`` reproduces it.

Windows: each iteration takes the 10 km window centred on the first unprocessed row and assigns
its estimate to every row in it, then jumps to the window's end. Because the back half of the next
window overwrites the front half of this one, every row ends up with the estimate from the 10 km
window that starts at its own 5 km chunk, i.e. 10 km windows at a 5 km stride, as the paper
describes (Section III.D.1).

Lead fallback: when a window has no predicted open water, the notebook (NB5 cell 24) treats
thin-ice segments as leads. The paper instead says such windows are filled by interpolation from
the nearest window. ``--lead-fallback thin_ice`` (default, notebook parity) or ``none`` (paper).

Smoothing: the Notebook 6 nanmin window is defined in rows, not metres. A track assembled from
tiles that are far apart would smooth across the gap, so ``--smooth-max-gap`` splits the series
at along-track jumps larger than that distance and smooths each piece independently.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple sea surface (min elev, avg elev, nearest min) — Notebook 5, cells 10/14/16
# ---------------------------------------------------------------------------

def compute_simple_sea_surface(input_data, radius):
    """Sliding-window (non-overlapping advance) sea surface from open-water segments."""
    input_data["sea_surf_min_elev"] = np.nan
    input_data["sea_surf_avg_elev"] = np.nan
    input_data["sea_surf_min_dist"] = np.nan

    x_all = input_data["x_atc"].to_numpy(dtype=float)
    h_all = input_data["h_cor_mean"].to_numpy(dtype=float)
    lab_all = input_data["pred_label"].to_numpy(dtype=float)
    n = len(input_data)
    min_elev_col = np.full(n, np.nan)
    avg_elev_col = np.full(n, np.nan)
    min_dist_col = np.full(n, np.nan)

    index = 0
    while index < n:
        base_x = x_all[index]
        # window: (base_x - radius, base_x + radius); data sorted by x_atc
        lo = np.searchsorted(x_all, base_x - radius, side="right")
        hi = np.searchsorted(x_all, base_x + radius, side="left")
        if hi <= lo:
            index += 1
            continue
        w_lab = lab_all[lo:hi]
        w_h = h_all[lo:hi]
        w_x = x_all[lo:hi]
        df_label = np.nanmax(w_lab)
        if df_label > 1:
            ow = w_lab == df_label
            min_elev = np.nanmin(w_h[ow])
            avg_elev = np.nanmean(w_h[ow])
            ow_idx = np.where(ow)[0]
            low10 = ow_idx[np.argsort(w_h[ow_idx], kind="stable")[:10]]
            low10_x = w_x[low10]
            low10_h = w_h[low10]
            # nearest of the 10 lowest open-water points, for every point in the window
            nearest = np.abs(w_x[:, None] - low10_x[None, :]).argmin(axis=1)
            min_dist_col[lo:hi] = low10_h[nearest]
            min_elev_col[lo:hi] = min_elev
            avg_elev_col[lo:hi] = avg_elev
        index = hi

    input_data["sea_surf_min_elev"] = min_elev_col
    input_data["sea_surf_avg_elev"] = avg_elev_col
    input_data["sea_surf_min_dist"] = min_dist_col

    for col in ["sea_surf_min_elev", "sea_surf_avg_elev", "sea_surf_min_dist"]:
        input_data[col] = input_data[col].interpolate(method="linear", limit_direction="both", axis=0)

    input_data["freeboard_min_elev"] = input_data["h_cor_mean"] - input_data["sea_surf_min_elev"]
    input_data["freeboard_avg_elev"] = input_data["h_cor_mean"] - input_data["sea_surf_avg_elev"]
    input_data["freeboard_min_dist_elev"] = input_data["h_cor_mean"] - input_data["sea_surf_min_dist"]


# ---------------------------------------------------------------------------
# NASA sea surface equation (Notebook 5, cell 24; paper Eqs. 2-3)
# ---------------------------------------------------------------------------

def lead_weights(hi, si, weight_form):
    si = np.where(si <= 0, 1e-10, si)
    hmin = hi.min()
    if weight_form == "notebook":
        return np.exp(-(hi - hmin) / si) ** 2
    return np.exp(-((hi - hmin) / si) ** 2)


def compute_nasa_sea_surface(input_data, radius, weight_form="paper", lead_fallback="thin_ice"):
    """Reference sea surface h_ref per 10 km window from open-water leads.

    ``lead_fallback="thin_ice"`` uses thin-ice segments as leads when a window has no predicted
    open water (notebook behaviour); ``"none"`` leaves the window NaN so that the later linear
    interpolation fills it from neighbouring windows (what the paper describes).
    """
    x_all = input_data["x_atc"].to_numpy(dtype=float)
    h_all = input_data["h_cor_mean"].to_numpy(dtype=float)
    s_all = input_data["height_sd"].to_numpy(dtype=float)
    lab_all = input_data["pred_label"].to_numpy(dtype=float)
    n = len(input_data)
    h_ref_col = np.full(n, np.nan)
    s_ref_col = np.full(n, np.nan)
    n_windows = n_fallback = 0

    index = 0
    while index < n:
        base_x = x_all[index]
        lo = np.searchsorted(x_all, base_x - radius, side="right")
        hi_ = np.searchsorted(x_all, base_x + radius, side="left")
        if hi_ <= lo:
            index += 1
            continue
        n_windows += 1
        w_lab = lab_all[lo:hi_]
        lead_idx = np.where(w_lab > 1)[0]
        if len(lead_idx) == 0 and lead_fallback == "thin_ice":
            lead_idx = np.where(w_lab > 0)[0]
            if len(lead_idx):
                n_fallback += 1

        if len(lead_idx) > 0:
            # consecutive-row groups form individual leads
            breaks = np.where(np.diff(lead_idx) != 1)[0]
            starts = np.concatenate(([lead_idx[0]], lead_idx[breaks + 1]))
            ends = np.concatenate((lead_idx[breaks], [lead_idx[-1]]))
            hlead, slead = [], []
            for a, b in zip(starts, ends):
                hi = h_all[lo + a: lo + b + 1]
                si = s_all[lo + a: lo + b + 1]
                wi = lead_weights(hi, si, weight_form)
                ai = wi / np.sum(wi)
                hlead.append(np.sum(ai * hi))
                slead.append(np.sum((ai ** 2) * (si ** 2)))
            hlead = np.array(hlead)
            slead = np.array(slead)
            slead = np.where(slead <= 0, 1e-10, slead)
            alead = (1 / slead) / np.sum(1 / slead)
            h_ref_col[lo:hi_] = np.sum(hlead * alead)
            s_ref_col[lo:hi_] = np.sum((alead ** 2) * slead)
        index = hi_

    n_empty = int(np.isnan(h_ref_col).sum())
    logger.info("NASA sea surface: %d windows, %d used thin ice as leads, %d rows without leads "
                "(interpolated)", n_windows, n_fallback, n_empty)
    input_data["new_h_ref"] = h_ref_col
    input_data["new_s_ref"] = s_ref_col
    input_data["new_h_ref"] = input_data["new_h_ref"].interpolate(
        method="linear", limit_direction="both", axis=0)
    input_data["freeboard_new_h_ref"] = input_data["h_cor_mean"] - input_data["new_h_ref"]


# ---------------------------------------------------------------------------
# Smoothed sea surface (Notebook 6)
# ---------------------------------------------------------------------------

def _smooth_piece(s, window):
    """Notebook 6 smooth_line on one contiguous piece: nanmin over ±window rows, then nearest."""
    if not s.notna().any() or window <= 0:
        return s
    out = s.rolling(2 * window + 1, center=True, min_periods=1).min()
    try:
        return out.interpolate(method="nearest", limit_direction="both")
    except Exception:  # scipy missing or single valid point
        return out.interpolate(method="linear", limit_direction="both")


def compute_smoothed_sea_surface(input_data, window, water_threshold=None, max_gap=None):
    """nanmin over ±window rows, then nearest-neighbour interpolation (Notebook 6 smooth_line).

    ``water_threshold`` (metres) reproduces the notebook's masking of h_ref values above the
    expected water level before smoothing (NB6 cells 9-19; the notebook derives the level as
    0.2 m + mean(ATL03 - ATL07), which needs ATL07 data).

    ``max_gap`` (metres): along-track jumps larger than this split the series into pieces that
    are smoothed independently, so a track assembled from distant tiles does not borrow its
    sea surface from tens of kilometres away. None smooths the whole series as one (notebook).
    """
    s = input_data["new_h_ref"].copy()
    if water_threshold is not None:
        n_masked = int((s > water_threshold).sum())
        s = s.where(s <= water_threshold)
        logger.info("Masked %d new_h_ref values above %.3f m before smoothing", n_masked, water_threshold)

    if max_gap is not None and max_gap > 0:
        gaps = np.flatnonzero(np.diff(input_data["x_atc"].to_numpy(dtype=float)) > max_gap)
        if len(gaps):
            logger.info("Smoothing %d pieces separately (along-track gaps > %.0f m at rows %s)",
                        len(gaps) + 1, max_gap, gaps.tolist())
        bounds = [0] + (gaps + 1).tolist() + [len(s)]
        pieces = [_smooth_piece(s.iloc[a:b], window) for a, b in zip(bounds[:-1], bounds[1:])]
        s = pd.concat(pieces)
    else:
        s = _smooth_piece(s, window)

    input_data["new_h_ref_smooth"] = s
    input_data["freeboard_new_h_ref_smooth"] = input_data["h_cor_mean"] - input_data["new_h_ref_smooth"]


def main():
    parser = argparse.ArgumentParser(
        description="Compute sea ice freeboard from LSTM classification results"
    )
    parser.add_argument("--input", required=True, help="Predictions CSV with pred_label column")
    parser.add_argument("--output", required=True, help="Output freeboard CSV")
    parser.add_argument("--radius", type=float, default=5000.0,
                        help="Sliding window radius in metres (default: 5000)")
    parser.add_argument("--method", choices=["nasa_sea_surface_eqn", "simple"],
                        default="nasa_sea_surface_eqn",
                        help="Sea surface detection method (default: nasa_sea_surface_eqn)")
    parser.add_argument("--weight-form", choices=["paper", "notebook"], default="paper",
                        help="Lead weight formula (default: paper Eq. 2)")
    parser.add_argument("--smooth-window", type=int, default=10000,
                        help="±rows for the nanmin smoothing of new_h_ref (Notebook 6: 10000; 0 disables)")
    parser.add_argument("--smooth-max-gap", type=float, default=10000.0,
                        help="Smooth pieces separated by along-track gaps larger than this (m) "
                             "independently (default: 10000; 0 = whole series, notebook behaviour)")
    parser.add_argument("--water-threshold", type=float, default=None,
                        help="Mask new_h_ref above this height (m) before smoothing, as Notebook 6 "
                             "does with 0.2 m + mean(ATL03 - ATL07) (optional)")
    parser.add_argument("--lead-fallback", choices=["thin_ice", "none"], default="thin_ice",
                        help="Windows without predicted open water: use thin ice as leads "
                             "(notebook, default) or leave them to interpolation (paper)")
    parser.add_argument("--track", type=str, default=None,
                        help="Keep only rows whose track id equals this value (per-track fan-out)")
    parser.add_argument("--elev-threshold", type=float, default=10.0,
                        help="Drop rows with h_cor_mean above this (m, default: 10)")
    args = parser.parse_args()

    logger.info("Loading %s", args.input)
    data = pd.read_csv(args.input)
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    for col in ("x_atc", "h_cor_mean", "height_sd", "pred_label"):
        if col not in data.columns:
            raise SystemExit(f"Input is missing required column '{col}'")

    if args.track:
        track_col = "track" if "track" in data.columns else ("track_x" if "track_x" in data.columns else None)
        if track_col is None:
            raise SystemExit("--track given but input has no 'track' column")
        data = data[data[track_col].astype(str) == args.track]
        logger.info("Filtered to track %s: %d rows", args.track, len(data))

    data = data[data.h_cor_mean <= args.elev_threshold]
    data = data[data["pred_label"].notna()]
    data = data.sort_values(by=["x_atc"]).reset_index(drop=True)
    logger.info("Input rows: %d", len(data))

    if len(data) == 0:
        logger.error("No rows to process — writing header-only output")
        data.to_csv(args.output, index=False)
        sys.exit(0)

    logger.info("Computing simple sea surface (radius=%.0f m)...", args.radius)
    compute_simple_sea_surface(data, args.radius)

    if args.method == "nasa_sea_surface_eqn":
        logger.info("Computing NASA sea surface equation (radius=%.0f m, weights=%s, fallback=%s)...",
                    args.radius, args.weight_form, args.lead_fallback)
        compute_nasa_sea_surface(data, args.radius, weight_form=args.weight_form,
                                 lead_fallback=args.lead_fallback)
        n_nan = int(data["new_h_ref"].isna().sum())
        if n_nan:
            logger.warning("new_h_ref is NaN for %d rows (no leads in track)", n_nan)
        compute_smoothed_sea_surface(data, args.smooth_window, args.water_threshold,
                                     max_gap=args.smooth_max_gap or None)

    data.to_csv(args.output, index=False)
    logger.info("Saved freeboard CSV to %s (%d rows)", args.output, len(data))


if __name__ == "__main__":
    main()
