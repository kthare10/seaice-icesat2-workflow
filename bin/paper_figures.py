#!/usr/bin/env python3
"""
Generate publication-quality figures and tables for the paper:
"Scalable Higher Resolution Polar Sea Ice Classification and Freeboard
Calculation from ICESat-2 ATL03 Data" (Iqrah et al., 2502.02700v1).

Tables I-V are transcribed from the paper (arXiv 2502.02700v1). Figures use pipeline output CSVs
when available, falling back to hardcoded paper values where possible.

Can run standalone or as a Pegasus workflow job (stage 8).

Usage (standalone):
    python paper_figures.py --tables-only --output-dir figures/
    python paper_figures.py --figure 5 --output-dir figures/
    python paper_figures.py --all --predictions pred.csv --freeboard fb.csv \
                            --atl07-csv atl07_1.csv atl07_2.csv --output-dir figures/

Usage (workflow job):
    python paper_figures.py --all --predictions LSTM_predictions_all.csv \
        --freeboard freeboard_merged.csv --atl07-csv atl07_1.csv atl07_2.csv \
        --output-dir . --output-tar paper_figures.tar.gz
"""

import argparse
import logging
import os
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication style (IEEE conference)
# ---------------------------------------------------------------------------
IEEE_SINGLE_COL = 3.5   # inches
IEEE_DOUBLE_COL = 7.0   # inches
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Table data (hardcoded from paper)
# ---------------------------------------------------------------------------

TABLE_I_DATA = {
    "title": "Table I: IS2 ATL03 and S2 coincident pairs (< 2 h) in the Ross Sea, Nov 2019",
    "columns": ["#", "IS2 acquisition (UTC)", "S2 acquisition (UTC)",
                 "Time difference (min)", "Shift of S2 image"],
    "rows": [
        ["1", "2019/11/03 18:44:32", "2019/11/03 18:34:59", "9.55", "550 m / NW"],
        ["2", "2019/11/04 19:53:11", "2019/11/04 19:45:29", "7.7", "0 m"],
        ["3", "2019/11/13 19:10:53", "2019/11/13 18:34:59", "35.9", "200 m / W"],
        ["4", "2019/11/16 19:28:13", "2019/11/16 18:44:59", "43.23", "0 m"],
        ["5", "2019/11/17 19:02:34", "2019/11/17 18:15:09", "47.57", "530 m / NW"],
        ["6", "2019/11/20 19:19:52", "2019/11/20 20:05:29", "45.62", "400 m / NW"],
        ["7", "2019/11/23 18:02:55", "2019/11/23 18:34:59", "32.07", "150 m / E"],
        ["8", "2019/11/26 18:20:14", "2019/11/26 18:44:59", "24.75", "350 m / SW"],
    ],
}

_SPARK_COLUMNS = ["Executors", "Cores", "Load Time (s)", "Map Time (s)", "Reduce Time (s)",
                  "Speedup Load", "Speedup Reduce"]

TABLE_II_DATA = {
    "title": "Table II: PySpark-based IS2 auto-labeling scalability over Google Cloud",
    "columns": _SPARK_COLUMNS,
    "rows": [
        ["1", "1", "108", "0.4", "390", "1", "1"],
        ["1", "2", "58", "0.4", "174", "1.86", "2.24"],
        ["1", "4", "33", "0.3", "72", "3.27", "5.42"],
        ["2", "1", "56", "0.3", "156", "1.93", "2.5"],
        ["2", "2", "31", "0.3", "84", "3.48", "4.64"],
        ["2", "4", "19", "0.3", "41", "5.68", "9.51"],
        ["4", "1", "31", "0.2", "78", "3.48", "5"],
        ["4", "2", "17", "0.2", "39", "6.35", "10"],
        ["4", "4", "12", "0.3", "24", "9", "16.25"],
    ],
}

TABLE_III_DATA = {
    "title": "Table III: DL models sea ice classification accuracy over IS2 ATL03 (%)",
    "columns": ["Model", "Accuracy", "Precision", "Recall", "F1 score"],
    "rows": [
        ["MLP", "91.80", "91.80", "91.80", "91.79"],
        ["LSTM", "96.56", "97.00", "96.09", "96.54"],
    ],
}

TABLE_IV_DATA = {
    "title": "Table IV: Distributed DL model training using Horovod on a DGX A100 cluster",
    "columns": ["No. of GPUs", "Time (s)", "Time (s)/Epoch", "Data/s", "Speedup"],
    "rows": [
        ["1", "280.72", "5.5", "585.88", "1.00"],
        ["2", "143.22", "2.778", "1160.81", "1.96"],
        ["4", "73.68", "1.45", "2229.56", "3.81"],
        ["6", "49.42", "0.97", "3330.03", "5.68"],
        ["8", "38.72", "0.79", "4248.56", "7.25"],
    ],
}

TABLE_V_DATA = {
    "title": "Table V: PySpark-based IS2 freeboard computation over Google Cloud",
    "columns": _SPARK_COLUMNS,
    "rows": [
        ["1", "1", "111", "0.4", "392", "1", "1"],
        ["1", "2", "60", "0.4", "177", "1.85", "2.21"],
        ["1", "4", "36", "0.3", "74", "3.08", "5.30"],
        ["2", "1", "58", "0.3", "159", "1.91", "2.47"],
        ["2", "2", "33", "0.3", "86", "3.36", "4.56"],
        ["2", "4", "21", "0.3", "44", "5.29", "8.91"],
        ["4", "1", "34", "0.2", "80", "3.26", "4.9"],
        ["4", "2", "20", "0.2", "41", "5.55", "9.56"],
        ["4", "4", "13", "0.3", "25", "8.54", "15.68"],
    ],
}

ALL_TABLES = {
    1: TABLE_I_DATA,
    2: TABLE_II_DATA,
    3: TABLE_III_DATA,
    4: TABLE_IV_DATA,
    5: TABLE_V_DATA,
}

# Track parameters from the paper
TRACK_PARAMS = {
    "track1": {
        "track_id": "20191104195311_05940510_gt2r",
        "day": 4,
        "beam": "gt2r",
        "lon1": -170.5,
        "lon2": -170.0,
    },
    "track2": {
        "track_id": "20191126182014_09290510_gt2r",
        "day": 26,
        "beam": "gt2r",
        "lon1": -162.96,
        "lon2": -162.70,
    },
}

# Paper's headline metrics (Table III and Fig. 4), used by the comparison figure.
PAPER_METRICS = {
    "LSTM": {"Accuracy": 96.56, "Precision": 97.00, "Recall": 96.09, "F1": 96.54},
    "MLP": {"Accuracy": 91.80, "Precision": 91.80, "Recall": 91.80, "F1": 91.79},
}
# Per-class accuracy as quoted in the paper's text (Section IV.C.1). The figure itself
# prints 60.35 for open water; the text says 60.25. The text value is used here.
PAPER_PER_CLASS = {"thick ice": 98.39, "thin ice": 73.80, "open water": 60.25}

# Confusion matrix transcribed from the paper's Fig. 4 (row-normalised percentages,
# rows = actual thick ice / thin ice / water, columns = predicted in the same order).
# The figure's open-water diagonal reads 60.35 (the text says 60.25); the figure value is
# kept here so the rows sum to 100.
CM_PAPER = np.array([
    [98.39, 1.35, 0.26],
    [19.25, 73.80, 6.95],
    [5.17, 34.48, 60.35],
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


def _try_import_seaborn():
    try:
        import seaborn as sns
        return sns
    except ImportError:
        return None


def _savefig(fig, output_dir, name, formats=("png",)):
    """Save figure in requested formats and return paths."""
    paths = []
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path)
        paths.append(path)
        logger.info("  Saved %s", path)
    plt.close(fig)
    return paths


def _load_csv(path):
    """Load CSV with pandas, return None on failure."""
    pd = _try_import_pandas()
    if pd is None:
        logger.warning("pandas not available — skipping CSV load")
        return None
    if path is None or not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def _filter_track(df, track_params):
    """Filter a DataFrame to a specific track (track id, else day+beam) and lon range."""
    if df is None or len(df) == 0:
        return None
    d = df.copy()

    track_col = "track" if "track" in d.columns else ("track_x" if "track_x" in d.columns else None)
    if track_col is not None:
        t = d[track_col].astype(str)
        if (t == track_params["track_id"]).any():
            d = d[t == track_params["track_id"]]
        else:  # legacy beam-only ids
            d = d[t.str.lower() == track_params["beam"]]
            if "day" in d.columns:
                d = d[d.day == track_params["day"]]
    elif "day" in d.columns:
        d = d[d.day == track_params["day"]]

    # Filter by lon range (skip if lon not available)
    if "lon" in d.columns:
        d = d[(d.lon >= track_params["lon1"]) & (d.lon <= track_params["lon2"])]

    if len(d) == 0:
        return None
    sort_col = "lon" if "lon" in d.columns else ("x_atc" if "x_atc" in d.columns else d.columns[0])
    return d.sort_values(sort_col).reset_index(drop=True)


def _filter_atl07_track(df, track_params):
    """Filter ATL07 DataFrame to a specific track."""
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    if "lon" in d.columns:
        d = d[(d.lon >= track_params["lon1"]) & (d.lon <= track_params["lon2"])]
    if len(d) == 0:
        return None
    return d.sort_values("lon").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def _render_table(table_data, output_dir):
    """Render a table as a matplotlib figure and LaTeX string."""
    title = table_data["title"]
    columns = table_data["columns"]
    rows = table_data["rows"]
    n_cols = len(columns)

    fig_width = max(IEEE_DOUBLE_COL, n_cols * 0.9)
    fig_height = max(1.5, 0.35 * (len(rows) + 1) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, len(rows) + 1):
        for j in range(n_cols):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#D9E2F3")
            else:
                cell.set_facecolor("white")

    safe_name = title.split(":")[0].strip().replace(" ", "_").lower()
    paths = _savefig(fig, output_dir, safe_name)

    # Generate LaTeX
    col_fmt = "c" * n_cols
    latex_lines = [
        f"% {title}",
        f"\\begin{{table}}[htbp]",
        f"  \\caption{{{title.split(': ', 1)[-1]}}}",
        f"  \\centering",
        f"  \\begin{{tabular}}{{{col_fmt}}}",
        f"    \\hline",
        f"    {' & '.join(columns)} \\\\",
        f"    \\hline",
    ]
    for row in rows:
        latex_lines.append(f"    {' & '.join(row)} \\\\")
    latex_lines += [
        f"    \\hline",
        f"  \\end{{tabular}}",
        f"\\end{{table}}",
    ]
    latex_str = "\n".join(latex_lines)

    tex_path = os.path.join(output_dir, f"{safe_name}.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    logger.info("  Saved %s", tex_path)
    paths.append(tex_path)

    return paths


def generate_table(table_num, output_dir):
    """Generate a specific table by number (1-5)."""
    if table_num not in ALL_TABLES:
        logger.error("Unknown table number: %d (valid: 1-5)", table_num)
        return []
    logger.info("Generating Table %d", table_num)
    return _render_table(ALL_TABLES[table_num], output_dir)


# ---------------------------------------------------------------------------
# Figure 4: Confusion matrix
# ---------------------------------------------------------------------------

def fig_4_confusion_matrix(output_dir, predictions_csv=None, test_predictions_csv=None):
    """3x3 confusion matrix heatmap (row-normalized percentages).

    Prefers the held-out test-split predictions written by train_lstm.py; falls back to the
    full predictions CSV (which includes training rows) and finally to the paper's values.
    """
    logger.info("Generating Figure 4: Confusion Matrix")
    if test_predictions_csv is not None and os.path.isfile(test_predictions_csv):
        predictions_csv = test_predictions_csv
        logger.info("  Using held-out test predictions %s", test_predictions_csv)
    elif predictions_csv is not None:
        logger.warning("  Using full predictions (includes training rows) for the confusion matrix")
    pd = _try_import_pandas()
    sns = _try_import_seaborn()

    cm_pct = CM_PAPER  # default to hardcoded

    # Compute from data if available
    if predictions_csv is not None and pd is not None:
        df = _load_csv(predictions_csv)
        if df is not None and "label" in df.columns and "pred_label" in df.columns:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(df["label"], df["pred_label"], labels=[0, 1, 2])
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct = cm.astype(float) / row_sums * 100
            logger.info("  Computed confusion matrix from predictions CSV")

    labels = ["thick ice", "thin ice", "water"]
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL + 0.5, IEEE_SINGLE_COL + 0.5))

    if sns is not None:
        if pd is not None:
            cm_df = pd.DataFrame(cm_pct, index=labels,
                                 columns=[f"Pred. {l}" for l in labels])
            sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                        cbar_kws={"label": "%"})
        else:
            sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                        xticklabels=[f"Pred. {l}" for l in labels],
                        yticklabels=labels, cbar_kws={"label": "%"})
    else:
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"Pred. {l}" for l in labels], rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(3):
            for j in range(3):
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, f"{cm_pct[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=9)
        fig.colorbar(im, ax=ax, label="%")

    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix (%)")

    return _savefig(fig, output_dir, "fig_04_confusion_matrix")


# ---------------------------------------------------------------------------
# Figure 5: Horovod training (4 subplots)
# ---------------------------------------------------------------------------

def fig_5_horovod_training(output_dir):
    """4-subplot chart using Table IV data: speedup, time, data/s, time/epoch."""
    logger.info("Generating Figure 5: Horovod Training Scalability")

    gpus = np.array([1, 2, 4, 6, 8])
    time_total = np.array([280.72, 143.22, 73.68, 49.42, 38.72])
    time_epoch = np.array([5.5, 2.778, 1.45, 0.97, 0.79])
    data_per_s = np.array([585.88, 1160.81, 2229.56, 3330.03, 4248.56])
    speedup = np.array([1.00, 1.96, 3.81, 5.68, 7.25])
    ideal_speedup = gpus.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(IEEE_DOUBLE_COL, 5))

    # (a) Speedup bar chart
    ax = axes[0, 0]
    x = np.arange(len(gpus))
    w = 0.35
    ax.bar(x - w / 2, speedup, w, label="Actual", color="#4472C4")
    ax.bar(x + w / 2, ideal_speedup, w, label="Ideal (linear)", color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels(gpus)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup")
    ax.set_title("(a) Training Speedup")
    ax.legend()
    ax.grid(axis="y", linestyle=":", linewidth=0.5)

    # (b) Total training time
    ax = axes[0, 1]
    ax.plot(gpus, time_total, "o-", color="#4472C4", linewidth=2, markersize=6)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Total Time (s)")
    ax.set_title("(b) Total Training Time")
    ax.grid(linestyle=":", linewidth=0.5)

    # (c) Data throughput
    ax = axes[1, 0]
    ax.plot(gpus, data_per_s, "s-", color="#70AD47", linewidth=2, markersize=6)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Samples/sec")
    ax.set_title("(c) Data Throughput per Epoch")
    ax.grid(linestyle=":", linewidth=0.5)

    # (d) Time per epoch
    ax = axes[1, 1]
    ax.plot(gpus, time_epoch, "^-", color="#ED7D31", linewidth=2, markersize=6)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Time/Epoch (s)")
    ax.set_title("(d) Time per Epoch")
    ax.grid(linestyle=":", linewidth=0.5)

    fig.tight_layout()
    return _savefig(fig, output_dir, "fig_05_horovod_training")


# ---------------------------------------------------------------------------
# Figures 6 & 7: Classification comparison (ATL03 vs ATL07)
# ---------------------------------------------------------------------------

def _fig_classification_comparison(output_dir, fig_num, track_key,
                                   predictions_csv=None, atl07_csv=None):
    """Two stacked subplots: (a) ATL03 classification, (b) ATL07 classification."""
    tp = TRACK_PARAMS[track_key]
    logger.info("Generating Figure %d: Classification — %s", fig_num, tp["track_id"])

    pred_df = _load_csv(predictions_csv)
    atl07_df = _load_csv(atl07_csv)

    track_data = _filter_track(pred_df, tp)
    atl07_data = _filter_atl07_track(atl07_df, tp)

    if track_data is None and atl07_data is None:
        logger.warning("  No data available for Figure %d — skipping", fig_num)
        return []

    fig, axes = plt.subplots(2, 1, figsize=(IEEE_DOUBLE_COL, 5), sharex=True)

    # (a) ATL03 classification
    ax = axes[0]
    x_col = "lon" if track_data is not None and "lon" in track_data.columns else "x_atc"
    x_label = "Longitude (deg)" if x_col == "lon" else "Along-track distance (m)"
    if track_data is not None:
        for label_val, color, name in [(0, "C0", "thick ice"),
                                        (1, "C1", "thin ice"),
                                        (2, "C2", "open water")]:
            subset = track_data[track_data.pred_label == label_val]
            field = "h_cor_mean"
            if field in subset.columns and x_col in subset.columns:
                ax.scatter(subset[x_col], subset[field], s=2, c=color, label=name)
    ax.set_ylabel("ATL03 elevation (m)")
    ax.set_ylim(-0.5, 3.5)
    ax.legend(loc="upper right", markerscale=4)
    ax.set_title("(a) ATL03 Classification")
    ax.grid(linestyle=":", linewidth=0.5)

    # (b) ATL07 classification
    ax = axes[1]
    if atl07_data is not None:
        # ATL07 label mapping: 2=thick, 1=thin, 0=water
        for label_val, color, name in [(2, "C0", "thick ice"),
                                        (1, "C1", "thin ice"),
                                        (0, "C2", "open water")]:
            subset = atl07_data[atl07_data.label == label_val]
            if "height" in subset.columns:
                ax.scatter(subset.lon, subset.height, s=2, c=color, label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel("ATL07 elevation (m)")
    ax.set_ylim(-0.5, 3.5)
    ax.legend(loc="upper right", markerscale=4)
    ax.set_title("(b) ATL07 Youngs Classification")
    ax.grid(linestyle=":", linewidth=0.5)

    fig.suptitle(f"Track: {tp['track_id']}", fontsize=9, y=1.01)
    fig.tight_layout()
    return _savefig(fig, output_dir, f"fig_{fig_num:02d}_classification_{track_key}")


def fig_6_classification(output_dir, predictions_csv=None, atl07_csv=None):
    return _fig_classification_comparison(output_dir, 6, "track1",
                                          predictions_csv, atl07_csv)


def fig_7_classification(output_dir, predictions_csv=None, atl07_csv=None):
    return _fig_classification_comparison(output_dir, 7, "track2",
                                          predictions_csv, atl07_csv)


# ---------------------------------------------------------------------------
# Figures 8 & 9: Sea surface comparison
# ---------------------------------------------------------------------------

def _fig_sea_surface(output_dir, fig_num, track_key, freeboard_csv=None, atl07_csv=None):
    """Two subplots: (a) 4 sea surface methods, (b) ATL03 vs ATL07 sea surface.

    Panel (b) draws the ATL07 ``h_ref`` from the reference CSV (Koo format) when one is
    given. The notebook instead merged ``h_ref`` onto the ATL03 rows first; the workflow
    never produces that merged column, so the reference frame is plotted directly.
    """
    tp = TRACK_PARAMS[track_key]
    logger.info("Generating Figure %d: Sea Surface — %s", fig_num, tp["track_id"])

    fb_df = _load_csv(freeboard_csv)
    track_data = _filter_track(fb_df, tp)
    atl07_data = _filter_atl07_track(_load_csv(atl07_csv), tp)

    if track_data is None:
        logger.warning("  No data available for Figure %d — skipping", fig_num)
        return []

    fig, axes = plt.subplots(2, 1, figsize=(IEEE_DOUBLE_COL, 5), sharex=True)

    # (a) Four sea surface methods
    ax = axes[0]
    x_col = "x_atc" if "x_atc" in track_data.columns else "lon"
    methods = [
        ("sea_surf_min_elev", "C1", "Minimum Elevation"),
        ("sea_surf_avg_elev", "C2", "Average Elevation"),
        ("sea_surf_min_dist", "C3", "Nearest Minimum Elevation"),
        ("new_h_ref_smooth", "C0", "NASA Sea Surface Formula (smoothed)"),
    ]
    if "new_h_ref_smooth" not in track_data.columns:
        methods[-1] = ("new_h_ref", "C0", "NASA Sea Surface Formula")
    for col, color, label in methods:
        if col in track_data.columns:
            ax.scatter(track_data[x_col], track_data[col], s=2, c=color, label=label)
    ax.set_ylabel("Sea Surface Elevation (m)")
    ax.set_ylim(-0.4, 0.4)
    ax.legend(loc="upper right", markerscale=4, fontsize=7)
    ax.set_title("(a) Sea Surface Estimation Methods")
    ax.grid(linestyle=":", linewidth=0.5)

    # (b) ATL03 vs ATL07 sea surface
    ax = axes[1]
    ss_col = "new_h_ref_smooth" if "new_h_ref_smooth" in track_data.columns else "new_h_ref"
    if ss_col in track_data.columns:
        ax.scatter(track_data[x_col], track_data[ss_col], s=2, label="ATL03 Sea Surface")
    if "h_ref" in track_data.columns:  # notebook-style merged column, if present
        ax.scatter(track_data[x_col], track_data["h_ref"], s=2, label="ATL07 Sea Surface")
    elif atl07_data is not None and "h_ref" in atl07_data.columns:
        atl07_x = "x" if x_col == "x_atc" and "x" in atl07_data.columns else (
            "lon" if "lon" in atl07_data.columns else None)
        if atl07_x is not None:
            ax.scatter(atl07_data[atl07_x], atl07_data["h_ref"], s=2, label="ATL07 Sea Surface")
        else:
            logger.warning("  ATL07 CSV has no x/lon column for Figure %d(b)", fig_num)
    ax.set_xlabel("Along-Track Distance (m)" if x_col == "x_atc" else "Longitude (deg)")
    ax.set_ylabel("Sea Surface Elevation (m)")
    ax.set_ylim(-0.4, 0.4)
    ax.legend(loc="upper right", markerscale=4)
    ax.set_title("(b) ATL03 vs ATL07 Sea Surface")
    ax.grid(linestyle=":", linewidth=0.5)

    fig.suptitle(f"Track: {tp['track_id']}", fontsize=9, y=1.01)
    fig.tight_layout()
    return _savefig(fig, output_dir, f"fig_{fig_num:02d}_sea_surface_{track_key}")


def fig_8_sea_surface(output_dir, freeboard_csv=None, atl07_csv=None):
    return _fig_sea_surface(output_dir, 8, "track1", freeboard_csv, atl07_csv)


def fig_9_sea_surface(output_dir, freeboard_csv=None, atl07_csv=None):
    return _fig_sea_surface(output_dir, 9, "track2", freeboard_csv, atl07_csv)


# ---------------------------------------------------------------------------
# Figures 10 & 11: Freeboard comparison (4 subplots)
# ---------------------------------------------------------------------------

def _fig_freeboard_comparison(output_dir, fig_num, track_key,
                               freeboard_csv=None, atl07_csv=None):
    """4 subplots: (a) ATL03 freeboard scatter, (b) ATL07 freeboard scatter,
    (c) freeboard distribution histogram, (d) point density."""
    tp = TRACK_PARAMS[track_key]
    logger.info("Generating Figure %d: Freeboard Comparison — %s", fig_num, tp["track_id"])

    fb_df = _load_csv(freeboard_csv)
    atl07_df = _load_csv(atl07_csv)

    track_data = _filter_track(fb_df, tp)
    atl07_data = _filter_atl07_track(atl07_df, tp)

    if track_data is None and atl07_data is None:
        logger.warning("  No data available for Figure %d — skipping", fig_num)
        return []

    fig, axes = plt.subplots(2, 2, figsize=(IEEE_DOUBLE_COL, 6))

    x_col = "lon" if track_data is not None and "lon" in track_data.columns else "x_atc"
    x_label = "Longitude (deg)" if x_col == "lon" else "Along-track distance (m)"

    # (a) ATL03 freeboard scatter
    ax = axes[0, 0]
    fb_col = None
    for c in ["freeboard_new_h_ref_smooth", "freeboard_new_h_ref", "freeboard_min_elev"]:
        if track_data is not None and c in track_data.columns:
            fb_col = c
            break
    if track_data is not None and fb_col is not None and x_col in track_data.columns:
        ice = track_data[track_data.pred_label <= 1]
        ax.scatter(ice[x_col], ice[fb_col], s=2, c="C0")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Freeboard (m)")
    ax.set_ylim(-0.5, 3.5)
    ax.set_title("(a) ATL03 Freeboard")
    ax.grid(linestyle=":", linewidth=0.5)

    # (b) ATL07 freeboard scatter
    ax = axes[0, 1]
    if atl07_data is not None and "freeboard" in atl07_data.columns:
        ice7 = atl07_data[atl07_data.label >= 1]
        ax.scatter(ice7.lon, ice7.freeboard, s=2, c="C1")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Freeboard (m)")
    ax.set_ylim(-0.5, 3.5)
    ax.set_title("(b) ATL07 Freeboard")
    ax.grid(linestyle=":", linewidth=0.5)

    # (c) Freeboard distribution histogram
    ax = axes[1, 0]
    if track_data is not None and fb_col is not None:
        ice = track_data[track_data.pred_label <= 1]
        vals = ice[fb_col].dropna().clip(0, 3.5)
        if len(vals) > 0:
            ax.hist(vals, bins=50, histtype="step", ec="blue", linewidth=2,
                    label="ATL03")
    if atl07_data is not None and "freeboard" in atl07_data.columns:
        ice7 = atl07_data[atl07_data.label >= 1]
        vals7 = ice7.freeboard.dropna().clip(0, 3.5)
        if len(vals7) > 0:
            ax.hist(vals7, bins=50, histtype="step", ec="red", linewidth=2,
                    label="ATL07")
        if "freeboard_10" in ice7.columns:
            vals10 = ice7.freeboard_10.dropna().clip(0, 3.5)
            if len(vals10) > 0:
                ax.hist(vals10, bins=50, histtype="step", ec="green", linewidth=2,
                        label="ATL10")
    ax.set_xlabel("Freeboard (m)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 3.5)
    ax.legend(fontsize=7)
    ax.set_title("(c) Freeboard Distribution")

    # (d) Point density comparison
    ax = axes[1, 1]
    if track_data is not None and fb_col is not None and x_col in track_data.columns:
        ice = track_data[track_data.pred_label <= 1]
        if len(ice) > 0:
            ax.hist(ice[x_col], bins=40, alpha=0.5, edgecolor="black", label="ATL03")
    if atl07_data is not None:
        ice7 = atl07_data[atl07_data.label >= 1]
        atl07_x = "lon" if "lon" in atl07_data.columns else "x"
        if len(ice7) > 0:
            ax.hist(ice7[atl07_x], bins=40, alpha=0.5, color="green", edgecolor="black",
                    label="ATL07")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.set_title("(d) Point Density")

    fig.suptitle(f"Track: {tp['track_id']}", fontsize=9, y=1.01)
    fig.tight_layout()
    return _savefig(fig, output_dir, f"fig_{fig_num:02d}_freeboard_{track_key}")


def fig_10_freeboard(output_dir, freeboard_csv=None, atl07_csv=None):
    return _fig_freeboard_comparison(output_dir, 10, "track1", freeboard_csv, atl07_csv)


def fig_11_freeboard(output_dir, freeboard_csv=None, atl07_csv=None):
    return _fig_freeboard_comparison(output_dir, 11, "track2", freeboard_csv, atl07_csv)



# ---------------------------------------------------------------------------
# Figures 12-15: this run's own diagnostics (not in the paper)
# ---------------------------------------------------------------------------

def _load_metrics(metrics_json):
    if not metrics_json or not os.path.isfile(metrics_json):
        return None
    import json
    with open(metrics_json) as f:
        return json.load(f)


def fig_12_training_curves(output_dir, metrics_json=None):
    """Accuracy and loss per epoch for train and validation."""
    logger.info("Generating Figure 12: Training Curves")
    m = _load_metrics(metrics_json)
    if m is None or not m.get("history"):
        logger.warning("  No training_metrics.json with history - skipping Figure 12")
        return []

    h = m["history"]
    epochs = np.arange(1, len(h.get("accuracy", [])) + 1)
    if len(epochs) == 0:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.8))
    ax = axes[0]
    if "accuracy" in h:
        ax.plot(epochs, h["accuracy"], label="train", color="C0")
    if "val_accuracy" in h:
        ax.plot(epochs, h["val_accuracy"], label="validation", color="C1")
    if "test_accuracy" in m:
        ax.axhline(m["test_accuracy"], ls="--", lw=1, color="C3",
                   label=f"test {m['test_accuracy'] * 100:.2f}%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Accuracy")
    ax.legend(fontsize=7)
    ax.grid(linestyle=":", linewidth=0.5)

    ax = axes[1]
    if "loss" in h:
        ax.plot(epochs, h["loss"], label="train", color="C0")
    if "val_loss" in h:
        ax.plot(epochs, h["val_loss"], label="validation", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal loss")
    ax.set_title("(b) Loss")
    ax.legend(fontsize=7)
    ax.grid(linestyle=":", linewidth=0.5)

    fig.tight_layout()
    return _savefig(fig, output_dir, "fig_12_training_curves")


def fig_13_paper_comparison(output_dir, metrics_json=None):
    """This run's held-out metrics against the values reported in the paper."""
    logger.info("Generating Figure 13: Comparison with the paper")
    m = _load_metrics(metrics_json)
    if m is None:
        logger.warning("  No training_metrics.json - skipping Figure 13")
        return []

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 3.0))

    # (a) Overall metrics
    names = ["Accuracy", "Precision", "Recall", "F1"]
    ours = [m.get("test_accuracy", 0) * 100, m.get("test_precision", 0) * 100,
            m.get("test_recall", 0) * 100, m.get("test_f1", 0) * 100]
    paper = [PAPER_METRICS["LSTM"][n] for n in names]
    mlp = [PAPER_METRICS["MLP"][n] for n in names]

    x = np.arange(len(names))
    w = 0.27
    ax = axes[0]
    ax.bar(x - w, mlp, w, label="Paper MLP", color="#A5A5A5")
    ax.bar(x, paper, w, label="Paper LSTM", color="#ED7D31")
    ax.bar(x + w, ours, w, label="This run", color="#4472C4")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(85, 100)
    ax.set_title("(a) Overall (held-out test split)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    for xi, v in zip(x + w, ours):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=6)

    # (b) Per-class accuracy
    classes = ["thick ice", "thin ice", "open water"]
    paper_pc = [PAPER_PER_CLASS[c] for c in classes]
    cm = m.get("test_confusion_matrix")
    if cm:
        ours_pc = [100.0 * cm[i][i] / max(1, sum(cm[i])) for i in range(3)]
    else:
        ours_pc = [0, 0, 0]

    x = np.arange(len(classes))
    w = 0.38
    ax = axes[1]
    ax.bar(x - w / 2, paper_pc, w, label="Paper", color="#ED7D31")
    ax.bar(x + w / 2, ours_pc, w, label="This run", color="#4472C4")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Per class")
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    for xi, v in zip(x - w / 2, paper_pc):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=6)
    for xi, v in zip(x + w / 2, ours_pc):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=6)

    fig.tight_layout()
    return _savefig(fig, output_dir, "fig_13_paper_comparison")


def fig_14_freeboard_all_tracks(output_dir, freeboard_csv=None):
    """Freeboard distribution and along-track median for every processed track."""
    logger.info("Generating Figure 14: Freeboard across all tracks")
    pd = _try_import_pandas()
    df = _load_csv(freeboard_csv)
    if pd is None or df is None:
        logger.warning("  No freeboard data - skipping Figure 14")
        return []

    fb_col = next((c for c in ("freeboard_new_h_ref_smooth", "freeboard_new_h_ref")
                   if c in df.columns), None)
    track_col = next((c for c in ("track", "track_x") if c in df.columns), None)
    if fb_col is None or track_col is None:
        logger.warning("  Freeboard/track column missing - skipping Figure 14")
        return []

    ice = df[df.pred_label <= 1] if "pred_label" in df.columns else df
    tracks = sorted(ice[track_col].astype(str).unique())

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 3.0))

    ax = axes[0]
    for i, t in enumerate(tracks):
        vals = ice.loc[ice[track_col].astype(str) == t, fb_col].dropna().clip(-0.5, 3.5)
        if len(vals):
            ax.hist(vals, bins=60, histtype="step", linewidth=1.5,
                    color=f"C{i}", label=t.replace("_", " "))
    ax.set_xlabel("Freeboard (m)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 2.5)
    ax.set_title("(a) Distribution per track")
    ax.legend(fontsize=6)
    ax.grid(linestyle=":", linewidth=0.5)

    ax = axes[1]
    medians = []
    for t in tracks:
        vals = ice.loc[ice[track_col].astype(str) == t, fb_col].dropna()
        medians.append(vals.median() if len(vals) else np.nan)
    y = np.arange(len(tracks))
    ax.barh(y, medians, color=[f"C{i}" for i in range(len(tracks))])
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("_", " ") for t in tracks], fontsize=6)
    ax.set_xlabel("Median freeboard (m)")
    ax.set_title("(b) Median per track")
    ax.grid(axis="x", linestyle=":", linewidth=0.5)
    for yi, v in zip(y, medians):
        if np.isfinite(v):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6)

    fig.tight_layout()
    return _savefig(fig, output_dir, "fig_14_freeboard_all_tracks")


def fig_15_class_elevation(output_dir, predictions_csv=None):
    """Elevation distribution and class balance of the classification output."""
    logger.info("Generating Figure 15: Elevation by predicted class")
    pd = _try_import_pandas()
    df = _load_csv(predictions_csv)
    if pd is None or df is None or "pred_label" not in df.columns:
        logger.warning("  No predictions - skipping Figure 15")
        return []

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 3.0))
    names = {0: "thick ice", 1: "thin ice", 2: "open water"}

    ax = axes[0]
    for lab, color in ((0, "C0"), (1, "C1"), (2, "C2")):
        vals = df.loc[df.pred_label == lab, "h_cor_mean"].dropna().clip(-0.5, 3.5)
        if len(vals):
            ax.hist(vals, bins=80, histtype="step", linewidth=1.5, color=color,
                    label=names[lab])
    ax.set_xlabel("ATL03 elevation above MSS (m)")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title("(a) Elevation by predicted class")
    ax.legend(fontsize=7)
    ax.grid(linestyle=":", linewidth=0.5)

    ax = axes[1]
    counts = [int((df.pred_label == l).sum()) for l in (0, 1, 2)]
    total = max(1, sum(counts))
    bars = ax.bar([names[l] for l in (0, 1, 2)], counts, color=["C0", "C1", "C2"])
    ax.set_ylabel("Segments")
    ax.set_yscale("log")
    ax.set_title("(b) Class balance")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n{100 * c / total:.1f}%",
                ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    return _savefig(fig, output_dir, "fig_15_class_elevation")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures and tables for sea ice classification paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tables-only --output-dir figures/
  %(prog)s --figure 5 --output-dir figures/
  %(prog)s --all --predictions pred.csv --freeboard fb.csv --atl07-csv atl07_*.csv -o figures/
  %(prog)s --all --output-tar paper_figures.tar.gz --output-dir .
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Generate all figures and tables")
    group.add_argument("--tables-only", action="store_true",
                       help="Generate only tables (no data files needed)")
    group.add_argument("--figure", type=int, choices=range(4, 16), metavar="N",
                       help="Generate a specific figure (4-15)")
    group.add_argument("--table", type=int, choices=range(1, 6), metavar="N",
                       help="Generate a specific table (1-5)")

    parser.add_argument("--predictions", default=None,
                        help="LSTM predictions CSV (for Figs 4, 6, 7)")
    parser.add_argument("--test-predictions", default=None,
                        help="Held-out test-split predictions from train_lstm.py (for Fig 4)")
    parser.add_argument("--metrics", default=None,
                        help="training_metrics.json from train_lstm.py (for Figs 12, 13)")
    parser.add_argument("--freeboard", nargs="+", default=None,
                        help="Freeboard output CSV(s) (for Figs 8-11, 14)")
    parser.add_argument("--atl07-csv", nargs="+", default=None,
                        help="ATL07 comparison CSV(s) (for Figs 6-11)")
    parser.add_argument("--atl10-csv", nargs="+", default=None,
                        help="ATL10 comparison CSV(s); merged into ATL07 as freeboard_10 column")
    parser.add_argument("--output-dir", "-o", default="figures",
                        help="Output directory (default: figures/)")
    parser.add_argument("--output-tar", default=None,
                        help="If set, bundle all outputs into this tar.gz file")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Merge multiple ATL07 CSVs into one
    atl07_path = None
    atl07_df = None
    if args.atl07_csv:
        pd = _try_import_pandas()
        if pd is not None:
            dfs = [_load_csv(f) for f in args.atl07_csv]
            dfs = [d for d in dfs if d is not None]
            if dfs:
                atl07_df = pd.concat(dfs, ignore_index=True)
                if len(dfs) > 1:
                    atl07_path = os.path.join(args.output_dir, "_atl07_merged.csv")
                    atl07_df.to_csv(atl07_path, index=False)
                    logger.info("Merged %d ATL07 files (%d rows) → %s",
                                len(dfs), len(atl07_df), atl07_path)
                else:
                    atl07_path = args.atl07_csv[0]
                    atl07_df = dfs[0]
            else:
                atl07_df = None
        else:
            atl07_df = None

    # Merge multiple ATL10 CSVs into one
    atl10_df = None
    if args.atl10_csv:
        pd = _try_import_pandas()
        if pd is not None:
            dfs = [_load_csv(f) for f in args.atl10_csv]
            dfs = [d for d in dfs if d is not None]
            if dfs:
                atl10_df = pd.concat(dfs, ignore_index=True)
                logger.info("Merged %d ATL10 files (%d rows)", len(dfs), len(atl10_df))

    # Merge ATL10 freeboard into ATL07 as freeboard_10 column
    if atl10_df is not None and atl07_df is not None:
        pd = _try_import_pandas()
        if (pd is not None
                and "freeboard" in atl10_df.columns
                and "freeboard_10" not in atl07_df.columns):
            atl07_df = pd.merge_asof(
                atl07_df.sort_values("lon"),
                atl10_df[["lon", "freeboard"]].sort_values("lon").rename(
                    columns={"freeboard": "freeboard_10"}),
                on="lon", direction="nearest",
            )
            atl07_path = os.path.join(args.output_dir, "_atl07_with_atl10.csv")
            atl07_df.to_csv(atl07_path, index=False)
            logger.info("Merged ATL10 into ATL07 → %s", atl07_path)

    # Merge multiple freeboard files into one path for downstream functions
    freeboard_merged = None
    if args.freeboard:
        pd = _try_import_pandas()
        if pd is not None:
            dfs = [_load_csv(f) for f in args.freeboard]
            dfs = [d for d in dfs if d is not None]
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                freeboard_merged = os.path.join(args.output_dir, "_freeboard_merged.csv")
                merged.to_csv(freeboard_merged, index=False)
                logger.info("Merged %d freeboard files (%d rows) → %s",
                            len(dfs), len(merged), freeboard_merged)
        if freeboard_merged is None and len(args.freeboard) == 1:
            freeboard_merged = args.freeboard[0]

    all_outputs = []

    if args.table:
        all_outputs.extend(generate_table(args.table, args.output_dir))

    elif args.tables_only:
        for t in range(1, 6):
            all_outputs.extend(generate_table(t, args.output_dir))

    elif args.figure:
        fig_funcs = {
            4: lambda: fig_4_confusion_matrix(args.output_dir, args.predictions, args.test_predictions),
            5: lambda: fig_5_horovod_training(args.output_dir),
            6: lambda: fig_6_classification(args.output_dir, args.predictions, atl07_path),
            7: lambda: fig_7_classification(args.output_dir, args.predictions, atl07_path),
            8: lambda: fig_8_sea_surface(args.output_dir, freeboard_merged, atl07_path),
            9: lambda: fig_9_sea_surface(args.output_dir, freeboard_merged, atl07_path),
            10: lambda: fig_10_freeboard(args.output_dir, freeboard_merged, atl07_path),
            11: lambda: fig_11_freeboard(args.output_dir, freeboard_merged, atl07_path),
            12: lambda: fig_12_training_curves(args.output_dir, args.metrics),
            13: lambda: fig_13_paper_comparison(args.output_dir, args.metrics),
            14: lambda: fig_14_freeboard_all_tracks(args.output_dir, freeboard_merged),
            15: lambda: fig_15_class_elevation(args.output_dir, args.predictions),
        }
        all_outputs.extend(fig_funcs[args.figure]())

    elif args.all:
        # Tables first (always work)
        for t in range(1, 6):
            all_outputs.extend(generate_table(t, args.output_dir))

        # Figures
        all_outputs.extend(fig_4_confusion_matrix(args.output_dir, args.predictions, args.test_predictions))
        all_outputs.extend(fig_5_horovod_training(args.output_dir))
        all_outputs.extend(fig_6_classification(args.output_dir, args.predictions, atl07_path))
        all_outputs.extend(fig_7_classification(args.output_dir, args.predictions, atl07_path))
        all_outputs.extend(fig_8_sea_surface(args.output_dir, freeboard_merged, atl07_path))
        all_outputs.extend(fig_9_sea_surface(args.output_dir, freeboard_merged, atl07_path))
        all_outputs.extend(fig_10_freeboard(args.output_dir, freeboard_merged, atl07_path))
        all_outputs.extend(fig_11_freeboard(args.output_dir, freeboard_merged, atl07_path))
        all_outputs.extend(fig_12_training_curves(args.output_dir, args.metrics))
        all_outputs.extend(fig_13_paper_comparison(args.output_dir, args.metrics))
        all_outputs.extend(fig_14_freeboard_all_tracks(args.output_dir, freeboard_merged))
        all_outputs.extend(fig_15_class_elevation(args.output_dir, args.predictions))

    # Bundle into tar if requested (for workflow output)
    if args.output_tar and all_outputs:
        tar_path = args.output_tar
        if not os.path.dirname(tar_path):  # bare filename -> place next to the outputs
            tar_path = os.path.join(args.output_dir, tar_path)
        with tarfile.open(tar_path, "w:gz") as tar:
            for path in all_outputs:
                if os.path.isfile(path):
                    tar.add(path, arcname=os.path.basename(path))
        logger.info("Bundled %d files into %s", len(all_outputs), tar_path)

    logger.info("Done — generated %d output files", len(all_outputs))


if __name__ == "__main__":
    main()
