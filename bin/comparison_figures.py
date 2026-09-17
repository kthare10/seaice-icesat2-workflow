#!/usr/bin/env python3
"""
Render this workflow's results as standalone single panels that match the framing
of the paper's figures, for the side-by-side tables in PAPER_COMPARISON.md.

paper_figures.py produces multi-panel composites intended to stand alone. Those do
not crop cleanly (shared x-axes leave the upper panel without tick labels), so the
comparison panels are drawn here from the same data instead.

Usage:
    python bin/comparison_figures.py --freeboard results5/freeboard_*.csv \
        --output-dir figures/run
"""

import argparse
import glob
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# The two tracks the paper plots, with the longitude windows used in its figures.
TRACKS = {
    "nov04_gt2r": {"track": "20191104195311_05940510_gt2r", "lon": (-170.5, -170.0)},
    "nov26_gt2r": {"track": "20191126182014_09290510_gt2r", "lon": (-162.96, -162.70)},
}
CLASSES = [(0, "C0", "thick ice"), (1, "C1", "thin ice"), (2, "C2", "open water")]


def _subset(df, key, use_lon=True):
    spec = TRACKS[key]
    track_col = "track" if "track" in df.columns else "track_x"
    d = df[df[track_col].astype(str) == spec["track"]]
    if use_lon and "lon" in d.columns:
        d = d[(d.lon >= spec["lon"][0]) & (d.lon <= spec["lon"][1])]
    return d.sort_values("x_atc")


def classification_panel(df, key, out):
    """Paper Figs. 6a / 7a: ATL03 elevation coloured by predicted class."""
    d = _subset(df, key)
    if len(d) == 0:
        logger.warning("  no rows for %s - skipping %s", key, out)
        return None
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for lab, color, name in CLASSES:
        s = d[d.pred_label == lab]
        if len(s):
            ax.scatter(s.lon, s.h_cor_mean, s=2, c=color, label=name)
    ax.set_xlabel("along track long (degree)")
    ax.set_ylabel("ATL03 elevation (m)")
    ax.set_ylim(-0.5, 3.5)
    ax.set_title("ATL03 Classification")
    ax.legend(loc="upper right", markerscale=4)
    ax.grid(linestyle=":", linewidth=0.5)
    fig.savefig(out)
    plt.close(fig)
    logger.info("  wrote %s (%d points)", out, len(d))
    return out


def sea_surface_panel(df, key, out):
    """Paper Figs. 8a / 9a: the four local sea surface estimates along track."""
    d = _subset(df, key, use_lon=False)
    if len(d) == 0:
        logger.warning("  no rows for %s - skipping %s", key, out)
        return None
    fig, ax = plt.subplots(figsize=(10, 3.4))
    series = [("sea_surf_min_elev", "green", "Minimum Elevation"),
              ("sea_surf_avg_elev", "orange", "Average Elevation"),
              ("sea_surf_min_dist", "red", "Nearest Minimum Elevation"),
              ("new_h_ref_smooth", "C0", "NASA's Sea Surface Formula")]
    # The paper plots these relative to the track's own mean sea surface; centring
    # on the median of the NASA estimate reproduces that framing.
    offset = d["new_h_ref_smooth"].median()
    for col, color, name in series:
        if col in d.columns:
            ax.scatter(d.x_atc, d[col] - offset, s=2, c=color, label=name)
    ax.set_xlabel("Along-Track Distance (m)")
    ax.set_ylabel("Sea Surface Elevation (m)")
    ax.set_ylim(-0.4, 0.4)
    ax.legend(loc="upper right", markerscale=4, fontsize=9)
    ax.grid(linestyle=":", linewidth=0.5)
    fig.savefig(out)
    plt.close(fig)
    logger.info("  wrote %s (%d points)", out, len(d))
    return out


def freeboard_panel(df, key, out):
    """Paper Figs. 10a / 11a: freeboard of ice segments along track."""
    d = _subset(df, key)
    d = d[d.pred_label <= 1]
    if len(d) == 0:
        logger.warning("  no rows for %s - skipping %s", key, out)
        return None
    col = ("freeboard_new_h_ref_smooth" if "freeboard_new_h_ref_smooth" in d.columns
           else "freeboard_new_h_ref")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.scatter(d.lon, d[col], s=2, c="C0", label="freeboard 2m ATL03")
    ax.set_xlabel("along track lat (degree)")
    ax.set_ylabel("elevation (m)")
    ax.set_ylim(-0.5, 3.5)
    ax.legend(loc="upper right", markerscale=4)
    ax.grid(linestyle=":", linewidth=0.5)
    fig.savefig(out)
    plt.close(fig)
    logger.info("  wrote %s (%d points)", out, len(d))
    return out


def density_panel(df, key, out):
    """Paper Fig. 10d: along-track point density of the ice segments."""
    d = _subset(df, key)
    d = d[d.pred_label <= 1]
    if len(d) == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.hist(d.lon, bins=40, alpha=0.6, edgecolor="black", label="freeboard 2m ATL03")
    ax.set_xlabel("along track lat (degree)")
    ax.set_ylabel("No. of data points")
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    logger.info("  wrote %s (%d points)", out, len(d))
    return out


# Paper Table IV: Horovod on a DGX A100 cluster, 20 epochs.
PAPER_TABLE_IV = {
    "gpus":       [1, 2, 4, 6, 8],
    "time_s":     [280.72, 143.22, 73.68, 49.42, 38.72],
    "time_epoch": [5.5, 2.778, 1.45, 0.97, 0.79],
    "speedup":    [1.00, 1.96, 3.81, 5.68, 7.25],
}


def horovod_speedup_panel(metrics_files, out):
    """Measured Horovod scaling against the paper's Table IV.

    Each metrics file is a training_metrics.json written by bin/train_lstm.py; the
    rank count comes from its "horovod_size" field, so the measured curve grows
    automatically as more configurations are run.
    """
    import json
    measured = {}
    epochs = {}
    for f in metrics_files:
        with open(f) as fh:
            m = json.load(fh)
        size = int(m.get("horovod_size", 1))
        measured[size] = float(m["train_time_seconds"])
        epochs[size] = int(m.get("epochs", 1)) or 1
    if not measured:
        logger.warning("  no metrics files - skipping Horovod panel")
        return None

    gpus = sorted(measured)
    base = measured[min(gpus)]
    speedup = [base / measured[g] for g in gpus]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    ax = axes[0]
    ax.plot(PAPER_TABLE_IV["gpus"], PAPER_TABLE_IV["speedup"], "o-", color="#ED7D31",
            label="Paper (DGX A100, Table IV)")
    ax.plot(gpus, speedup, "s-", color="#4472C4", label="This workflow")
    ax.plot(PAPER_TABLE_IV["gpus"], PAPER_TABLE_IV["gpus"], ":", color="gray", label="Ideal (linear)")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup")
    ax.set_xticks(PAPER_TABLE_IV["gpus"])
    ax.set_title("(a) Training speedup")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", linewidth=0.5)
    for g, sp in zip(gpus, speedup):
        ax.annotate(f"{sp:.2f}x", (g, sp), textcoords="offset points", xytext=(6, -10), fontsize=8)

    # Per epoch, because the two runs use different epoch counts (20 vs 50).
    ax = axes[1]
    ax.plot(PAPER_TABLE_IV["gpus"], PAPER_TABLE_IV["time_epoch"], "o-", color="#ED7D31",
            label="Paper (DGX A100)")
    ax.plot(gpus, [measured[g] / epochs[g] for g in gpus], "s-", color="#4472C4",
            label="This workflow (T4 / RTX 6000)")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Time per epoch (s)")
    ax.set_xticks(PAPER_TABLE_IV["gpus"])
    ax.set_yscale("log")
    ax.set_title("(b) Time per epoch")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", linewidth=0.5, which="both")
    for g in gpus:
        ax.annotate(f"{measured[g] / epochs[g]:.1f}s", (g, measured[g] / epochs[g]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("  wrote %s (measured %s)", out, {g: round(measured[g], 1) for g in gpus})
    return out


def main():
    parser = argparse.ArgumentParser(description="Render comparison panels for PAPER_COMPARISON.md")
    parser.add_argument("--freeboard", nargs="+", required=True, help="Freeboard CSV(s)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="training_metrics.json file(s) for the Horovod scaling panel")
    parser.add_argument("--output-dir", default="figures/run")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.metrics:
        mpaths = []
        for pat in args.metrics:
            mpaths.extend(glob.glob(pat))
        horovod_speedup_panel(sorted(mpaths), f"{args.output_dir}/fig16_horovod_scaling.png")

    paths = []
    for pat in args.freeboard:
        paths.extend(glob.glob(pat))
    if not paths:
        raise SystemExit("No freeboard CSVs matched")
    df = pd.concat([pd.read_csv(p) for p in sorted(paths)], ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(df), len(paths))

    o = args.output_dir
    classification_panel(df, "nov04_gt2r", f"{o}/fig06a_classification_atl03_nov04_gt2r.png")
    classification_panel(df, "nov26_gt2r", f"{o}/fig07a_classification_atl03_nov26_gt2r.png")
    sea_surface_panel(df, "nov04_gt2r", f"{o}/fig08a_sea_surface_methods_nov04_gt2r.png")
    freeboard_panel(df, "nov04_gt2r", f"{o}/fig10a_freeboard_atl03_nov04_gt2r.png")
    density_panel(df, "nov04_gt2r", f"{o}/fig10d_point_density_nov04_gt2r.png")


if __name__ == "__main__":
    main()
