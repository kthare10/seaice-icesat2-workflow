# Sea Ice Classification & Freeboard — Pegasus Workflow

A Pegasus WMS workflow that classifies 2 m ICESat-2 ATL03 segments into thick ice, thin ice and
open water with an LSTM, then derives local sea surface height and freeboard from the result.

> Iqrah, Koo, Wang, Xie & Prasad, *"Scalable Higher Resolution Polar Sea Ice Classification and
> Freeboard Calculation from ICESat-2 ATL03 Data"*, IPDPSW 2025 (arXiv:2502.02700).

The workflow builds its own **Apptainer** containers as part of the run — there is no Docker
anywhere, and no image to build or pull by hand before you start. See [Containers](#containers).

For how this implementation compares with the paper and the author's notebooks, see
[`GAP_ANALYSIS.md`](GAP_ANALYSIS.md). For measured results against the paper's reported numbers, see
[`PAPER_COMPARISON.md`](PAPER_COMPARISON.md).

---

## Reproducing the published results

This walks from a clean checkout to the numbers in
[`PAPER_COMPARISON.md`](PAPER_COMPARISON.md), using the author's corrected dataset. It takes about
half an hour, most of which is one training job and the first container build.

### What you need

| | |
|---|---|
| Submit host | Pegasus 5.0+, HTCondor, `apptainer`, Python 3 |
| Compute | At least one worker with a GPU (training and inference); the rest is CPU |
| Memory | 12 GB for the heaviest job, so worker slots of 16 GB are enough |
| Disk | ~10 GB: the two container images are 0.5 GB and 3.8 GB, plus ~150 MB of output |
| Network | Only for the container builds, which pull their base images once |

Check the pool can satisfy the GPU request before you start:

```bash
condor_status -af Machine Memory GPUs      # need a machine with GPUs >= 1
```

Nothing else is required. There are no credentials to configure and no images to build by hand,
because the workflow builds its own.

### Step 1 — Get the data

The input is six manually corrected ICESat-2 ATL03 track CSVs over the Ross Sea (4 and 26 November
2019, beams gt1r and gt2r), about 50 MB. It is the data the paper's figures are based on.

> **This repository does not distribute the dataset.** The manual label corrections are the work of
> the paper's authors, and whether and where to publish it is their decision. Request it from them
> — contact details are in the paper and in their repository at
> <https://github.com/jmiqra/Sentinel-2_Sea-Ice_Classification>.

What you need is six files named `ATL03_<datetime>_<rgt>_<tile>_<beam>_labeled_10m_done.csv`, each
a 2 m resampled ATL03 track with a `label` column (0 thick ice, 1 thin ice, 2 open water) and a
`h_cor_mean` column giving corrected elevation above mean sea surface.

Once you have the archive, unpack it so the CSVs land in `data/IS2_Corrected_data/`:

```bash
unzip IS2_Corrected_data.zip -d data/

# or, if it is published at a URL later:
#   curl -L -o IS2_Corrected_data.zip "<DATASET_URL>" && unzip IS2_Corrected_data.zip -d data/
```

The workflow reads whatever directory you point `--input-dir` at, so the location is not fixed; the
commands below assume `data/IS2_Corrected_data/`.

Confirm you have six files:

```bash
ls data/IS2_Corrected_data/*.csv | wc -l      # expect 6
```

They are already label-corrected (`*_done.csv`), which the generator detects so it skips the
auto-label correction stage. That stage must never run twice on the same file.

### Step 2 — Generate the workflow

```bash
python workflow_generator.py --input-dir data/IS2_Corrected_data -o workflow.yml
```

The log should end with:

```
Unique tracks (granule+beam): 4 -> ['20191104195311_05940510_gt1r', ...]
workflow seaice-classification-freeboard with 20 jobs generated
```

Four tracks from six files is correct: two Sentinel-2 tiles cover the same granule and beam, and
they are merged before windowing.

### Step 3 — Submit and watch

```bash
pegasus-plan --submit -s condorpool -o local workflow.yml
```

The command prints a submit directory. Use it to follow along:

```bash
pegasus-status -l <submit-dir>        # progress
pegasus-analyzer  <submit-dir>        # only if something fails
```

Expect roughly this, on a pool like the one above:

| Phase | Time |
|---|---|
| Container builds (first run only) | 10-17 min, in parallel |
| 6 preprocess jobs | under 1 min each, in parallel |
| Prepare, then train on GPU | ~12 min, dominated by training |
| Inference, freeboard, validation, figures | under 1 min total |
| **Total** | **~28 min cold, ~19 min with images already built** |

### Step 4 — Check you got the published numbers

```bash
python -c "
import json; m = json.load(open('output/training_metrics.json'))
print('accuracy %.4f  F1 %.4f' % (m['test_accuracy'], m['test_f1']))
print('per-class %%:', [round(100*r[i]/sum(r), 2) for i, r in enumerate(m['test_confusion_matrix'])])
"
```

You should see something close to:

```
accuracy 0.9593  F1 0.9594
per-class %: [97.28, 80.19, 76.49]          # thick ice, thin ice, open water
```

Training is stochastic and no seed is fixed for weight initialisation or dropout, so expect overall
accuracy within a few tenths of a percent and per-class figures within a few percent. The paper
reports 96.56 % overall, and 98.39 / 73.80 / 60.25 per class. See
[`PAPER_COMPARISON.md`](PAPER_COMPARISON.md) for the full discussion of where and why the numbers
differ.

`output/` should hold 15 files: the prepared dataset, model, normalisation parameters, metrics, test
predictions, all-track predictions, four freeboard CSVs, four validation reports, and the figure
bundle.

### Step 5 — Look at the figures

```bash
mkdir -p figs && tar xzf output/paper_figures.tar.gz -C figs && ls figs
```

Figures 4 to 15 and Tables I to V. The ATL07 and ATL10 panels in figures 6 to 11 are empty unless
you supply reference data; see [What is still missing](PAPER_COMPARISON.md#what-is-still-missing).

To regenerate the side-by-side comparison panels used in `PAPER_COMPARISON.md`:

```bash
python bin/comparison_figures.py --freeboard 'output/freeboard_*.csv' \
    --metrics 'output/training_metrics.json' --output-dir figures/run
```

### Optional — run it again faster, or distributed

The container builds are the only slow part of a cold run. Keep the images and skip those jobs:

```bash
mkdir -p containers
# The images the run just built are in the workflow scratch directory:
cp scratch/*/pegasus/seaice-classification-freeboard/run*/seaice_*.sif containers/
# (or build them by hand - see Containers)

python workflow_generator.py --input-dir data/IS2_Corrected_data \
    --prebuilt-containers ./containers -o workflow.yml
```

To train data-parallel across 2 GPUs on one worker:

```bash
python workflow_generator.py --input-dir data/IS2_Corrected_data \
    --horovod --n-gpus 2 -o workflow.yml
```

See [Distributed training with Horovod](#distributed-training-with-horovod) for the ceiling and what
to expect. On this hardware it preserves accuracy but gains little time, and
[`PAPER_COMPARISON.md`](PAPER_COMPARISON.md) explains why.

### If something goes wrong

Start with `pegasus-analyzer <submit-dir>`, which names the failed job and prints its output. The
[Troubleshooting](#troubleshooting) table at the end covers the failures seen so far.

---

## Starting from raw NASA granules instead

If you do not have labelled CSVs, work through [Preparing input data](#preparing-input-data) to
download ATL03 and Sentinel-2, extract photon segments, and auto-label them. That path needs NASA
Earthdata credentials and produces the `*_labeled_10m.csv` files this workflow consumes.

---

## How it works

Two phases. Data preparation runs on the submit host and is only needed if you do not already have
labeled CSVs. Everything after that is the Pegasus DAG.

```
 DATA PREPARATION (submit host, optional)     PEGASUS WORKFLOW (cluster)
 ────────────────────────────────────────     ──────────────────────────
 1. Download ATL03/07/10 + Sentinel-2   ──►   0. build_container  [CPU] [GPU]
 2. Extract ATL03 photons → CSV               1. (opt) extract_atl07 / extract_atl10  [M]
 3. Auto-label via Sentinel-2 overlay         2. (opt) autolabel_correct  [N]
    (assigns label 0/1/2)                     3. preprocess_atl03  [N]
                                              4. prepare_lstm_data  (fan-in)
                                              5. train_lstm  (GPU)
                                              6. inference_lstm  (GPU)
                                              7. compute_freeboard  [T]
                                              8. validate_freeboard  [T]
                                              9. paper_figures  (fan-in)
```

`N` = one job per input file, `T` = one job per unique track, `M` = one job per granule.

| Stage | Script | Container | Memory | GPU |
|---|---|---|---|---|
| Build container | `bin/build_container.sh` | — (runs on the submit host) | 8 GB | — |
| Extract ATL07 / ATL10 | `bin/extract_atl07.py`, `bin/extract_atl10.py` | CPU | 4 GB | — |
| Auto-label correction | `bin/autolabel_correct.py` | CPU | 4 GB | — |
| Preprocess ATL03 | `bin/preprocess_atl03.py` | CPU | 8 GB | — |
| Prepare LSTM data | `bin/prepare_lstm_data.py` | CPU | 12 GB | — |
| Train LSTM | `bin/train_lstm.py` | GPU | 12 GB | 1 |
| Inference LSTM | `bin/inference_lstm.py` | GPU | 12 GB | 1 |
| Compute freeboard | `bin/compute_freeboard.py` | CPU | 8 GB | — |
| Validate freeboard | `bin/validate_freeboard.py` | CPU | 4 GB | — |
| Paper figures | `bin/paper_figures.py` | CPU | 8 GB | — |

### What a "track" is

A track is one **granule + beam**, for example `20191104195311_05940510_gt1r`. Beam names alone are
not unique: `gt1r` appears on every date. Several input files can belong to one track when a pass
crosses more than one Sentinel-2 tile (`T02CNA` and `T02CNC` both cover `05940510_gt1r`); those are
merged before the sliding window is applied, so the along-track series stays continuous.

Freeboard and validation fan out per unique track, not per input file.

---

## Containers

The workflow uses two Apptainer images, and **builds them itself**:

| Image | Definition | Base | Used by |
|---|---|---|---|
| `seaice_cpu.sif` | `Apptainer/seaice_cpu.def` | `python:3.10-slim-bookworm` | everything except training and inference |
| `seaice_gpu.sif` | `Apptainer/seaice_gpu.def` | `tensorflow/tensorflow:2.14.0-gpu` | `train_lstm`, `inference_lstm` |
| `seaice_gpu_horovod.sif` | `Apptainer/seaice_gpu_horovod.def` | same, plus OpenMPI and Horovod | the GPU stages when `--horovod` is given |

**The images hold dependencies and nothing else.** Neither definition has a `%files` section, so no
file from this repository is ever copied inside. Everything the workflow needs at run time is
bind-mounted instead, which is what keeps the images stable while the code changes.

### How the build fits into the DAG

Two `build_container` jobs sit at the head of the workflow. Each takes a `.def` file as its only
input and produces a `.sif` as its output. From Pegasus's point of view the
image is just another file, so it flows down the DAG like any other product: every job that needs an
image declares it as an input, and Pegasus stages it to that job. Nothing is pre-staged, and the
`build_container_gpu` job only blocks the two GPU stages.

```
build_container_cpu ──► preprocess_atl03, prepare_lstm_data, compute_freeboard,
                        validate_freeboard, paper_figures, extract_*
build_container_gpu ──► train_lstm, inference_lstm
```

The build jobs are pinned to the **submit host** (`execution.site = local`). Unprivileged
`apptainer build` needs user-namespace support that compute nodes often lack, and building centrally
avoids shipping a multi-gigabyte image back from a worker only to redistribute it. Apptainer caches
the base layers in `$APPTAINER_CACHEDIR` (`~/.apptainer/cache` by default), so a second run of an
unchanged definition is far quicker than the first.

### Scripts are mounted, never baked in

Each compute job stages three things into its working directory: the `bin/seaice_run.sh` wrapper,
its own `bin/<stage>.py` script, and the `.sif`. The wrapper bind-mounts that directory into the
container and runs the script from there:

```bash
apptainer exec [--nv] --no-home --bind "$PWD:/srv" --pwd /srv <image>.sif python3 /srv/<stage>.py <args>
```

The Python therefore lives outside the image and is mounted in at run time. Edit a stage script,
rerun, and the same image is reused — no rebuild, no cache invalidation, no re-staging of a
multi-gigabyte file. Rebuild only when a dependency changes, which means editing the `%post` section
of a definition. GPU stages get `--nv` so the host NVIDIA driver is passed through.

The same applies to data: input CSVs, models and intermediate products are all staged by Pegasus and
mounted, never built into an image.

### Building by hand, and reusing images

Container builds are the slowest part of a cold run (the GPU image pulls a ~3 GB base). To build
once and reuse across runs:

```bash
mkdir -p containers
apptainer build containers/seaice_cpu.sif Apptainer/seaice_cpu.def
apptainer build containers/seaice_gpu.sif Apptainer/seaice_gpu.def

python workflow_generator.py --input-dir data/IS2_Corrected_data \
    --prebuilt-containers ./containers -o workflow.yml
```

`--prebuilt-containers DIR` drops the two build jobs and registers the existing images as workflow
inputs instead. Because the definitions copy nothing from the repository, you can build them from
any directory.

To inspect an image:

```bash
apptainer exec containers/seaice_cpu.sif python3 -c "import pandas, rasterio; print('ok')"
apptainer shell containers/seaice_gpu.sif
```

---

## Preparing input data

Skip this section if you already have labeled CSVs. The workflow's input is a directory of track
CSVs carrying a `label` column (`0` thick ice, `1` thin ice, `2` open water).

### Step 1: Download raw data

ICESat-2 access needs NASA Earthdata credentials (register at <https://urs.earthdata.nasa.gov/>):

```bash
export EARTHDATA_TOKEN="your-token"          # preferred
# or: export EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=...

pip install earthaccess pystac-client planetary-computer rasterio h5py pyproj opencv-python-headless
python bin/download_data.py --all --output-dir ./data/
```

This creates `data/atl03/`, `data/atl07/`, `data/atl10/` and `data/sentinel2/`. Defaults follow the
paper: Ross Sea, November 2019.

**Useful variations**:

```bash
# Dry run — list what would be downloaded without fetching anything
python bin/download_data.py --all --dry-run --output-dir ./data/

# Download only ATL03 for the 8 paper RGTs
python bin/download_data.py --products atl03 \
    --rgt 0578,0594,0731,0779,0838,0884,0929,0961 --output-dir ./data/

# Custom region, date range, and product selection
python bin/download_data.py --products atl03,atl07 --bbox -180,-78,-150,-60 \
    --start-date 2020-01-01 --end-date 2020-01-31 --output-dir ./data/

# Limit downloads for quick testing
python bin/download_data.py --products atl03 --max-granules 3 --output-dir ./data/
```

<details>
<summary>All download options</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--products` | `atl03` | Comma-separated: atl03, atl07, atl10, sentinel2 |
| `--all` | — | Download all products |
| `--output-dir` | required | Output directory |
| `--region` | `ross_sea` | Predefined region (ross_sea, weddell_sea, beaufort_sea, arctic_ocean, southern_ocean) |
| `--bbox` | — | Custom `min_lon,min_lat,max_lon,max_lat` (overrides --region) |
| `--start-date` | `2019-11-01` | Start date |
| `--end-date` | `2019-11-30` | End date |
| `--rgt` | — | Comma-separated RGT filter (ICESat-2 only) |
| `--dry-run` | false | List granules without downloading |
| `--max-granules` | — | Limit per ICESat-2 product |
| `--max-cloud-cover` | 30 | Sentinel-2 cloud cover threshold (%) |
| `--max-scenes` | 20 | Max Sentinel-2 scenes |

</details>

### Step 2: Extract ATL03 photons to CSV

Bins photons into 2 m along-track segments with per-segment statistics and geophysical corrections.

```bash
python bin/extract_atl03.py --input-dir ./data/atl03/ --output-dir ./data/extracted/
```

Beams default to `strong`, which reads `orbit_info/sc_orient` and picks the strong beams for that
orientation (`1` forward gives `gt1r/gt2r/gt3r`, as in November 2019; `0` backward gives the left
beams). Pass `--beams gt1r,gt2r` to override.

**Output**: One CSV per granule-beam combination in `data/extracted/` (e.g. `ATL03_20191103184432_05780510_007_01_gt1l.csv`).

Processing steps per beam:
1. Read photon-level data (`h_ph`, `lat_ph`, `lon_ph`, `signal_conf_ph`) from HDF5
2. Filter for high-confidence sea-ice signal photons (`signal_conf >= 3`)
3. Bin photons into 2-meter along-track segments
4. Compute per-segment statistics (mean/median/std height, photon counts, background rate)
5. Apply first-photon bias correction and geophysical corrections (geoid, DAC, tide, MSS)
6. Write CSV with 41 columns matching the pipeline's expected format

<details>
<summary>Output columns</summary>

| Column | Description |
|--------|-------------|
| `Ori_Id` | Segment index |
| `year`, `month`, `day`, `hour`, `minute`, `second` | Acquisition timestamp |
| `lat`, `lon` | Geographic coordinates (WGS84) |
| `x`, `y` | Projected coordinates (EPSG:3976 Antarctic Polar Stereographic) |
| `dac`, `geoid`, `tide` | Geophysical corrections (meters) |
| `s_azi`, `s_ele` | Satellite azimuth and elevation |
| `N` | Number of photons in segment |
| `height_mean`, `height_med`, `height_sd` | Elevation statistics (meters) |
| `pcnt_mean/sd/med` | Photon count percentile statistics |
| `pcnth_mean/sd/med` | Height percentile statistics |
| `bcnt_mean/sd/med` | Background photon count statistics |
| `brate_mean/sd/med` | Background rate statistics |
| `fpb_corr` | First-photon bias correction (meters) |
| `mss` | Mean sea surface height (meters) |
| `h_cor_mean` | Corrected mean elevation above MSS (meters) |
| `h_cor_med` | Corrected median elevation above MSS (meters) |
| `x_atc` | Along-track distance (meters) |
| `geometry` | WKT point in projected coordinates |

</details>

<details>
<summary>All extraction options</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | — | Directory containing ATL03 HDF5 files |
| `--input` | — | Specific HDF5 file(s) (supports glob patterns) |
| `--output-dir` | required | Output directory for CSV files |
| `--beams` | `strong` | `strong` picks the strong beams from `orbit_info/sc_orient`; or a comma-separated list such as `gt1r,gt2r` |
| `--bin-size` | `2.0` | Along-track bin size in meters |

</details>

### Step 2b: Extract ATL07/ATL10 to CSV (optional)

Reference products for validation and the comparison figures.

```bash
python bin/extract_atl07.py --input-dir ./data/atl07/ --output-dir ./data/atl07_csv/
python bin/extract_atl10.py --input-dir ./data/atl10/ --output-dir ./data/atl10_csv/
```

**Output**: One CSV per granule (all 6 beams combined, distinguished by `beam` column).

**ATL07 output columns**: `beam, lat, lon, x, seg_id, height, h_ref, label, error, quality, confidence, ssh_flag, mss, dac, geoid, tide, freeboard, day, hour, minute, second`

**ATL10 output columns**: `beam, lat, lon, x, seg_id, height, label, h_ref, freeboard, fb_confidence, fb_quality, h_norm, error, ssh_flag, day, hour, minute, second`

ATL07 extraction computes `freeboard = max(0, height - h_ref)`. Both scripts parse timestamps from `delta_time` + ATLAS epoch (2018-01-01). Path variants across HDF5 versions (v2–v7) are handled automatically.

<details>
<summary>All ATL07/ATL10 extraction options</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | — | Directory containing HDF5 files |
| `--input` | — | Specific HDF5 file(s) (supports glob patterns) |
| `--output-dir` | required | Output directory for CSV files |
| `--beams` | all 6 beams | Comma-separated beam names |

</details>

**Alternative**: Instead of extracting on the submit host, you can pass raw HDF5 files to the workflow generator with `--raw-atl07-dir` / `--raw-atl10-dir` and extraction will run as workflow jobs (see Step 4).

**Alternative**: pass the raw HDF5 directories to the generator with `--raw-atl07-dir` /
`--raw-atl10-dir` and the extraction runs as workflow jobs instead.

### Step 3: Auto-label using Sentinel-2

Overlays the tracks on coincident Sentinel-2 imagery and classifies each segment by HSV brightness:

| Class | Label | HSV Value | Appearance |
|---|---|---|---|
| Thick ice | 0 | V >= 205 | Bright white |
| Thin ice | 1 | 31 <= V < 205 | Medium |
| Open water | 2 | V < 31 | Dark |

```bash
python bin/autolabel_s2.py --tracks-dir ./data/extracted/ \
    --sentinel2 ./data/sentinel2/sentinel2_scenes.tar.gz --output-dir ./data/labeled/
```

The script:
1. Loads Sentinel-2 RGB bands (B04/B03/B02) from the tar.gz or a directory
2. Segments each scene into ice classes using HSV thresholds
3. Projects each track's lon/lat onto the S2 raster's pixel grid (via pyproj + rasterio)
4. Finds the best-coverage scene for each track
5. Looks up the label at each segment's pixel location
6. Writes `*_labeled_10m.csv` files with three new columns: `pix_x`, `pix_y`, `label`

If you already have extracted S2 scenes (not in a tar.gz):

```bash
python bin/autolabel_s2.py \
    --tracks-dir ./data/extracted/ \
    --sentinel2-dir ./data/sentinel2/scenes/ \
    --output-dir ./data/labeled/
```

<details>
<summary>All labeling options</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--tracks-dir` | required | Directory with extracted ATL03 track CSVs |
| `--sentinel2` | — | Path to `sentinel2_scenes.tar.gz` |
| `--sentinel2-dir` | — | Directory of extracted S2 scene subdirectories (each with B02/B03/B04 GeoTIFFs) |
| `--output-dir` | required | Output directory for labeled CSVs |

</details>

The resulting `*_labeled_10m.csv` files are the workflow's input.

> **Manual correction.** The paper's results use labels that were hand-corrected in cloudy areas and
> at ice-type transitions, producing `*_labeled_10m_done.csv`. The generator detects `*_done.csv`
> inputs and automatically skips the `autolabel_correct` stage, which must never run twice on the
> same file — it trims open water from the track ends.

---

## Running the workflow

### Generate

```bash
python workflow_generator.py --input-dir ./data/labeled/ -o workflow.yml
```

Common variations:

```bash
# Reproduce the paper's architecture instead of the notebook's tuned model
python workflow_generator.py --input-dir ./data/labeled/ --preset paper -o workflow.yml

# Quick end-to-end smoke test
python workflow_generator.py --input-dir ./data/labeled/ --max-tracks 2 --epochs 5 -o test.yml

# With ATL07/ATL10 reference CSVs, so the comparison panels are populated
python workflow_generator.py --input-dir ./data/labeled/ \
    --atl07-dir ./data/atl07_csv/ --atl10-dir ./data/atl10_csv/ -o workflow.yml

# Let the workflow extract ATL07/ATL10 from raw HDF5 as jobs
python workflow_generator.py --input-dir ./data/labeled/ \
    --raw-atl07-dir ./data/atl07/ --raw-atl10-dir ./data/atl10/ -o workflow.yml

# Reuse images you already built
python workflow_generator.py --input-dir ./data/labeled/ \
    --prebuilt-containers ./containers -o workflow.yml
```

<details>
<summary>All workflow generator options</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | required | Directory with labeled track CSVs |
| `--pattern` | `*_labeled_10m_done.csv` | Glob pattern for input files |
| `--preset` | `notebook` | LSTM architecture. `notebook`: the author's tuned model (LSTM 48 tanh, dropout 0.4, 2 x Dense 16, lr 0.000889, 50 epochs). `paper`: Section III.B as written (LSTM 16 ELU, dropout 0.2, 7 dense layers, lr 0.003, 20 epochs) |
| `--epochs` | preset | Training epochs (overrides the preset) |
| `--batch-size` | 32 | Batch size |
| `--units` / `--dropout` / `--lr` | preset | Override individual preset values |
| `--radius` | 5000 | Sliding window radius in metres |
| `--weight-form` | `paper` | Lead weight in the NASA sea-surface equation. `paper`: exp(-((h-hmin)/sigma)^2), Eq. 2. `notebook`: the author's exp(-(h-hmin)/sigma)^2 |
| `--smooth-window` | 10000 | Rows either side for the nanmin smoothing of `new_h_ref` (Notebook 6); 0 disables |
| `--smooth-max-gap` | 10000 | Pieces of a track separated by an along-track jump larger than this (m) are smoothed independently, so tiles tens of km apart do not share a sea surface. 0 smooths the whole track as one, as the notebook does |
| `--water-threshold` | off | Mask the NASA sea surface above this height (m) before smoothing. Notebook 6 does this with 0.2 m + mean(ATL03 − ATL07), which needs ATL07 data, so it is off by default |
| `--lead-fallback` | `thin_ice` | Sea-surface windows with no predicted open water: `thin_ice` uses thin-ice segments as leads (the notebook); `none` leaves them to interpolation from neighbouring windows (what the paper describes) |
| `--zscore` | off | Enable the z-score stage that the author's final pipeline leaves disabled |
| `--prebuilt-containers DIR` | — | Reuse `seaice_cpu.sif` / `seaice_gpu.sif` from DIR and drop the build jobs |
| `--n-gpus` | 1 | GPUs requested for training |
| `--horovod` | off | Data-parallel training: swaps in the Horovod GPU image and launches training under `horovodrun -np <n-gpus>`. See [Distributed training with Horovod](#distributed-training-with-horovod) |
| `--skip-autolabel` / `--force-autolabel` | auto | Correction is skipped automatically when every input is `*_done.csv` |
| `--max-tracks` | — | Use only the first N input files |
| `--atl07-dir` / `--atl10-dir` | — | Reference CSVs (`ATL07*.csv`, `ATL10*.csv`) for validation and figures |
| `--raw-atl07-dir` / `--raw-atl10-dir` | — | Raw HDF5 directories; extraction runs as workflow jobs |
| `-e, --execution-site` | `condorpool` | Execution site name |
| `-o, --output` | `workflow.yml` | Output file |

</details>

### Submit and monitor

```bash
pegasus-plan --submit -s condorpool -o local workflow.yml

pegasus-status -l <submit-dir>        # progress
pegasus-analyzer <submit-dir>         # why a job failed
pegasus-statistics -s all <submit-dir># runtimes per stage
```

### Outputs

Everything lands in `output/`:

| File | Contents |
|---|---|
| `ATL03_cor_label_LSTM_ann_prepared.csv` | Feature vectors: label, track, metadata, 5 timesteps x 8 features |
| `lstm_sea_ice_model.h5`, `norm_params.json` | Trained model and the normalisation statistics used at inference |
| `training_metrics.json` | Held-out test accuracy, confusion matrix, per-class report, history |
| `test_predictions.csv` | Predictions on the held-out 20% split (the basis for Figure 4) |
| `LSTM_predictions_all.csv` | Per-segment class probabilities and `pred_label` for every track |
| `freeboard_<track>.csv` | Sea surface by four methods, plus `freeboard_new_h_ref_smooth` |
| `report_<track>.tar.gz` | Per-track validation plots |
| `paper_figures.tar.gz` | Figures 4-15 and Tables I-V |

---

## Distributed training with Horovod

Training is the longest stage (about 13 minutes for 50 epochs on one GPU). Horovod runs it
data-parallel across several GPUs.

```bash
python workflow_generator.py --input-dir data/IS2_Corrected_data \
    --horovod --n-gpus 2 -o workflow.yml
pegasus-plan --submit -s condorpool -o local workflow.yml
```

That changes three things:

1. The GPU image becomes `Apptainer/seaice_gpu_horovod.def`, which adds OpenMPI and Horovod on top
   of the same TensorFlow base. It is a separate definition because Horovod compiles from source and
   adds roughly 10 to 20 minutes to a cold build, which runs without `--horovod` should not pay.
2. `train_lstm` requests `request_gpus = <n-gpus>` instead of 1.
3. The wrapper launches `horovodrun -np <n-gpus> python3 train_lstm.py ...` instead of plain
   `python3`, so there is one rank per GPU. `bin/train_lstm.py` already calls `hvd.init()`, pins a
   GPU per rank, scales the learning rate by the rank count, broadcasts initial variables from rank
   0, and writes output only from rank 0.

### The ceiling is one worker

HTCondor's vanilla universe gives a job a **single machine**, so every rank must fit on one worker.
On this cluster that means `--n-gpus 2`:

| Worker | GPUs |
|---|---|
| GPN-gpu-worker-1 | 2 x Tesla T4 |
| NCSA-gpu-worker-1 | 2 x Quadro RTX 6000 |

`--n-gpus 4` would never match a slot and the job would sit idle. The generator warns about this.
Going beyond one node needs HTCondor's parallel universe with MPI over SSH between workers, which
this workflow does not set up; the two GPU nodes also carry different GPU models, so the faster one
would wait on the slower.

The paper reports up to 8 GPUs on a single DGX A100, where all 8 are in one machine. The same
approach works here, bounded by the hardware.

### Collectives

The definition installs the standalone NCCL wheel and builds Horovod against it, because
TensorFlow's image ships NCCL headers but no shared library. If that is unavailable the build falls
back to MPI collectives automatically, which is adequate at this scale. The build log says which was
used.

### Checking it worked

```bash
apptainer exec containers/seaice_gpu_horovod.sif horovodrun --check-build
```

In the job log, `seaice_run.sh` prints the exact command, and each rank reports its id:
`Horovod rank 0/2`, `Horovod rank 1/2`.

---

## Reference

### LSTM model

- **Input**: 5 timesteps x 8 features. Feature columns are selected **by name**
  (`<feature><offset>`, offsets -2 to +2); metadata (`track`, `x_atc`, `year`, `month`, `day`,
  `lon`, `lat`) travels alongside and is never fed to the model.
- **Features**: `h_cor_mean`, `h_diff`, `rel_height_min_elev`, `height_sd`, `pcnth_mean`,
  `pcnt_mean`, `bcnt_mean`, `brate_mean`. These are the eight the notebooks build; the paper says
  six per timestep without naming them, and the notebook's final model cell declares an input
  shape of `(5, 6)`, so which six the paper trained on is not recoverable.
- **`rel_height_min_elev` is derived from the labels.** It is the height above the lowest
  open-water segment in the surrounding 10 km window, where "open water" is taken from the
  ground-truth `label` column (Notebook 2). The classifier therefore sees a feature computed from
  its neighbours' true labels. This follows the notebooks and the paper's numbers carry the same
  property, so the comparison is like for like, but the accuracy is not a fair estimate of
  performance on unlabelled tracks, where this feature cannot be computed.
- **`--preset notebook`** (default): LSTM(48, tanh) → Dropout(0.4) → Dense(16, elu) → Dropout(0.4)
  → Dense(16, elu) → Dropout(0.4) → Dense(3, softmax); Adam 0.000889; 50 epochs.
- **`--preset paper`**: LSTM(16, elu) → Dropout(0.2) → Dense(32, 96, 32, 16, 112, 48, 64; elu) →
  Dense(3, softmax); Adam 0.003; 20 epochs. The paper puts dropout in the LSTM layer only, so this
  preset adds none after the dense layers (`--dense-dropout` overrides).
- **Loss**: `CategoricalFocalCrossentropy(alpha=[0.05, 0.45, 0.60], gamma=2.0)`, for the heavy class
  imbalance toward thick ice. The paper does not state its alpha; the notebook's is used.
- **Normalisation**: `(x - mean) / (1 - std)`, the notebook's formula, with the statistics saved to
  `norm_params.json` and reloaded at inference. Note what this does to the inputs: features with
  std below 1 are scaled up (elevation by about 2x), features with std above 1 are sign-flipped
  and shrunk. On this data `bcnt_mean` (std ≈ 326) and `brate_mean` (std ≈ 1.4e6) are scaled by
  about −0.003 and −7e−7, so two of the eight features are effectively zero at the model input.
  `--norm zscore` gives a standard z-score instead; it is not the default because the published
  numbers were obtained with the notebook formula.
- **Split**: 60/20/20 train/validation/test, random state 20. Reported metrics are on the held-out
  test split, never on training rows. The normalisation statistics are computed on the whole
  labelled set before the split, as in the notebook.
- **F1, precision, recall** in `training_metrics.json` are the notebook's custom Keras metrics:
  per-batch values on rounded softmax outputs, averaged over batches. They are what the paper's
  Table III reports, and differ from scikit-learn's macro or weighted F1, which the per-class
  report in the same file provides.

### Freeboard

Local sea surface comes from open-water segments in a 10 km window (5 km radius) using NASA's
weighted-lead equation, then the Notebook 6 smoothing (a `nanmin` rolling window followed by
nearest-neighbour interpolation). Freeboard is `h_cor_mean` minus that surface. Three simpler
surfaces (minimum, average and nearest-minimum elevation) are computed alongside for comparison.

Details that matter when reading the result against the paper:

- **Windows advance 5 km, not 10.** Each iteration assigns its 10 km window's estimate to every
  row in it and then jumps to the window's end; the next window's back half overwrites this one's
  front half. Every row therefore carries the estimate from the 10 km window that starts at its
  own 5 km chunk, which is the 10 km window with 5 km overlap the paper describes.
- **Windows without open water use thin ice as leads** (`--lead-fallback thin_ice`, the notebook's
  behaviour). The paper says such windows are interpolated from their neighbours; `--lead-fallback
  none` does that. On the author's four tracks 15 of 70 windows take the fallback, and it is the
  main reason the raw NASA surface is rougher than the three simpler ones before smoothing.
- **The smoothing window is 10,000 rows either side**, about 20 km at 2 m spacing, and is defined
  in rows, not metres. `--smooth-max-gap` (default 10 km) splits a track at larger along-track
  jumps so that a track assembled from distant Sentinel-2 tiles is smoothed piecewise.
- **Notebook 6 masks the surface before smoothing** at 0.2 m above the mean ATL03−ATL07 height
  difference for the figures that compare against ATL07. That needs ATL07 data, so it is off by
  default (`--water-threshold`).

### ATL07/ATL10 reference data

Validation follows the notebooks, which compare against Koo et al. (2023) ATL07 CSVs whose `label`
is a classifier output (`0` water, `1` thin ice, `2` thick ice) with `h_ref`, `freeboard` and
`freeboard_10`. CSVs produced by `bin/extract_atl07.py` instead carry NASA's `height_segment_type`;
`validate_freeboard.py` remaps those to water/ice (`--atl07-label-scheme`). Thin ice cannot be
recovered from the NASA codes.

### Data transfers

All transfers use CondorIO (`pegasus.data.configuration = condorio`), so no shared filesystem is
needed between workers. `pegasus.transfer.worker.package = true` uses the submit host's worker
package rather than downloading one per job.

---

## Running stages standalone

Every stage is a normal CLI program, useful for debugging outside Pegasus:

```bash
# inside the container
apptainer exec containers/seaice_cpu.sif python3 bin/preprocess_atl03.py \
    --input track_done.csv --output enriched.csv --radius 5000

# the wrapper the workflow uses, with the same argument layout
bin/seaice_run.sh --container containers/seaice_cpu.sif --script bin/prepare_lstm_data.py -- \
    --input-dir ./enriched/ --output prepared.csv --nearby 2

# GPU stages
bin/seaice_run.sh --container containers/seaice_gpu.sif --script bin/train_lstm.py --nv -- \
    --data prepared.csv --model-output model.h5 --epochs 50
```

## Licence and attribution

This workflow is licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).

Two things in this repository are **not** covered by that licence:

- **`figures/paper/`** — panels reproduced from the paper for comparison, under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), the licence the paper
  carries. See [`figures/paper/README.md`](figures/paper/README.md).
- **The input data** — not distributed here at all; see [Step 1](#step-1--get-the-data).

The scientific method implemented here is the authors' work, not this repository's. If you use this
workflow, cite the paper:

```bibtex
@inproceedings{iqrah2025seaice,
  title     = {Scalable Higher Resolution Polar Sea Ice Classification and
               Freeboard Calculation from {ICESat-2} {ATL03} Data},
  author    = {Iqrah, Jurdana Masuma and Koo, Younghyun and Wang, Wei and
               Xie, Hongjie and Prasad, Sushil K.},
  booktitle = {IEEE International Parallel and Distributed Processing Symposium
               Workshops (IPDPSW)},
  year      = {2025},
  note      = {arXiv:2502.02700}
}
```

Differences between this implementation and the published method are recorded in
[`GAP_ANALYSIS.md`](GAP_ANALYSIS.md), and measured results are compared in
[`PAPER_COMPARISON.md`](PAPER_COMPARISON.md).

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `build_container` fails with a FATAL from apptainer | Unprivileged build needs user namespaces. Check `allow user ns = yes` in `apptainer.conf`, or build by hand and use `--prebuilt-containers` |
| Held job: "Transfer output files failure ... No such file" | The stage exited before writing its declared output. Read the job's `.out`/`.err` under the submit directory with `pegasus-analyzer` |
| Jobs idle forever | No slot matches the memory or GPU request. Compare `condor_status -af Machine Memory GPUs` with the per-stage table above |
| Container build is slow on every run | Expected on a cold cache. Build once and pass `--prebuilt-containers` |
| `OSError: Read-only file system: '/home/...'` from horovodrun or matplotlib | Something inside the container wrote under `$HOME`, which `--no-home` leaves unmounted. `bin/seaice_run.sh` sets `HOME=/srv` (the job directory) to prevent this; check that it is the version being staged |
| A fix to `bin/*.sh` or `bin/*.py` does not take effect on a retry | Pegasus copies executables into the workflow scratch at stage-in, so a released or retried job re-uses that copy. Overwrite the file under `scratch/.../<run>/` too, or replan |
| Empty ATL07/ATL10 panels in the figures | No reference data was staged. Pass `--atl07-dir` / `--atl10-dir` |
