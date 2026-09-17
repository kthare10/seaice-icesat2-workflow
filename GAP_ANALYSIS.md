# Gap Analysis: seaice-workflow vs. paper (Iqrah et al., IPDPSW 2025) and author notebooks

Date: 2026-09-16
Inputs reviewed: `2502.02700v1.pdf`, notebooks 1–6 in `../Scaled_IS2_Classification_Freeboard/`,
`workflow_generator.py`, `bin/*.py`, `Docker/*`, the April run in `output/`, and the author's
new dataset `data/IS2_Corrected_data.zip` (6 corrected, labelled track CSVs).

## 1. Bottom line

The DAG shape is right and the per-stage scripts follow the notebooks closely, but the
workflow **cannot currently produce a valid model or freeboard**. The April run in `output/`
already shows it: `training_metrics.json` has `test_accuracy = 0.0` and `test_loss = NaN`, and
`LSTM_predictions_all.csv` carries only `pred_label0`. Running the CPU stages on the author's
six files (this session, in the project venv) reproduces the root causes:

| Symptom | Evidence |
|---|---|
| Metadata treated as model features | prepared CSV has 46 columns (42 expected). `train_lstm.py` pops only `label`/`track`, so `x_atc, day, lon, lat` become the first four "features" of timestep 0 and `pcnth_mean2 … brate_mean2` fall off the end. `output/norm_params.json` confirms `x_atc, day, lon, lat` were normalised as features. |
| NaN features poison training | 693 feature vectors (all of `T02CNA_gt2r`, a 698-row all-thick-ice segment) have NaN `rel_height_min_elev*` because there is no open water in the track, so the window sea surface is NaN and interpolation cannot fill it. Nothing drops them → NaN loss. |
| Freeboard computed on normalised heights | `inference_lstm.py` builds the output from `norm_df` and renames `h_cor_mean0` → `h_cor_mean`. `compute_freeboard.py` therefore runs the NASA sea-surface equation on z-scored values, not metres. The notebook (NB4 cells 4/9) re-attaches the **raw** centre features. |
| Probabilities dropped | The "strip neighbour columns" filter `c.endswith("1"/"2")` also removes `pred_label1` and `pred_label2`. |
| Tracks collide | `track` is just the beam (`gt1r`/`gt2r`). In the author's data both beams appear on Nov 4 **and** Nov 26 with overlapping `x_atc` (gt2r: 28316536–28519422 vs 28333160–28384618). `compute_freeboard --track gt2r` sorts both days into one series and windows across them. |
| Corrected inputs get re-trimmed | Default pattern is `*_labeled_10m_done.csv` but `--skip-autolabel` is off. `autolabel_correct.py` scans backwards to the *last interior* open-water chunk and deletes everything after it: 42154→31105, 25093→18349, 20911→15088 rows (24–28 % loss). NB1 only applied this to raw auto-labelled files, never to `_done` files. |
| Containers missing | Generator references `containers/seaice_cpu.sif` / `seaice_gpu.sif`; the directory does not exist. README documents Docker Hub images instead. |

## 2. What the author actually ran (from the notebooks)

Pipeline used for the corrected-label ("cor_label") results, in order:

1. `done/*_labeled_10m_done.csv` (manually corrected) → NB2 cell 14: per-file sliding window
   (radius 5000 m, non-overlapping advance) → `rel_sea_surf_{min_elev,avg_elev,min_dist}`
2. NB2 cell 16: linear interpolation (`limit_direction="both"`) → `rel_height_*`
3. NB2 cell 19: `h_diff = h_cor_mean - h_cor_med`
4. NB3 cell 6: `make_input(..., nearby=2)` with 8 features → `*_ann_new_param.csv` (label + 40 cols, **no track**)
5. NB3 cell 11: concat, `track = filename[-22:-18]` (beam only) → `ATL03_cor_label_LSTM_ann_prepared.csv` (122,936 rows)
6. NB3 cells 18–23: pop label/track, `norm(x) = (x-mean)/(1-std)`, reshape `row[1:41]` → (N,5,8), 60/20/20 split
7. NB3 cell 37: model = LSTM(48, tanh) → Dropout 0.4 → Dense16 elu → Dropout → Dense16 elu → Dropout → Dense3 softmax; Adam 0.000889; CategoricalFocalCrossentropy(alpha=[0.05,0.45,0.60], γ=2); 50 epochs (an alternative tuner run in cell 32 picked LSTM 80 elu, 3 dense)
8. NB3 cell 51 / NB4: inference on the **all-tracks** set (1.3 M windows, 15 files) normalised with *its own* mean/std; predictions merged back to original rows by exact float match on six feature columns (`pd.merge`)
9. NB5: sea surface (min/avg/nearest-min) on `pred_label`, then NASA equation → `new_h_ref`, interpolate, `freeboard_new_h_ref`
10. NB6: per (day, beam): threshold `new_h_ref <= 0.2 + mean(height_diff_03_07)`, `smooth_line` = nanmin over ±10 000 rows, nearest interpolation → `new_h_ref_smooth`, `freeboard_new_h_ref_smooth`; compared with Koo's ATL07 CSVs (`csv_Iqrah/ATL07-02_*.csv`: `h_ref`, `freeboard`, `freeboard_10`, `h_ref_10`, label 0=water/1=thin/2=thick)

Notes that matter for the workflow:

- The **z-score stage is not part of this pipeline** (NB2 cell 6 has `break` before it and writes to a
  directory nobody reads). `preprocess_atl03.py` runs it anyway and drops rows with NaN z-score.
- The 122,936-row training set is consistent with only four files (T02CNC gt1r/gt2r, T03CWT gt1r/gt2r);
  the two `T02CNA` files in the new zip were apparently **not** in the paper's training set. With all six
  files the workflow produces 134,937 vectors.
- The paper's Table I covers eight IS2/S2 pairs, but the shared data covers exactly the two tracks used
  in Figs. 6–11 (`20191104195311_05940510` and `20191126182014_09290510`), so `paper_figures.TRACK_PARAMS`
  already matches the data.
- Strong beams in Nov 2019 were the **right** beams (`gt1r/gt2r/gt3r`, per the author's files and NB1/NB3),
  i.e. the spacecraft was in forward orientation (`sc_orient = 1`: weak beams lead, strong beams are
  `gt*r`; backward is the reverse). `extract_atl03.py` hard-coded `gt1l/gt2l/gt3l`; the April run used those.

## 3. Paper vs. notebook vs. workflow (model)

| Item | Paper §III.B | Notebook (final cells) | Workflow default |
|---|---|---|---|
| Features / timestep | 6 | 8 | 8 |
| Timesteps | 5 (n±2) | 5 | 5 |
| LSTM | 16 units, ELU | 48 tanh (cell 37); tuner best 80 elu (cell 32) | 48 tanh |
| Dropout | 0.2 | 0.4 | 0.4 |
| Dense stack | 7 layers: 32, 96, 32, 16, 112, 48, 64 (ELU) | 2 × 16 ELU | 2 × 16 ELU |
| Optimiser / LR | Adam 0.003 | Adam 0.000889 | Adam 0.000889 |
| Loss | focal | CategoricalFocalCrossentropy α=[.05,.45,.60] γ=2 | same |
| Epochs / batch | 20 / 32 | 50 / 32 | 50 / 32 |
| Split | 80/20 | 60/20/20 (rs=20) | 60/20/20 (rs=42) |
| Normalisation | not stated | (x−mean)/(1−std) recomputed per dataset | same formula, params saved from training (improvement) |
| Reported accuracy | 96.56 % (test) | 95.47 % test (cell 38) | 0.0 (broken) |

The paper's architecture resembles tuner trial 1 in NB3 cell 31 (16 units, dropout 0.2, LR 0.0032).
Neither the workflow default nor the notebook's final model is the paper's model.

## 4. Other discrepancies

- **NASA lead weight.** Paper Eq. 2: `w_i = exp(-((h_i - h_min)/σ_i)^2)`. Notebook and
  `compute_freeboard.py`: `np.exp(-(hi-hmin)/si)**2` = `exp(-2(h-hmin)/σ)`. The workflow copied the
  notebook's operator-precedence bug. (The workflow did fix the notebook's `loc[id1:id2+1]` off-by-one.)
- **Smoothed sea surface missing.** The product plotted in Figs. 8–11 is `new_h_ref_smooth` (threshold +
  nanmin window + nearest interpolation, NB6). `compute_freeboard.py` stops at `new_h_ref`;
  `validate_freeboard.py` defines `smooth_line` but never calls it; `paper_figures.py` plots raw `new_h_ref`.
- **Confusion matrix / accuracy.** `inference_lstm.py` and `paper_figures.fig_4` score on the whole
  prepared set (train+val+test). Paper Table III / Fig. 4 are on the 20 % test split.
- **ATL07/ATL10 semantics.** `validate_freeboard.py` and `paper_figures.py` assume Koo-format CSVs
  (`label` 0=water/1=thin/2=thick from Koo's classifier, `h_ref`, `freeboard_10`, `h_ref_10`).
  `extract_atl07.py` writes `label = height_segment_type` (NASA codes 0–9) and `h_ref = height_segment_lpe`;
  `>= 1` is then read as "ice" and `2` as "thick ice", which is not what those codes mean. Needs a
  remap (e.g. `ssh_flag`/`height_segment_type` → water vs ice) or a note that the comparison requires
  Koo's CSVs. Also `date_str = f"201911{day:02d}"` hard-codes year/month.
- **Hard-coded "paper" tables are not from the paper.** `paper_figures.py` Table III lists LSTM
  0.9447/0.8871/0.7945/0.8327; the paper has 96.56/97.00/96.09/96.54 (Multi-Layer Perceptron 91.80/…/91.79). Table I lists
  S2 scene IDs like `T57CWS` and shifts in hours; the paper has time differences in minutes and S2 shifts
  in metres/direction (e.g. 9.55 min, 550 m NW). Table II/V executor timings and Table IV `Data/s`
  (11 422 vs paper 585.88) also differ. `CM_PAPER` diagonals (98.39/73.80/60.25) are correct.
- **RGT list.** `download_data.PAPER_RGTS` has `0884`; the author's Nov 23 file is `ATL03_20191123180255_08830510`
  (RGT 0883). It also contains `0961` (Nov 28, not in Table I) and no Nov 17 track. Paper uses release 006.
- **Inference scope.** The paper infers over all Nov 2019 tracks (15 files); the workflow only infers on
  the training set. A separate `--inference-dir` would need a label-free `rel_height_min_elev`
  (the author used threshold-based "ow_labeled" pre-labels for that set).
- **Horovod.** `--horovod` imports `horovod.tensorflow.keras`, but `requirements-gpu.txt`/`Dockerfile.gpu`
  do not install it and the job is launched as a plain `python` process (no `horovodrun -np N`).
- **Robustness.** `make_input_with_track` requires exact float equality for the 2 m spacing (fine for the
  author's integer `x_atc`, fragile for `extract_atl03.py` floats). `--pattern` silently falls back to
  `*.csv`. `del_last_first_chunk` trims 100 rows from the start even when the track starts with ice.

## 5. Changes needed (prioritised)

### P0 — required for a correct run on `data/IS2_Corrected_data`

1. **Select features by name, not position** (`train_lstm.py`, `inference_lstm.py`): build the 40 feature
   columns as `[f"{f}{t}" for t in (-2,-1,0,1,2) for f in FEATURES]`, pop everything else as metadata,
   assert `X.shape[1:] == (5, 8)`. Save the feature list in `norm_params.json`.
2. **Attach raw centre features to predictions** (`inference_lstm.py`): take `h_cor_mean0`, `height_sd0`, …
   from the un-normalised frame; keep `pred_label0/1/2` (replace the `endswith` filter with a regex on
   `^(feature)(-2|-1|1|2)$`).
3. **Handle NaN features** (`prepare_lstm_data.py`): `dropna(subset=feature_cols)` per file and log the
   count; warn when a track has no open water (all-NaN `rel_sea_surf_*`). Optionally exclude such tracks
   from training to match the author's set.
4. **Unique track id**: `re.search(r"ATL03_(\d{14})_(\d{8})_.*?(gt\d[lr])")` → `20191104195311_05940510_gt1r`.
   Use it in `prepare_lstm_data.py`, `compute_freeboard --track`, `validate_freeboard`, `paper_figures`
   (derive beam with `gt\d[lr]`). Fan out freeboard/validate per **unique** track id, not per input file;
   the two `T02CNA`/`T02CNC` gt1r tiles are one granule+beam and should be one series.
5. **Do not trim `_done` inputs**: default `--skip-autolabel` on when the pattern ends in `_done.csv`, and make
   `del_last_first_chunk` a no-op unless the first/last row is open water.
6. **Containers**: either add a documented `apptainer pull` step producing `containers/*.sif`, or point the
   `Container` objects at `docker://kthare10/seaice-icesat2-{cpu,gpu}:latest`. Add `horovod` to the GPU
   image or remove the `--horovod` flag until a launcher exists.

### P1 — fidelity to the notebooks and the paper

7. Remove the z-score stage from `preprocess_atl03.py` (or default it off); it is not in the author's
   pipeline and is the slowest step.
8. Fix the lead weight to the paper's `exp(-((h-hmin)/σ)^2)`; keep `--weight-form notebook` for parity runs.
9. Add `new_h_ref_smooth` / `freeboard_new_h_ref_smooth` (threshold, nanmin window `--smooth-window 10000`,
   nearest interpolation) to `compute_freeboard.py`; plot the smoothed series in `paper_figures.py`.
10. Make the model configurable (`--lstm-activation`, `--dense-units 16,16`, `--alpha`) with presets
    `--preset notebook` (current default) and `--preset paper` (16 ELU, 0.2, 7 dense, LR 0.003, 20 epochs).
11. Evaluate on the held-out split: have `train_lstm.py` write test-set predictions (or split indices) and
    let `paper_figures.fig_4` use them.
12. Fix or drop the hard-coded tables in `paper_figures.py`; transcribe Tables I–V from the PDF if kept.
13. ATL07/ATL10: remap NASA codes to water/ice before comparison, verify `h_ref` source against the ATBD,
    and carry `year`/`month` (or a date string) through `prepare_lstm_data.py` instead of hard-coding 201911.
14. `download_data.py`: RGT 0884 → 0883, drop 0961, add the Nov 17 track; `extract_atl03.py`: pick strong
    beams from `orbit_info/sc_orient` (1 = forward → `gt*r`, as in Nov 2019; 0 = backward → `gt*l`).

### P2 — hygiene

15. Tolerance-based 2 m spacing check; no silent `*.csv` fallback; remove dead `smooth_line` in
    `validate_freeboard.py`; write `year/month` into prepared CSV.
16. Optional `--inference-dir` for unlabeled tracks (design decision on the label-dependent feature).

## 6. Running on the author's data after the P0 fixes

```bash
unzip data/IS2_Corrected_data.zip -d data/
python workflow_generator.py --input-dir data/IS2_Corrected_data --skip-autolabel --epochs 50 -o workflow.yml
pegasus-plan --submit -s condorpool -o local workflow.yml
```

Expected: 6 preprocess jobs, 1 prepare (~134 k vectors, 40 features), train/inference, then 4 freeboard
and 4 validate jobs (unique granule+beam: 05940510 gt1r/gt2r, 09290510 gt1r/gt2r), and paper figures for
the two paper tracks.

## 7. Status (applied 2026-09-16)

Applied: P0 items 1–6, P1 items 7–14, P2 item 15. Verified locally on the six author files with the
project venv (TensorFlow 2.14, CPU):

| Stage | Result |
|---|---|
| autolabel_correct on `_done` input | no-op (42154 rows kept) |
| preprocess (no z-score) | 6 enriched files; warning for the no-open-water segment |
| prepare | 4 unique tracks, 134,249 vectors, 48 columns (2 + 6 metadata + 40 features), 694 NaN vectors dropped |
| train (3 epochs only) | test accuracy 0.939, F1 0.937; `test_predictions.csv` written |
| inference | 20 columns, `h_cor_mean` in metres (−0.01 … 3.56 m), all three probabilities kept |
| compute_freeboard × 4 | `new_h_ref` 0.33–0.51 m, `new_h_ref_smooth` and `freeboard_new_h_ref_smooth` present |
| validate / paper_figures | run without ATL07 data; tables I–V now transcribed from the paper |
| workflow_generator | 18 jobs (6 preprocess, prepare, train, inference, 4 freeboard, 4 validate, figures); docker:// containers used because no local SIF exists |

Not applied (needs a decision or data): Horovod image/launcher (item 6, second half), an inference
set of unlabeled tracks (item 16). `pegasus.transfer.container.onhost` triggers an "unrecognized
property" warning from the Python API; harmless.

## 8. Server run (pegasus-submit, 2026-09-17, run0002)

Deployed to `~/seaice-icesat2-workflow` on the `pegasus` submit host (Pegasus 6.0.0.dev0, HTCondor
24.12.22, 5 workers; GPUs on GPN Tesla T4 and NCSA Quadro RTX 6000) and run on the author's six
corrected CSVs (MD5-verified after transfer). Containers were pre-pulled to
`containers/seaice_{cpu,gpu}.sif` and selected with `--container-mode sif`.

**run0001 failed** in `prepare_lstm_data` (exit 2). Pegasus writes job arguments unquoted into the
generated shell wrapper, so `--pattern *_enriched.csv` was glob-expanded by bash into six filenames
and argparse rejected them. Fixed by dropping the flag from the generator (the job directory holds
only the enriched CSVs) and making `prepare_lstm_data.py` skip its own output file.

**run0002: 18 tasks / 39 jobs, 100 % success, no retries, 20 min 44 s wall time.**

| Stage | Jobs | Total runtime |
|---|---|---|
| preprocess_atl03 | 6 | 160 s (max 45 s) |
| prepare_lstm_data | 1 | 6 s |
| train_lstm (GPU, 50 epochs) | 1 | 767 s |
| inference_lstm (GPU) | 1 | 16 s |
| compute_freeboard | 4 | 9 s |
| validate_freeboard | 4 | 10 s |
| paper_figures | 1 | 17 s |

Model (held-out 20 % test split, 26,850 samples of 134,249):

| Metric | This run | Paper (Table III / Fig. 4) |
|---|---|---|
| Accuracy | 95.90 % | 96.56 % |
| Precision | 96.47 % | 97.00 % |
| Recall | 95.25 % | 96.09 % |
| F1 | 95.85 % | 96.54 % |
| Thick ice | 97.30 % | 98.39 % |
| Thin ice | 75.89 % | 73.80 % |
| Open water | 84.22 % | 60.25 % |

Overall accuracy is 0.66 points below the paper on four tracks instead of the paper's larger set;
thin ice and open water, the classes the paper struggled with, are better here (open water +24 points).

Freeboard (ice segments, smoothed NASA sea surface):

| Track | Rows | h_ref (m) | h_ref smoothed (m) | Mean freeboard (m) |
|---|---|---|---|---|
| 20191104195311_05940510_gt1r | 49,451 | 0.426 | 0.258 | 0.553 |
| 20191104195311_05940510_gt2r | 40,429 | 0.413 | 0.300 | 0.535 |
| 20191126182014_09290510_gt1r | 24,365 | 0.621 | 0.475 | 0.946 |
| 20191126182014_09290510_gt2r | 20,004 | 0.780 | 0.476 | 0.895 |

Outputs in `~/seaice-icesat2-workflow/output/` on the server; figures and one validation report
copied to `results/` locally. ATL07/ATL10 comparison panels are empty because no reference data was
staged.

## 9. Container migration (2026-09-17)

Docker was removed. The workflow now builds its own Apptainer images as part of the DAG, which
closes P0 item 6 from section 5 ("Containers missing") without requiring anything to be prepared by
hand before planning.

- `Docker/Dockerfile.{cpu,gpu}` and `requirements-{cpu,gpu}.txt` deleted; replaced by
  `Apptainer/seaice_{cpu,gpu}.def` with the dependency lists inline in `%post`.
- Two `build_container` jobs head the DAG, pinned to the submit host with
  `add_selector_profile(execution_site="local")`. Unprivileged `apptainer build` succeeds there but
  fails on the compute nodes, and building centrally avoids shipping a multi-GB image back from a
  worker before redistributing it.
- Each `.sif` is an ordinary workflow output, so Pegasus stages it only to the jobs that declare it:
  the CPU image to eight stages, the GPU image to `train_lstm` and `inference_lstm` alone.
- Neither definition has a `%files` section. Nothing from the repository is copied into an image;
  `bin/seaice_run.sh` bind-mounts the job directory (`--bind $PWD:/srv --pwd /srv`) so the staged
  `bin/<stage>.py` runs from outside the image. Editing workflow code never invalidates an image.
- `--prebuilt-containers DIR` skips the build jobs and reuses existing images.

Three defects surfaced and were fixed while validating this on the server:

| Failure | Cause | Fix |
|---|---|---|
| Both build jobs exit 255 with no output | HTCondor's local universe provides no `HOME`, and `set -u` aborted the wrapper before its first `echo` | `export HOME="${HOME:-$PWD}"` in both wrappers; cache and tmp directories passed as env profiles |
| `pip install -r /tmp/requirements-cpu.txt`: no such file | Apptainer bind-mounts the host `/tmp` over the container's during `%post`, hiding files staged there by `%files` | `%files` removed entirely; dependencies declared inline |
| Build failure reported as success | `set -e` exited before the `status=$?` check could run | capture with `|| status=$?` |

### Validation run (run0005, 2026-09-17)

The Apptainer-only workflow ran end to end on the author's six tracks: **20 tasks / 41 jobs, 100 %
success, no retries, 27 min 32 s wall time**, with both images built inside the workflow.

| Stage | Jobs | Total runtime |
|---|---|---|
| build_container (CPU 527 MB, GPU 3.6 GB) | 2 | 710 s (GPU 530 s) |
| preprocess_atl03 | 6 | 166 s |
| prepare_lstm_data | 1 | 8 s |
| train_lstm (GPU, 50 epochs) | 1 | 759 s |
| inference_lstm (GPU) | 1 | 16 s |
| compute_freeboard | 4 | 10 s |
| validate_freeboard | 4 | 11 s |
| paper_figures | 1 | 13 s |

Science output is unchanged from the Docker-based run0002, which is the point of the migration:

| Metric | run0002 (Docker images) | run0005 (workflow-built Apptainer) |
|---|---|---|
| Test accuracy | 95.90 % | 95.93 % |
| Test F1 | 95.85 % | 95.94 % |
| Thick ice | 97.30 % | 97.28 % |
| Thin ice | 75.89 % | 80.19 % |
| Open water | 84.22 % | 76.49 % |
| Freeboard, 05940510 gt1r | 0.553 m | 0.529 m |
| Freeboard, 09290510 gt2r | 0.895 m | 0.897 m |

Per-class figures move by a few points between runs because training is stochastic (no seed is fixed
for weight initialisation or dropout); overall accuracy is stable to 0.03 points.

The container builds add roughly 12 minutes to a cold run. `--prebuilt-containers DIR` removes that
cost once the images exist, and Apptainer's layer cache makes a warm rebuild substantially cheaper.

---

A detailed side-by-side of this workflow's measured results against the paper's reported numbers
lives in [`PAPER_COMPARISON.md`](PAPER_COMPARISON.md), together with the list of figures that are
measured versus transcribed, and what data is still needed to close the remaining gaps.

## 10. Horovod (2026-09-17)

`--horovod` now performs real data-parallel training rather than setting a flag nothing acts on.

- `Apptainer/seaice_gpu_horovod.def` adds OpenMPI and Horovod to the TensorFlow base. It installs
  the standalone NCCL wheel and points Horovod's CMake at it, because the TensorFlow image ships
  NCCL headers but no shared library; if that is unavailable the build retries with MPI collectives.
  `horovodrun --check-build` reports NCCL, MPI and Gloo tensor operations.
- `bin/seaice_run.sh --np N` launches the stage under `horovodrun -np N -H localhost:N`.
- The generator selects the Horovod image, sets `request_gpus = --n-gpus`, and warns that all ranks
  must fit on one worker (2 GPUs here) because HTCondor's vanilla universe gives a job one machine.

**Two defects were fixed in `bin/train_lstm.py` before the first real run**, both of which would have
made a Horovod run meaningless rather than merely slow:

| Defect | Consequence |
|---|---|
| The optimizer was never wrapped in `hvd.DistributedOptimizer` | Gradients were never averaged across ranks, so each rank trained an independent model. This is step 3 of the paper's own Horovod integration list |
| The training and validation indices were not sharded per rank | Every rank fitted the identical full split, so N ranks did N times the work for no speedup |

Both are now in place: the optimizer is wrapped when running under Horovod, and each rank takes a
disjoint stride `idx[rank::size]`, so epoch cost scales as 1/size. The learning rate was already
scaled by `hvd.size()`, which is only correct once the effective batch actually grows.

A third defect appeared at run time: `horovodrun` creates `$HOME/.horovod` before launching, and the
container runs with `--no-home`, so the host home is not mounted and the path is read-only. The
wrapper now sets `HOME=/srv`, the bind-mounted job directory, which also keeps matplotlib and
similar caches inside the job sandbox.

### Horovod validation run (run0009, 2026-09-17)

18 tasks / 39 jobs, 100 % success, 19 min 4 s wall time, reusing prebuilt images. Both ranks took
disjoint shards and reported identical test metrics, confirming the DistributedOptimizer averages
gradients rather than leaving two independent models. Measured speedup at 2 GPUs was 1.05x against
the paper's reported 1.96x; the model is far too small for the step to be GPU-bound, so the
all-reduce cancels the benefit of sharding. Details and the suggested remedy are in
`PAPER_COMPARISON.md`.

A third run-time defect was found and fixed here: `apptainer exec --env HOME=...` is refused by
Apptainer ("Overriding HOME environment variable with APPTAINERENV_HOME is not permitted"), so the
first HOME fix silently did nothing and `horovodrun` failed again on the read-only host home. The
wrapper now applies the redirect with `env HOME=/srv` **inside** the container, verified against the
built image before resubmitting.

Note for future recovery work: patching an executable in the workflow scratch and releasing a held
job does not work, because Pegasus records a checksum at stage-in and rejects the modified file.
Replan instead, and pass `--prebuilt-containers` so the container builds are not repeated.
