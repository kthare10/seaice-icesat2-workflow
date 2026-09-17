# Sea Ice Classification & Freeboard Pegasus Workflow — Implementation Plan

## Context

The sea ice pipeline (Iqrah et al., IPDPSW 2025) currently lives in 6 Jupyter notebooks inside
`Scaled_IS2_Classification_Freeboard/`. We convert these into a production Pegasus WMS workflow
at `seaice-workflow/` with standalone CLI scripts, Apptainer definitions, and a class-based workflow generator.

## Files to Create (16 total)

```
seaice-workflow/
├── workflow_generator.py
├── bin/
│   ├── download_data.py          # Pre-workflow data download utility
│   ├── extract_atl03.py          # HDF5 → per-beam CSV extraction
│   ├── autolabel_s2.py           # HSV-based labeling via Sentinel-2 imagery
│   ├── autolabel_correct.py
│   ├── preprocess_atl03.py
│   ├── prepare_lstm_data.py
│   ├── train_lstm.py
│   ├── inference_lstm.py
│   ├── compute_freeboard.py
│   ├── validate_freeboard.py
│   ├── build_container.sh      # Builds a .sif from a .def, as a workflow job
│   └── seaice_run.sh           # Runs a staged stage script inside a staged .sif
├── Apptainer/
│   ├── seaice_cpu.def          # Dependencies only - no %files section
│   └── seaice_gpu.def
└── README.md
```

## DAG Structure

```
build_container[CPU,GPU] (submit host) → images staged to the jobs that need them

autolabel_correct[N] → preprocess_atl03[N] → prepare_lstm_data (fan-in)
→ train_lstm (GPU) → inference_lstm (GPU) → compute_freeboard[T] → validate_freeboard[T]
→ paper_figures (fan-in)          (N = per input file, T = per unique granule+beam track)
```

## Key Design Decisions

1. **8 features per timestep**: h_cor_mean, h_diff, rel_height_min_elev, height_sd, pcnth_mean, pcnt_mean, bcnt_mean, brate_mean. LSTM input shape: (N, 5, 8).
2. **Eliminate global data_copy** — pass DataFrames as explicit input_data parameters.
3. **Fix normalization bug** — save norm params at training time, reload at inference.
4. **Replace deprecated df.append()** with pd.concat() throughout.
5. **Apptainer containers built by the workflow itself** (two `build_container` jobs on the
   submit host). The images carry dependencies only; stage scripts are bind-mounted at run time,
   so code changes never force a rebuild. GPU stages run with `--nv`.
6. **CondorIO** for all data transfers.
