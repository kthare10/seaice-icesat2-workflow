# Replay of the author's notebooks

`build_replay.py` copies the author's NB2 and NB3 cells (from
`Scaled_IS2_Classification_Freeboard/`) unedited, in order, into two scripts, with
the minimal patches listed below. Every patch is printed at the top of each run's log.

```bash
python3 build_replay.py ../../Scaled_IS2_Classification_Freeboard   # writes replay_*.py
./run_prep.sh ~/nb_replay data/IS2_Corrected_data containers/seaice_cpu.sif
cp replay_train_*.py train_chain.sh ~/nb_replay/ && (cd ~/nb_replay && ./train_chain.sh)
```

| Stage | Cells | Image |
|---|---|---|
| prep | NB2 2, 10, 14, 16, 19; NB3 1, 3, 4, 6, 11 | `seaice_cpu.sif` (geopandas) |
| train | NB3 1, 17-21, 23, 25, 37 (model), 38 (evaluation) | `seaice_gpu.sif` (TensorFlow 2.14) |

Patches (the saved notebooks cannot run as-is):

- pandas-1.x `DataFrame.append` restored as `concat` (removed in pandas 2.0; NB3 cells 3, 11).
- NB2 cell 2: unused TensorFlow import dropped in the CPU prep stage.
- NB3 cell 6: a debug `display` + `break` before `make_input` removed; as saved it writes nothing.
- NB3 cell 18: rows with NaN features dropped. `T02CNA_gt2r` (698 rows, all thick ice) has no
  open water, so its sea surface is NaN; without this the loss is NaN and every window is
  predicted thick ice.
- NB3 cells 20/37 disagree: cell 20 builds 8 features per timestep, cell 37 declares `(5, 6)`.
  Two arms, one patched line each: `6feat` (cell 20 keeps the first six, dropping `bcnt_mean`
  and `brate_mean`) and `8feat` (cell 37 widened to `(5, 8)`).

Not replayable: NB1 (needs Sentinel-2 GeoTIFFs and `rembg`), the keras-tuner search (NB3 27-35),
NB4-NB6 (read files that are not shipped).
