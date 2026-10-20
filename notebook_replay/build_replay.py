#!/usr/bin/env python3
"""Build verbatim replays of the author's notebooks (Scaled_IS2_Classification_Freeboard).

Cells are copied unedited, in order, except for the minimal patches listed in PATCHES;
each patch is printed when the replay runs, so the deviations are part of the log.

replay_prep.py  (CPU image): NB2 cells 2,10,14,16,19 -> NB3 cells 1,3,4,6,11
                 six *_labeled_10m_done.csv -> ATL03_cor_label_LSTM_ann_prepared.csv
replay_train.py (GPU image): NB3 cells 1,17,18,19,20,21,23,25,37,38
                 prepared CSV -> split -> norm -> cell-37 model -> cell-38 evaluation
Skipped: NB1 (needs S2 GeoTIFFs + rembg), keras-tuner search (27-35), alternative
cell-40 model, NB4-NB6 (read files that are not shipped).
"""
import json, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "../../Scaled_IS2_Classification_Freeboard"
NB2 = "2_ATL03_Modeling_rel_height_iqr_all_2025_shifted_ow_cor_label.ipynb"
NB3 = "3_ATL03_prepare_data_LSTM_training_2025_cor_label.ipynb"

# (notebook, cell, old, new, reason)
PATCHES = [
    (NB2, 2, "import tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\n",
     "", "TensorFlow import dropped in the CPU prep stage; NB2's cor_label chain never uses it"),
    (NB3, 6, "    display(dup_removed)\n    print(\"done\")\n    #break\n    #dup_removed.to_csv(file.replace(dir_path, dest_path).replace(\".csv\", \"_dr.csv\"))\n\n    break\n",
     "    print(\"done\")\n", "debug `display` + `break` before make_input removed; as saved the cell writes nothing"),
]
HEADER = '''import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None
def display(*a, **k): pass
import pandas as _pd
if not hasattr(_pd.DataFrame, "append"):
    # pandas >= 2 removed DataFrame.append; restore its pandas-1.x meaning (concat).
    _pd.DataFrame.append = lambda self, other, ignore_index=False, **k: _pd.concat(
        [self, other], ignore_index=ignore_index)
print("REPLAY of {nbs}; patches applied:")
print("  - all cells: pandas-1.x DataFrame.append restored as concat (removed in pandas 2.0; used in NB3 cells 3 and 11)")
{patch_lines}
'''

def cells(nb, idx):
    d = json.load(open(os.path.join(SRC, nb)))
    out = []
    for i in idx:
        c = d["cells"][i]
        assert c["cell_type"] == "code", (nb, i)
        src = "".join(c["source"])
        for pnb, pi, old, new, _ in PATCHES:
            if pnb == nb and pi == i:
                assert src.count(old) == 1, f"patch anchor not found: {nb}[{i}]"
                src = src.replace(old, new)
        src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("!", "%")))
        out.append(f"# ===== {nb} cell {i} =====\n{src}\n")
    return out

def write(path, parts, patches, footer=""):
    plines = "\n".join(f'print("  - {nb[:1]}NB cell {i}: {why}")' for nb, i, _, _, why in patches) or 'print("  (none)")'
    with open(path, "w") as f:
        f.write(HEADER.format(nbs=path, patch_lines=plines) + "\n".join(parts) + footer)

prep = cells(NB2, [2, 10, 14, 16, 19]) + cells(NB3, [1, 3, 4, 6, 11])
write("replay_prep.py", prep, PATCHES)
footer = '''
import json as _json
_json.dump({"test_accuracy": float(test_acc), "test_f1": float(test_f1),
            "test_precision": float(test_precision), "test_recall": float(test_recall),
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
           open("replay_metrics.json", "w"), indent=2)
from sklearn.metrics import confusion_matrix as _cm
_pred = model.predict(X_test, verbose=0).argmax(1); _true = np.asarray(Y_test).argmax(1)
print("REPLAY confusion (rows truth thick/thin/water):", _cm(_true, _pred, labels=[0, 1, 2]).tolist())
print("REPLAY metrics:", open("replay_metrics.json").read())
'''
# The six shared files include T02CNA_gt2r (698 rows, all thick ice): no open water, so its
# sea surface is NaN everywhere and NB2's interpolation cannot fill it. The notebook has no
# NaN handling; with those 693 windows the loss is NaN from the first epoch and the model
# predicts thick ice everywhere (accuracy = thick-ice share). Applied to both arms.
PATCHES.append((NB3, 18, "lstm_features = dataset_accu.copy()\n",
                "lstm_features = dataset_accu.dropna().copy()\n",
                "rows with NaN features dropped (693 windows of T02CNA_gt2r, a 698-row all-thick-ice "
                "file with no open water); without this the loss is NaN and every window is predicted thick ice"))

# Notebook inconsistency: cell 20 builds 8 features per timestep but cell 37 declares
# (None, 5, 6), so the saved cells cannot run together. Two arms, one patched line each.
ARMS = {
    "8feat": (NB3, 37, "batch_input_shape=(None, 5, 6)", "batch_input_shape=(None, 5, 8)",
              "cell 37 input shape (5,6) -> (5,8) to match cells 6/20 as saved (8 features)"),
    "6feat": (NB3, 20, "  point_r2 =  row[1:9]\n  point_r1 =  row[9:17]\n  point_0 = row[17:25]\n"
                       "  point_l1 = row[25:33]\n  point_l2 = row[33:41]\n",
              "  point_r2 =  row[1:7]\n  point_r1 =  row[9:15]\n  point_0 = row[17:23]\n"
              "  point_l1 = row[25:31]\n  point_l2 = row[33:39]\n",
              "cell 20 keeps the first 6 of 8 features per timestep (drops bcnt_mean, brate_mean) "
              "to match cell 37's (5,6)"),
}
for arm, patch in ARMS.items():
    PATCHES.append(patch)
    write(f"replay_train_{arm}.py", cells(NB3, [1, 17, 18, 19, 20, 21, 23, 25, 37, 38]),
          [p for p in PATCHES if p[0] == NB3 and p[1] in (18, 20, 37)], footer)
    PATCHES.pop()
print("wrote replay_prep.py, replay_train_{6feat,8feat}.py")
