#!/usr/bin/env python3
"""
Train the LSTM classifier for 3-class sea ice classification (thick ice / thin ice / open water).

Source: Notebook 3 — cells 18-23 (split), 19 (norm), 20 (reshape), 25 (metrics),
        37 (tuned model; "notebook" preset). The "paper" preset is the architecture described in
        Section III.B of Iqrah et al. (IPDPSW 2025).

Fixes relative to the notebook / earlier workflow version:
- Feature columns are selected BY NAME in a fixed timestep-major order; metadata columns
  (track, x_atc, day, lon, lat, ...) are never fed to the model.
- Normalisation parameters (and the feature list) are saved so inference uses the training
  statistics instead of recomputing them on the inference set.
- Rows with NaN features/labels are dropped.
- Held-out test predictions are written so downstream figures use the test split, not the
  whole dataset.
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "h_cor_mean", "h_diff", "rel_height_min_elev", "height_sd",
    "pcnth_mean", "pcnt_mean", "bcnt_mean", "brate_mean",
]
N_CLASSES = 3
CLASS_NAMES = ["thick_ice", "thin_ice", "open_water"]

# Model presets. "notebook" = NB3 cell 37 (tuned, used for the cor_label results);
# "paper" = architecture stated in the paper (LSTM 16 ELU, dropout 0.2, 7 dense layers, Adam 0.003).
PRESETS = {
    "notebook": dict(units=48, lstm_activation="tanh", dropout=0.4,
                     dense_units="16,16", dense_activation="elu",
                     lr=0.000889, alpha="0.05,0.45,0.60", epochs=50),
    "paper": dict(units=16, lstm_activation="elu", dropout=0.2,
                  dense_units="32,96,32,16,112,48,64", dense_activation="elu",
                  lr=0.003, alpha="0.05,0.45,0.60", epochs=20),
}


# ---------------------------------------------------------------------------
# Custom metrics (Notebook 3, cell 25)
# ---------------------------------------------------------------------------

def recall_m(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))


# ---------------------------------------------------------------------------
# Feature handling
# ---------------------------------------------------------------------------

def feature_columns(features, nearby):
    """Timestep-major column order, identical to prepare_lstm_data.py."""
    return [f"{f}{t}" for t in range(-nearby, nearby + 1) for f in features]


def select_features(df, features, nearby):
    cols = feature_columns(features, nearby)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Prepared CSV is missing feature columns: {missing[:5]}"
                         f"{' ...' if len(missing) > 5 else ''}")
    return df[cols]


def compute_norm_params(features_df, formula):
    return {
        "formula": formula,
        "mean": features_df.mean().to_dict(),
        "std": features_df.std().to_dict(),
    }


def normalize(features_df, norm_params):
    """Notebook formula (x - mean) / (1 - std) by default, or a standard z-score."""
    means = pd.Series(norm_params["mean"])[features_df.columns]
    stds = pd.Series(norm_params["std"])[features_df.columns]
    if norm_params.get("formula", "notebook") == "zscore":
        return (features_df - means) / stds.replace(0, 1.0)
    return (features_df - means) / (1 - stds)


def reshape_to_lstm(np_array, n_timesteps, n_features):
    """(N, timesteps*features) -> (N, timesteps, features); columns are timestep-major."""
    assert np_array.shape[1] == n_timesteps * n_features, (
        f"expected {n_timesteps * n_features} feature columns, got {np_array.shape[1]}")
    return np_array.reshape(len(np_array), n_timesteps, n_features)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_lstm_model(n_timesteps, n_features, units, lstm_activation, dropout,
                     dense_units, dense_activation, lr, alpha, n_classes=N_CLASSES,
                     hvd=None):
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense

    model = Sequential()
    model.add(LSTM(units, activation=lstm_activation, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(dropout))
    for du in dense_units:
        model.add(Dense(du, activation=dense_activation))
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation="softmax"))

    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=np.array(alpha, dtype=float), gamma=2.0, from_logits=False,
        label_smoothing=0.0, axis=-1, reduction="sum_over_batch_size",
        name="categorical_focal_crossentropy",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if hvd is not None:
        # Step 3 of the paper's Horovod integration: the wrapped optimizer averages
        # gradients across ranks with a ring all-reduce. Without this the ranks
        # would train independent models and nothing would be distributed.
        optimizer = hvd.DistributedOptimizer(optimizer)
    model.compile(optimizer=optimizer, loss=focal_loss,
                  metrics=["accuracy", f1_m, precision_m, recall_m])
    return model


def _parse_list(s, cast):
    return [cast(v) for v in str(s).split(",") if str(v).strip() != ""]


def main():
    parser = argparse.ArgumentParser(description="Train LSTM sea ice classifier")
    parser.add_argument("--data", required=True, help="Prepared CSV from prepare_lstm_data.py")
    parser.add_argument("--model-output", required=True, help="Output model file (.h5)")
    parser.add_argument("--norm-params-output", default="norm_params.json")
    parser.add_argument("--metrics-output", default="training_metrics.json")
    parser.add_argument("--test-predictions-output", default="test_predictions.csv",
                        help="Held-out test-split predictions CSV")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                        help="Comma-separated per-timestep feature names")
    parser.add_argument("--nearby", type=int, default=2, help="Neighbours per side (default: 2)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="notebook",
                        help="Architecture preset; explicit flags below override it")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units", type=int, default=None, help="LSTM units")
    parser.add_argument("--lstm-activation", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dense-units", default=None, help="Comma-separated dense layer sizes")
    parser.add_argument("--dense-activation", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", default=None, help="Focal-loss class weights, comma-separated")
    parser.add_argument("--norm", choices=["notebook", "zscore"], default="notebook",
                        help="Normalisation formula (default: notebook (x-mean)/(1-std))")
    parser.add_argument("--seed", type=int, default=20, help="Split random state (notebook: 20)")
    parser.add_argument("--horovod", action="store_true", help="Use Horovod distributed training")
    args = parser.parse_args()

    cfg = dict(PRESETS[args.preset])
    for key in ("epochs", "units", "lstm_activation", "dropout", "dense_units",
                "dense_activation", "lr", "alpha"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    dense_units = _parse_list(cfg["dense_units"], int)
    alpha = _parse_list(cfg["alpha"], float)
    features = [f.strip() for f in args.features.split(",")]
    n_timesteps = 2 * args.nearby + 1
    n_features = len(features)
    logger.info("Preset %s -> %s", args.preset, cfg)

    # Horovod setup
    hvd = None
    if args.horovod:
        try:
            import horovod.tensorflow.keras as hvd
        except ImportError as e:
            raise SystemExit("--horovod requested but horovod is not installed in this image") from e
        hvd.init()
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
        logger.info("Horovod rank %d/%d", hvd.rank(), hvd.size())

    # Load data
    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "label" not in df.columns:
        raise SystemExit("Prepared CSV has no 'label' column")

    feat_df = select_features(df, features, args.nearby)
    meta_cols = [c for c in df.columns if c not in feat_df.columns and c != "label"]
    logger.info("Feature columns: %d, metadata columns: %s", feat_df.shape[1], meta_cols)

    valid = ~(feat_df.isna().any(axis=1) | df["label"].isna())
    if (~valid).any():
        logger.warning("Dropping %d rows with NaN features/labels", int((~valid).sum()))
    df = df[valid].reset_index(drop=True)
    feat_df = feat_df[valid].reset_index(drop=True)
    labels = df["label"].astype(int).to_numpy()
    logger.info("Samples: %d, label counts: %s", len(df),
                dict(zip(*np.unique(labels, return_counts=True))))

    # Normalisation params from the full labelled set (as in the notebook), saved for inference
    norm_params = compute_norm_params(feat_df, args.norm)
    norm_params.update({"features": features, "nearby": args.nearby,
                        "feature_columns": list(feat_df.columns)})
    with open(args.norm_params_output, "w") as f:
        json.dump(norm_params, f, indent=2)
    logger.info("Saved normalisation params to %s", args.norm_params_output)

    X = reshape_to_lstm(normalize(feat_df, norm_params).to_numpy(dtype=np.float32),
                        n_timesteps, n_features)
    from tensorflow.keras.utils import to_categorical
    Y = to_categorical(labels, num_classes=N_CLASSES)
    logger.info("LSTM input shape: %s, labels shape: %s", X.shape, Y.shape)

    # Train/val/test split 60/20/20 on indices (Notebook 3, cell 23)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.2, shuffle=True, random_state=args.seed)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25, shuffle=True,
                                          random_state=args.seed)
    logger.info("Train: %d, Val: %d, Test: %d", len(idx_train), len(idx_val), len(idx_test))

    # Data parallelism: each rank trains on a disjoint shard, so an epoch costs
    # roughly 1/size of the single-GPU epoch. Gradients are averaged by the
    # DistributedOptimizer, so the effective batch is batch_size * size, which is
    # why the learning rate is scaled by the same factor.
    if hvd is not None:
        n_before = len(idx_train)
        idx_train = idx_train[hvd.rank()::hvd.size()]
        idx_val = idx_val[hvd.rank()::hvd.size()]
        logger.info("Horovod rank %d/%d: training on %d of %d windows (val %d)",
                    hvd.rank(), hvd.size(), len(idx_train), n_before, len(idx_val))

    lr = cfg["lr"] * (hvd.size() if hvd is not None else 1)
    model = build_lstm_model(n_timesteps, n_features, cfg["units"], cfg["lstm_activation"],
                             cfg["dropout"], dense_units, cfg["dense_activation"], lr, alpha,
                             hvd=hvd)
    model.summary(print_fn=logger.info)

    callbacks = []
    if hvd is not None:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

    logger.info("Training for %d epochs (batch %d)...", cfg["epochs"], args.batch_size)
    begin = time.time()
    history = model.fit(
        X[idx_train], Y[idx_train],
        epochs=cfg["epochs"], batch_size=args.batch_size,
        validation_data=(X[idx_val], Y[idx_val]),
        callbacks=callbacks,
        verbose=2 if (hvd is None or hvd.rank() == 0) else 0,
    )
    train_time = time.time() - begin
    logger.info("Training time: %.1f seconds", train_time)

    test_loss, test_acc, test_f1, test_prec, test_rec = model.evaluate(
        X[idx_test], Y[idx_test], verbose=0)
    logger.info("Test — Acc: %.4f, F1: %.4f, Prec: %.4f, Rec: %.4f",
                test_acc, test_f1, test_prec, test_rec)

    if hvd is None or hvd.rank() == 0:
        model.save(args.model_output)
        logger.info("Model saved to %s", args.model_output)

        # Held-out test predictions + sklearn report
        from sklearn.metrics import classification_report, confusion_matrix
        probs = model.predict(X[idx_test], verbose=0)
        pred = probs.argmax(axis=1)
        y_test = labels[idx_test]
        test_df = df.loc[idx_test, [c for c in ["track"] + meta_cols if c in df.columns]].copy()
        test_df = test_df.loc[:, ~test_df.columns.duplicated()]
        test_df["label"] = y_test
        test_df["pred_label"] = pred
        for k in range(N_CLASSES):
            test_df[f"pred_label{k}"] = probs[:, k]
        test_df.to_csv(args.test_predictions_output, index=False)
        logger.info("Test predictions saved to %s (%d rows)", args.test_predictions_output, len(test_df))

        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        report = classification_report(y_test, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        print(classification_report(y_test, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                    zero_division=0))

        metrics = {
            "preset": args.preset,
            "config": {**cfg, "dense_units": dense_units, "alpha": alpha,
                       "batch_size": args.batch_size, "norm": args.norm, "seed": args.seed},
            "features": features,
            "n_samples": int(len(df)),
            "n_train": int(len(idx_train)), "n_val": int(len(idx_val)), "n_test": int(len(idx_test)),
            "horovod_size": int(hvd.size()) if hvd is not None else 1,
            "epochs": int(cfg["epochs"]),
            "train_time_seconds": train_time,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_confusion_matrix": cm.tolist(),
            "test_classification_report": report,
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()
                        if k in ("accuracy", "val_accuracy", "loss", "val_loss")},
        }
        with open(args.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", args.metrics_output)


if __name__ == "__main__":
    main()
