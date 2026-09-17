#!/usr/bin/env python3
"""
LSTM batch inference for sea ice classification.

Source: Notebook 3 cells 44-49 and Notebook 4 cells 4-9 (predict, argmax, re-attach the RAW
centre-point features and metadata).

Fixes relative to the notebook / earlier workflow version:
- Uses the normalisation statistics and feature list saved by train_lstm.py.
- Selects feature columns by name; metadata (track, x_atc, day, lon, lat, ...) is passed through.
- Re-attaches the un-normalised centre-point features (h_cor_mean, height_sd, ...) so that
  compute_freeboard.py works in metres, as the notebook did.
- Keeps all three class probabilities (pred_label0/1/2).
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "h_cor_mean", "h_diff", "rel_height_min_elev", "height_sd",
    "pcnth_mean", "pcnt_mean", "bcnt_mean", "brate_mean",
]
CLASS_NAMES = ["thick_ice", "thin_ice", "open_water"]


def feature_columns(features, nearby):
    return [f"{f}{t}" for t in range(-nearby, nearby + 1) for f in features]


def normalize(features_df, norm_params):
    means = pd.Series(norm_params["mean"])[features_df.columns]
    stds = pd.Series(norm_params["std"])[features_df.columns]
    if norm_params.get("formula", "notebook") == "zscore":
        return (features_df - means) / stds.replace(0, 1.0)
    return (features_df - means) / (1 - stds)


# Custom metrics needed to deserialise the model
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


def main():
    parser = argparse.ArgumentParser(description="LSTM sea ice inference")
    parser.add_argument("--model", required=True, help="Trained model file (.h5)")
    parser.add_argument("--data", required=True, help="Prepared CSV with feature vectors")
    parser.add_argument("--norm-params", required=True, help="norm_params.json from training")
    parser.add_argument("--output", required=True, help="Output predictions CSV")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    import tensorflow as tf

    with open(args.norm_params) as f:
        norm_params = json.load(f)
    features = norm_params.get("features", DEFAULT_FEATURES)
    nearby = int(norm_params.get("nearby", 2))
    fcols = norm_params.get("feature_columns") or feature_columns(features, nearby)
    n_timesteps, n_features = 2 * nearby + 1, len(features)

    logger.info("Loading model from %s", args.model)
    model = tf.keras.models.load_model(
        args.model, custom_objects={"recall_m": recall_m, "precision_m": precision_m, "f1_m": f1_m})

    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    missing = [c for c in fcols if c not in df.columns]
    if missing:
        raise SystemExit(f"Prepared CSV is missing feature columns: {missing[:5]}")

    feat_df = df[fcols]
    valid = ~feat_df.isna().any(axis=1)
    if (~valid).any():
        logger.warning("Dropping %d rows with NaN features", int((~valid).sum()))
        df = df[valid].reset_index(drop=True)
        feat_df = feat_df[valid].reset_index(drop=True)

    X = normalize(feat_df, norm_params).to_numpy(dtype=np.float32)
    X = X.reshape(len(X), n_timesteps, n_features)
    logger.info("LSTM input shape: %s", X.shape)

    logger.info("Running inference...")
    prediction = model.predict(X, batch_size=args.batch_size, verbose=0)

    # Output: metadata + RAW centre-point features + probabilities + argmax label
    meta_cols = [c for c in df.columns if c not in fcols]
    out = df[meta_cols].copy()
    for f in features:
        out[f] = df[f"{f}0"].to_numpy()
    for k in range(prediction.shape[1]):
        out[f"pred_label{k}"] = prediction[:, k]
    out["pred_label"] = np.argmax(prediction, axis=1)

    out.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s (%d rows, %d columns)", args.output, len(out), len(out.columns))

    if "label" in out.columns and out["label"].notna().any():
        from sklearn.metrics import accuracy_score, classification_report
        lab = out["label"].notna()
        acc = accuracy_score(out.loc[lab, "label"].astype(int), out.loc[lab, "pred_label"].astype(int))
        logger.info("Accuracy on all labelled rows (includes training data; see "
                    "training_metrics.json for the held-out test score): %.4f", acc)
        print(classification_report(out.loc[lab, "label"].astype(int),
                                    out.loc[lab, "pred_label"].astype(int),
                                    labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
