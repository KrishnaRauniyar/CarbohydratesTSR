#!/usr/bin/env python
"""Evaluate a saved TSR DNN model on a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    ensure_output_dirs,
    evaluate_predictions,
    extract_labels_from_protein,
    load_artifact,
    load_input_dataframe,
    save_predictions,
    transform_with_preprocessor,
)
from visualization import (
    plot_confusion_matrix,
    plot_pca_samples,
    plot_per_class_metric,
    plot_prediction_confidence,
    plot_top_confused_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TSR DNN model.")
    parser.add_argument("--input_csv", required=True, help="CSV to evaluate.")
    parser.add_argument("--model_path", required=True, help="Path to best_model.keras or final_model.keras.")
    parser.add_argument(
        "--artifact_dir",
        required=True,
        help="Directory containing preprocessor.joblib and label_encoder.joblib.",
    )
    parser.add_argument("--output_dir", default="outputs/evaluation", help="Directory for evaluation outputs.")
    parser.add_argument("--label_from", default="protein", help="Column whose final underscore token is the class label.")
    parser.add_argument("--make_embedding_plots", action="store_true", help="Plot PCA of penultimate-layer embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_output_dirs(Path(args.output_dir))
    artifact_dir = Path(args.artifact_dir)

    # Delay TensorFlow import so --help works on systems where the runtime is
    # installed but not loadable until the proper module/conda env is active.
    try:
        from tensorflow.keras.models import Model, load_model
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported. Activate the same working "
            "TensorFlow environment used for training before evaluation."
        ) from exc

    model = load_model(args.model_path)
    preprocessor = load_artifact(artifact_dir / "preprocessor.joblib")
    label_encoder = load_artifact(artifact_dir / "label_encoder.joblib")
    class_names = label_encoder.classes_.tolist()

    df = load_input_dataframe(Path(args.input_csv), args.label_from)
    labels = extract_labels_from_protein(df[args.label_from])
    y = label_encoder.transform(labels)
    X_raw = _numeric_features_from_saved_schema(df, preprocessor)
    X = transform_with_preprocessor(X_raw, preprocessor)

    y_prob = model.predict(X.to_numpy(dtype=np.float32), verbose=0)
    metrics, report_df, cm_df = evaluate_predictions(y, y_prob, class_names, "evaluation")
    pd.DataFrame([metrics]).to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
    report_df.to_csv(paths["metrics"] / "classification_report_evaluation.csv", index=False)
    cm_df.to_csv(paths["metrics"] / "confusion_matrix_evaluation.csv")

    pred_df = save_predictions(
        df[args.label_from].tolist(),
        y,
        y_prob,
        class_names,
        paths["predictions"] / "predictions_evaluation.csv",
    )
    pred_df.loc[~pred_df["is_correct"]].to_csv(paths["predictions"] / "misclassified_samples.csv", index=False)

    cm = cm_df.to_numpy()
    plot_confusion_matrix(cm, class_names, paths["plots"] / "confusion_matrix_evaluation.png")
    plot_confusion_matrix(cm, class_names, paths["plots"] / "confusion_matrix_evaluation_normalized.png", normalized=True)
    plot_per_class_metric(report_df, "precision", paths["plots"] / "per_class_precision_evaluation.png")
    plot_per_class_metric(report_df, "recall", paths["plots"] / "per_class_recall_evaluation.png")
    plot_per_class_metric(report_df, "f1-score", paths["plots"] / "per_class_f1_score_evaluation.png")
    plot_top_confused_pairs(cm, class_names, paths["plots"] / "top_confused_class_pairs_evaluation.png")
    plot_prediction_confidence(pred_df, paths["plots"])

    if args.make_embedding_plots and "penultimate" in [layer.name for layer in model.layers]:
        embedding_model = Model(inputs=model.input, outputs=model.get_layer("penultimate").output)
        embeddings = embedding_model.predict(X.to_numpy(dtype=np.float32), verbose=0)
        embedding_df = pd.DataFrame(embeddings)
        plot_pca_samples(
            embedding_df,
            pred_df["true_label"].tolist(),
            paths["plots"] / "embedding_pca_by_true_class.png",
            "Penultimate-Layer Embedding PCA by True Class",
        )
        plot_pca_samples(
            embedding_df,
            pred_df["predicted_label"].tolist(),
            paths["plots"] / "embedding_pca_by_predicted_class.png",
            "Penultimate-Layer Embedding PCA by Predicted Class",
        )

    print(f"Evaluation complete. Outputs saved to: {Path(args.output_dir).resolve()}")


def _numeric_features_from_saved_schema(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Rebuild raw numeric feature columns expected by the saved preprocessor."""

    columns = list(preprocessor["medians"].keys())
    data = pd.DataFrame(index=df.index)
    for column in columns:
        if column in df.columns:
            data[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            data[column] = np.nan
    return data


if __name__ == "__main__":
    main()
