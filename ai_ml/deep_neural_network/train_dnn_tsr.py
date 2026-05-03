#!/usr/bin/env python
"""Train an improved DNN classifier for TSR key-frequency feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import (
    apply_training_balance,
    coerce_features_to_numeric,
    ensure_output_dirs,
    evaluate_predictions,
    extract_labels_from_protein,
    filter_min_class_count,
    fit_preprocessor,
    initial_feature_columns,
    load_input_dataframe,
    make_preprocessing_report,
    print_device_information,
    remove_duplicate_rows,
    report_to_dict,
    save_artifact,
    save_json,
    save_predictions,
    save_split_statistics,
    set_global_seed,
    stratified_train_val_test_split,
    transform_with_preprocessor,
)
from visualization import (
    plot_balance_distribution,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_frequency_summary,
    plot_pca_samples,
    plot_per_class_metric,
    plot_prediction_confidence,
    plot_split_distributions,
    plot_top_confused_pairs,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DNN classifier for protein/ligand TSR key-frequency features."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV with protein and TSR key-frequency columns.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for models, metrics, plots, and logs.")
    parser.add_argument("--label_from", default="protein", help="Column whose final underscore token is the class label.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Held-out test fraction.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation fraction used for early stopping.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout probability in hidden layers.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument(
        "--balance_method",
        default="class_weight",
        choices=["none", "class_weight", "smote", "smoteenn", "random_oversample"],
        help="Training-set-only imbalance handling strategy.",
    )
    parser.add_argument("--use_log1p", action="store_true", help="Apply log1p to count/frequency features.")
    parser.add_argument(
        "--scaler",
        default="standard",
        choices=["standard", "robust", "none"],
        help="Feature scaler fitted on the training split only.",
    )
    parser.add_argument(
        "--min_class_count",
        type=int,
        default=3,
        help="Remove classes with fewer than this many samples before splitting.",
    )
    parser.add_argument("--smote_k_neighbors", type=int, default=5, help="Requested SMOTE k_neighbors.")
    parser.add_argument(
        "--rare_feature_min_total",
        type=float,
        default=0.0,
        help="Optionally remove features whose training-set total frequency is below this value.",
    )
    parser.add_argument(
        "--hidden_units",
        default="512,256,128,64",
        help="Comma-separated hidden layer widths before the penultimate layer.",
    )
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"], help="Optimizer.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience on validation loss.")
    parser.add_argument(
        "--top_n_plots",
        type=int,
        default=50,
        help="Maximum number of classes/features shown in crowded bar plots.",
    )
    parser.add_argument(
        "--make_pca",
        action="store_true",
        help="Save a PCA visualization of processed samples colored by class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    paths = ensure_output_dirs(output_dir)
    device_info = print_device_information()

    config = vars(args).copy()
    config["input_csv"] = str(input_csv)
    config["output_dir"] = str(output_dir)
    config["device_info"] = device_info
    save_json(config, paths["root"] / "config.json")
    save_json(config, paths["config"] / "config.json")

    # TensorFlow imports are intentionally delayed until after argument parsing
    # so --help and lightweight inspection work even on login nodes or local
    # environments where TensorFlow is not fully configured.
    try:
        from model import MacroF1Callback, build_tsr_dnn
        from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported. Activate a working TensorFlow "
            "environment before training, or install the dependencies in "
            "deep_neural_network/requirements.txt."
        ) from exc

    df_original = load_input_dataframe(input_csv, args.label_from)
    input_rows = len(df_original)
    input_columns = len(df_original.columns)
    original_labels = extract_labels_from_protein(df_original[args.label_from])
    plot_class_distribution(
        original_labels,
        paths["plots"] / "class_distribution_before_splitting.png",
        "Class Distribution Before Splitting",
        top_n=args.top_n_plots,
    )

    df, duplicate_rows_removed = remove_duplicate_rows(df_original)
    rows_after_duplicate_removal = len(df)
    labels = extract_labels_from_protein(df[args.label_from])
    df, labels, removed_classes = filter_min_class_count(df, labels, args.min_class_count)
    rows_after_min_class_filter = len(df)
    labels = labels.reset_index(drop=True)
    df = df.reset_index(drop=True)

    if labels.nunique() < 2:
        raise ValueError("At least two classes are required after --min_class_count filtering.")

    class_counts = labels.value_counts().sort_index().rename_axis("label").reset_index(name="count")
    class_counts["percent"] = class_counts["count"] / class_counts["count"].sum() * 100.0
    class_counts.to_csv(paths["metrics"] / "class_counts_after_filtering.csv", index=False)

    candidates = initial_feature_columns(df, args.label_from)
    if not candidates:
        raise ValueError("No candidate feature columns were found.")

    X_numeric_raw, non_numeric_removed, all_missing_removed, missing_values_before_fill = coerce_features_to_numeric(
        df, candidates
    )
    if X_numeric_raw.shape[1] == 0:
        raise ValueError("No numeric TSR feature columns remain after validation.")

    plot_feature_frequency_summary(X_numeric_raw, paths["plots"], top_n=args.top_n_plots)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_.tolist()

    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        df=df,
        labels_encoded=y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    split_labels = {
        "train": labels.iloc[train_idx].tolist(),
        "validation": labels.iloc[val_idx].tolist(),
        "test": labels.iloc[test_idx].tolist(),
    }
    save_split_statistics(split_labels, paths["metrics"] / "split_class_statistics.csv")
    plot_split_distributions(split_labels, paths["plots"] / "class_distribution_train_validation_test.png", args.top_n_plots)

    X_train_raw = X_numeric_raw.iloc[train_idx].reset_index(drop=True)
    X_val_raw = X_numeric_raw.iloc[val_idx].reset_index(drop=True)
    X_test_raw = X_numeric_raw.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    preprocessor_state, X_train = fit_preprocessor(
        X_train_raw,
        use_log1p=args.use_log1p,
        scaler_name=args.scaler,
        rare_feature_min_total=args.rare_feature_min_total,
    )
    X_val = transform_with_preprocessor(X_val_raw, preprocessor_state)
    X_test = transform_with_preprocessor(X_test_raw, preprocessor_state)

    preprocessing_report = make_preprocessing_report(
        input_rows=input_rows,
        rows_after_duplicate_removal=rows_after_duplicate_removal,
        rows_after_min_class_filter=rows_after_min_class_filter,
        input_columns=input_columns,
        initial_feature_count=len(candidates),
        non_numeric_removed=non_numeric_removed,
        all_missing_removed=all_missing_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        missing_values_before_fill=missing_values_before_fill,
        classes_removed_by_min_count=removed_classes,
        preprocessor_state=preprocessor_state,
    )
    save_json(report_to_dict(preprocessing_report), paths["metrics"] / "processed_dataset_summary.json")
    pd.Series(preprocessor_state["selected_features"], name="feature").to_csv(
        paths["metrics"] / "selected_features.csv", index=False
    )

    X_train_balanced, y_train_balanced, class_weight, balance_report = apply_training_balance(
        X_train,
        y_train,
        method=args.balance_method,
        random_state=args.random_state,
        smote_k_neighbors=args.smote_k_neighbors,
    )
    _save_balance_outputs(balance_report, class_names, paths)
    plot_balance_distribution(
        balance_report["before"],
        balance_report["after"],
        class_names,
        paths["plots"] / "class_distribution_before_after_balancing.png",
        args.top_n_plots,
    )

    hidden_units = tuple(int(value.strip()) for value in args.hidden_units.split(",") if value.strip())
    model = build_tsr_dnn(
        input_dim=X_train.shape[1],
        num_classes=len(class_names),
        hidden_units=hidden_units,
        dropout=args.dropout,
        l2_strength=args.l2,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
    )
    with (paths["models"] / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, args.patience // 3), verbose=1),
        ModelCheckpoint(paths["models"] / "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
        MacroF1Callback(validation_data=(X_val.to_numpy(dtype=np.float32), y_val)),
        CSVLogger(paths["logs"] / "training_log.csv"),
    ]

    history = model.fit(
        X_train_balanced.to_numpy(dtype=np.float32),
        y_train_balanced,
        validation_data=(X_val.to_numpy(dtype=np.float32), y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    model.save(paths["models"] / "final_model.keras")
    save_artifact(preprocessor_state, paths["models"] / "preprocessor.joblib")
    save_artifact(label_encoder, paths["models"] / "label_encoder.joblib")
    save_json({"class_names": class_names}, paths["config"] / "classes.json")
    if class_weight:
        save_json({class_names[int(k)]: v for k, v in class_weight.items()}, paths["config"] / "class_weights.json")

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(paths["metrics"] / "training_history.csv", index=False)
    plot_training_history(history_df, paths["plots"])

    metrics_summary = []
    split_payloads = {
        "train": (X_train, y_train, df.iloc[train_idx][args.label_from].tolist()),
        "validation": (X_val, y_val, df.iloc[val_idx][args.label_from].tolist()),
        "test": (X_test, y_test, df.iloc[test_idx][args.label_from].tolist()),
    }
    prediction_frames = {}
    for split_name, (X_split, y_split, proteins) in split_payloads.items():
        metrics, report_df, cm_df, pred_df = evaluate_split(
            model=model,
            X=X_split,
            y=y_split,
            proteins=proteins,
            class_names=class_names,
            split_name=split_name,
            paths=paths,
        )
        metrics_summary.append(metrics)
        prediction_frames[split_name] = pred_df
        if split_name == "test":
            cm = cm_df.to_numpy()
            plot_confusion_matrix(cm, class_names, paths["plots"] / "confusion_matrix_test.png")
            plot_confusion_matrix(
                cm,
                class_names,
                paths["plots"] / "confusion_matrix_test_normalized.png",
                normalized=True,
            )
            plot_per_class_metric(report_df, "precision", paths["plots"] / "per_class_precision_test.png", args.top_n_plots)
            plot_per_class_metric(report_df, "recall", paths["plots"] / "per_class_recall_test.png", args.top_n_plots)
            plot_per_class_metric(report_df, "f1-score", paths["plots"] / "per_class_f1_score_test.png", args.top_n_plots)
            plot_top_confused_pairs(cm, class_names, paths["plots"] / "top_confused_class_pairs_test.png")
            plot_prediction_confidence(pred_df, paths["plots"])

    pd.DataFrame(metrics_summary).to_csv(paths["metrics"] / "metrics_summary.csv", index=False)

    if args.make_pca:
        all_X = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
        all_labels = split_labels["train"] + split_labels["validation"] + split_labels["test"]
        plot_pca_samples(all_X, all_labels, paths["plots"] / "pca_samples_by_true_class.png", "PCA of Processed TSR Features")
        test_pred_labels = prediction_frames["test"]["predicted_label"].tolist()
        plot_pca_samples(
            X_test,
            prediction_frames["test"]["true_label"].tolist(),
            paths["plots"] / "pca_test_samples_by_true_class.png",
            "PCA of Test TSR Features by True Class",
            predicted_labels=test_pred_labels,
        )

    print("Training complete.")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print("Best model:", (paths["models"] / "best_model.keras").resolve())


def evaluate_split(
    *,
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    proteins,
    class_names,
    split_name: str,
    paths: Dict[str, Path],
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_prob = model.predict(X.to_numpy(dtype=np.float32), verbose=0)
    metrics, report_df, cm_df = evaluate_predictions(y, y_prob, class_names, split_name)
    report_df.to_csv(paths["metrics"] / f"classification_report_{split_name}.csv", index=False)
    cm_df.to_csv(paths["metrics"] / f"confusion_matrix_{split_name}.csv")
    pred_df = save_predictions(
        proteins,
        y,
        y_prob,
        class_names,
        paths["predictions"] / f"predictions_{split_name}.csv",
    )
    if split_name == "test":
        pred_df.loc[~pred_df["is_correct"]].to_csv(
            paths["predictions"] / "misclassified_samples.csv", index=False
        )
        cm_df.to_csv(paths["metrics"] / "confusion_matrix.csv")
    return metrics, report_df, cm_df, pred_df


def _save_balance_outputs(balance_report: Dict, class_names, paths: Dict[str, Path]) -> None:
    save_json(balance_report, paths["metrics"] / "balance_report.json")
    rows = []
    for stage in ["before", "after"]:
        for encoded_label, count in balance_report[stage].items():
            rows.append(
                {
                    "stage": stage,
                    "label": class_names[int(encoded_label)],
                    "encoded_label": int(encoded_label),
                    "count": int(count),
                }
            )
    pd.DataFrame(rows).to_csv(paths["metrics"] / "balance_class_distribution.csv", index=False)


if __name__ == "__main__":
    main()
