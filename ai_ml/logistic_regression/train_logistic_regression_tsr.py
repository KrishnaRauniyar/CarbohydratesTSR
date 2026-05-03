#!/usr/bin/env python
"""Train a Logistic Regression classifier for TSR key-frequency feature tables."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

THIS_DIR = Path(__file__).resolve().parent
DNN_HELPER_DIR = THIS_DIR.parent / "deep_neural_network"
sys.path.insert(0, str(DNN_HELPER_DIR))

from utils import (  # noqa: E402
    apply_training_balance,
    coerce_features_to_numeric,
    compute_class_weights,
    ensure_output_dirs,
    evaluate_predictions,
    extract_labels_from_protein,
    filter_min_class_count,
    fit_preprocessor,
    initial_feature_columns,
    load_input_dataframe,
    make_preprocessing_report,
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
from visualization import (  # noqa: E402
    plot_balance_distribution,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_frequency_summary,
    plot_per_class_metric,
    plot_prediction_confidence,
    plot_split_distributions,
    plot_top_confused_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Logistic Regression for TSR features.")
    parser.add_argument("--input_csv", required=True, help="Input CSV with protein/Protein Name and TSR features.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    parser.add_argument("--label_from", default="Protein Name", help="Column whose final underscore token is the label.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Held-out test fraction.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--balance_method",
        default="class_weight",
        choices=["none", "class_weight", "smote", "smoteenn", "random_oversample"],
        help="Training-set-only imbalance handling.",
    )
    parser.add_argument("--use_log1p", action="store_true", help="Apply log1p to count/frequency features.")
    parser.add_argument(
        "--scaler",
        default="standard",
        choices=["standard", "robust", "none"],
        help="Feature scaler. Logistic Regression usually needs standard or robust scaling.",
    )
    parser.add_argument("--min_class_count", type=int, default=3, help="Drop classes below this count before splitting.")
    parser.add_argument("--smote_k_neighbors", type=int, default=5, help="SMOTE k_neighbors.")
    parser.add_argument("--rare_feature_min_total", type=float, default=0.0, help="Optional rare-feature threshold.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength.")
    parser.add_argument(
        "--penalty",
        default="l2",
        choices=["l1", "l2", "elasticnet", "none"],
        help="Regularization penalty.",
    )
    parser.add_argument(
        "--solver",
        default="saga",
        choices=["lbfgs", "saga", "liblinear", "newton-cg", "sag"],
        help="sklearn logistic regression solver.",
    )
    parser.add_argument("--l1_ratio", type=float, default=None, help="Elastic-net mixing value when penalty=elasticnet.")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum optimization iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Optimization tolerance.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="CPU workers where supported.")
    parser.add_argument(
        "--backend",
        default="sklearn",
        choices=["sklearn", "auto", "cuml"],
        help="Use sklearn, or try RAPIDS cuML when available.",
    )
    parser.add_argument("--top_n_plots", type=int, default=50, help="Top N classes/features in crowded plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)

    output_dir = Path(args.output_dir)
    paths = ensure_output_dirs(output_dir)
    config = vars(args).copy()
    save_json(config, paths["root"] / "config.json")
    save_json(config, paths["config"] / "config.json")

    df_original = load_input_dataframe(Path(args.input_csv), args.label_from)
    original_labels = extract_labels_from_protein(df_original[args.label_from])
    plot_class_distribution(
        original_labels,
        paths["plots"] / "class_distribution_before_splitting.png",
        "Class Distribution Before Splitting",
        top_n=args.top_n_plots,
    )

    df, duplicate_rows_removed = remove_duplicate_rows(df_original)
    labels = extract_labels_from_protein(df[args.label_from])
    df, labels, removed_classes = filter_min_class_count(df, labels, args.min_class_count)
    df = df.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    if labels.nunique() < 2:
        raise ValueError("At least two classes are required after --min_class_count filtering.")

    candidates = initial_feature_columns(df, args.label_from)
    X_numeric_raw, non_numeric_removed, all_missing_removed, missing_values_before_fill = coerce_features_to_numeric(
        df, candidates
    )
    if X_numeric_raw.shape[1] == 0:
        raise ValueError("No numeric TSR feature columns remain after validation.")
    plot_feature_frequency_summary(X_numeric_raw, paths["plots"], top_n=args.top_n_plots)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_.tolist()
    save_json({"class_names": class_names}, paths["config"] / "classes.json")

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

    report = make_preprocessing_report(
        input_rows=len(df_original),
        rows_after_duplicate_removal=len(df_original) - duplicate_rows_removed,
        rows_after_min_class_filter=len(df),
        input_columns=len(df_original.columns),
        initial_feature_count=len(candidates),
        non_numeric_removed=non_numeric_removed,
        all_missing_removed=all_missing_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        missing_values_before_fill=missing_values_before_fill,
        classes_removed_by_min_count=removed_classes,
        preprocessor_state=preprocessor_state,
    )
    save_json(report_to_dict(report), paths["metrics"] / "processed_dataset_summary.json")
    pd.Series(preprocessor_state["selected_features"], name="feature").to_csv(
        paths["metrics"] / "selected_features.csv", index=False
    )

    X_train_balanced, y_train_balanced, _, balance_report = apply_training_balance(
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

    model, backend_used = build_logistic_regression(args, y_train_balanced)
    config["backend_used"] = backend_used
    save_json(config, paths["root"] / "config.json")
    save_json(config, paths["config"] / "config.json")

    start = time.time()
    model.fit(X_train_balanced, y_train_balanced)
    training_seconds = time.time() - start
    save_json({"training_seconds": training_seconds, "backend_used": backend_used}, paths["metrics"] / "training_time.json")

    save_artifact(model, paths["models"] / "logistic_regression_model.joblib")
    save_artifact(preprocessor_state, paths["models"] / "preprocessor.joblib")
    save_artifact(label_encoder, paths["models"] / "label_encoder.joblib")
    save_coefficients(model, preprocessor_state["selected_features"], class_names, paths, args.top_n_plots)

    metrics_summary = []
    for split_name, X_split, y_split, proteins in [
        ("train", X_train, y_train, df.iloc[train_idx][args.label_from].tolist()),
        ("validation", X_val, y_val, df.iloc[val_idx][args.label_from].tolist()),
        ("test", X_test, y_test, df.iloc[test_idx][args.label_from].tolist()),
    ]:
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
        if split_name == "test":
            cm = cm_df.to_numpy()
            plot_confusion_matrix(cm, class_names, paths["plots"] / "confusion_matrix_test.png")
            plot_confusion_matrix(cm, class_names, paths["plots"] / "confusion_matrix_test_normalized.png", normalized=True)
            plot_per_class_metric(report_df, "precision", paths["plots"] / "per_class_precision_test.png", args.top_n_plots)
            plot_per_class_metric(report_df, "recall", paths["plots"] / "per_class_recall_test.png", args.top_n_plots)
            plot_per_class_metric(report_df, "f1-score", paths["plots"] / "per_class_f1_score_test.png", args.top_n_plots)
            plot_top_confused_pairs(cm, class_names, paths["plots"] / "top_confused_class_pairs_test.png")
            plot_prediction_confidence(pred_df, paths["plots"])

    pd.DataFrame(metrics_summary).to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
    print(f"Logistic Regression training complete. Outputs saved to: {output_dir.resolve()}")


def build_logistic_regression(args: argparse.Namespace, y_train: np.ndarray):
    class_weight = None
    if args.balance_method == "class_weight":
        class_weight = compute_class_weights(y_train)

    if args.backend in {"auto", "cuml"}:
        if class_weight is not None:
            message = "cuML LogisticRegression does not support sklearn-style class_weight reliably; using sklearn."
            if args.backend == "cuml":
                warnings.warn(message)
            else:
                warnings.warn(message)
        else:
            try:
                from cuml.linear_model import LogisticRegression as CuMLLogisticRegression

                model = CuMLLogisticRegression(
                    C=args.C,
                    penalty=args.penalty if args.penalty != "none" else "none",
                    max_iter=args.max_iter,
                    tol=args.tol,
                    fit_intercept=True,
                    verbose=0,
                )
                return model, "cuml"
            except Exception as exc:
                if args.backend == "cuml":
                    raise RuntimeError("Requested --backend cuml, but cuML could not be imported.") from exc
                warnings.warn(f"cuML is unavailable; falling back to sklearn. Details: {exc}")

    penalty = None if args.penalty == "none" else args.penalty
    if penalty == "elasticnet" and args.solver != "saga":
        raise ValueError("--penalty elasticnet requires --solver saga.")
    if penalty == "l1" and args.solver not in {"saga", "liblinear"}:
        raise ValueError("--penalty l1 requires --solver saga or liblinear.")

    model = LogisticRegression(
        C=args.C,
        penalty=penalty,
        solver=args.solver,
        l1_ratio=args.l1_ratio,
        class_weight=class_weight,
        max_iter=args.max_iter,
        tol=args.tol,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=1,
    )
    return model, "sklearn"


def evaluate_split(*, model, X, y, proteins, class_names, split_name: str, paths: Dict[str, Path]):
    y_prob = _predict_proba_numpy(model, X)
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
        pred_df.loc[~pred_df["is_correct"]].to_csv(paths["predictions"] / "misclassified_samples.csv", index=False)
        cm_df.to_csv(paths["metrics"] / "confusion_matrix.csv")
    return metrics, report_df, cm_df, pred_df


def _predict_proba_numpy(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    if hasattr(probabilities, "to_numpy"):
        probabilities = probabilities.to_numpy()
    return np.asarray(probabilities)


def save_coefficients(model, feature_names, class_names, paths: Dict[str, Path], top_n: int) -> None:
    if not hasattr(model, "coef_"):
        return
    import matplotlib.pyplot as plt

    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape[1] != len(feature_names):
        return

    coefficient_df = pd.DataFrame(coef, columns=feature_names)
    coefficient_df.insert(0, "class", class_names[: coef.shape[0]])
    coefficient_df.to_csv(paths["metrics"] / "class_coefficients.csv", index=False)

    importance = np.mean(np.abs(coef), axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_coefficient": importance})
    importance_df = importance_df.sort_values("mean_abs_coefficient", ascending=False)
    importance_df.to_csv(paths["metrics"] / "feature_importances.csv", index=False)

    plot_df = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(5, min(18, len(plot_df) * 0.35))))
    plt.barh(plot_df["feature"], plot_df["mean_abs_coefficient"], color="#4A5568")
    plt.title("Top Logistic Regression Coefficients")
    plt.xlabel("Mean absolute coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "top_feature_importances.png", dpi=300, bbox_inches="tight")
    plt.close()


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

