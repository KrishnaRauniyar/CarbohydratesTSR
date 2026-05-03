"""Publication-oriented plotting helpers for TSR DNN experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
except Exception:
    sns = None


DEFAULT_DPI = 300


def _finish_plot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()


def plot_class_distribution(labels: Sequence[str], output_path: Path, title: str, top_n: int = 50) -> None:
    counts = pd.Series(labels).value_counts()
    _plot_counts(counts, output_path, title, "Class label", "Samples", top_n=top_n)


def plot_split_distributions(labels_by_split: Mapping[str, Sequence[str]], output_path: Path, top_n: int = 50) -> None:
    rows = []
    all_counts = pd.concat([pd.Series(labels) for labels in labels_by_split.values()]).value_counts()
    selected_labels = all_counts.head(top_n).index.tolist()
    for split, labels in labels_by_split.items():
        counts = pd.Series(labels).value_counts()
        for label in selected_labels:
            rows.append({"split": split, "label": label, "count": int(counts.get(label, 0))})
    data = pd.DataFrame(rows)

    plt.figure(figsize=(max(10, min(22, len(selected_labels) * 0.45)), 6))
    if sns is not None:
        sns.barplot(data=data, x="label", y="count", hue="split")
    else:
        pivot = data.pivot(index="label", columns="split", values="count").fillna(0)
        pivot.plot(kind="bar", ax=plt.gca())
    plt.title("Class Distribution by Split")
    plt.xlabel("Class label")
    plt.ylabel("Samples")
    plt.xticks(rotation=60, ha="right")
    _finish_plot(output_path)


def plot_balance_distribution(
    before_counts: Mapping,
    after_counts: Mapping,
    class_names: Sequence[str],
    output_path: Path,
    top_n: int = 50,
) -> None:
    rows = []
    before_named = {_class_name(k, class_names): v for k, v in before_counts.items()}
    after_named = {_class_name(k, class_names): v for k, v in after_counts.items()}
    selected = pd.Series(before_named).sort_values(ascending=False).head(top_n).index.tolist()
    for label in selected:
        rows.append({"stage": "before", "label": label, "count": int(before_named.get(label, 0))})
        rows.append({"stage": "after", "label": label, "count": int(after_named.get(label, 0))})
    data = pd.DataFrame(rows)

    plt.figure(figsize=(max(10, min(22, len(selected) * 0.45)), 6))
    if sns is not None:
        sns.barplot(data=data, x="label", y="count", hue="stage")
    else:
        data.pivot(index="label", columns="stage", values="count").plot(kind="bar", ax=plt.gca())
    plt.title("Training Class Distribution Before and After Balancing")
    plt.xlabel("Class label")
    plt.ylabel("Samples")
    plt.xticks(rotation=60, ha="right")
    _finish_plot(output_path)


def plot_training_history(history_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_history_pair(history_df, "loss", "val_loss", "Loss", output_dir / "training_validation_loss.png")
    _plot_history_pair(
        history_df,
        "accuracy",
        "val_accuracy",
        "Accuracy",
        output_dir / "training_validation_accuracy.png",
    )
    if "val_macro_f1" in history_df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="Validation macro F1", linewidth=2)
        plt.title("Validation Macro F1 by Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.ylim(0, 1)
        plt.legend()
        _finish_plot(output_dir / "validation_macro_f1.png")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    normalized: bool = False,
    max_classes_for_annotations: int = 35,
) -> None:
    matrix = cm.astype(float)
    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    n_classes = len(class_names)
    fig_size = max(8, min(24, n_classes * 0.45))
    plt.figure(figsize=(fig_size, fig_size))
    annotate = n_classes <= max_classes_for_annotations
    fmt = ".2f" if normalized else "d"
    data_for_plot = matrix if normalized else matrix.astype(int)
    if sns is not None:
        sns.heatmap(
            data_for_plot,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            annot=annotate,
            fmt=fmt,
            cbar=True,
            square=True,
        )
    else:
        plt.imshow(data_for_plot, cmap="Blues")
        plt.colorbar()
        plt.xticks(np.arange(n_classes), class_names, rotation=90)
        plt.yticks(np.arange(n_classes), class_names)
    plt.title("Normalized Confusion Matrix" if normalized else "Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    _finish_plot(output_path)


def plot_per_class_metric(report_df: pd.DataFrame, metric: str, output_path: Path, top_n: int = 50) -> None:
    class_rows = report_df[~report_df["label"].isin(["accuracy", "macro avg", "weighted avg"])].copy()
    if metric not in class_rows.columns:
        return
    class_rows = class_rows.sort_values(metric, ascending=True).tail(top_n)

    plt.figure(figsize=(10, max(5, min(18, len(class_rows) * 0.35))))
    plt.barh(class_rows["label"], class_rows[metric], color="#3C78D8")
    plt.title(f"Per-Class {metric.replace('-', ' ').title()}")
    plt.xlabel(metric.replace("-", " ").title())
    plt.ylabel("Class label")
    plt.xlim(0, 1)
    _finish_plot(output_path)


def plot_top_confused_pairs(cm: np.ndarray, class_names: Sequence[str], output_path: Path, top_n: int = 25) -> pd.DataFrame:
    rows = []
    for true_idx, true_label in enumerate(class_names):
        for pred_idx, pred_label in enumerate(class_names):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                rows.append({"true_label": true_label, "predicted_label": pred_label, "count": count})
    confused = pd.DataFrame(rows).sort_values("count", ascending=False) if rows else pd.DataFrame(rows)
    confused.head(top_n).to_csv(output_path.with_suffix(".csv"), index=False)
    if confused.empty:
        return confused

    plot_df = confused.head(top_n).copy()
    plot_df["pair"] = plot_df["true_label"].astype(str) + " -> " + plot_df["predicted_label"].astype(str)
    plt.figure(figsize=(10, max(5, min(14, len(plot_df) * 0.35))))
    plt.barh(plot_df["pair"][::-1], plot_df["count"][::-1], color="#A23B72")
    plt.title("Top Confused Class Pairs")
    plt.xlabel("Misclassified samples")
    plt.ylabel("True -> predicted")
    _finish_plot(output_path)
    return confused


def plot_prediction_confidence(pred_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.histplot(pred_df["predicted_probability"], bins=30, kde=True, color="#2F855A")
    else:
        plt.hist(pred_df["predicted_probability"], bins=30, color="#2F855A", alpha=0.8)
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Predicted class probability")
    plt.ylabel("Samples")
    _finish_plot(output_dir / "prediction_confidence_distribution.png")

    plt.figure(figsize=(7, 5))
    if sns is not None:
        sns.boxplot(data=pred_df, x="is_correct", y="predicted_probability")
    else:
        pred_df.boxplot(column="predicted_probability", by="is_correct", ax=plt.gca())
        plt.suptitle("")
    plt.title("Confidence for Correct vs Incorrect Predictions")
    plt.xlabel("Prediction is correct")
    plt.ylabel("Predicted class probability")
    _finish_plot(output_dir / "correct_vs_incorrect_confidence.png")


def plot_feature_frequency_summary(X_raw_numeric: pd.DataFrame, output_dir: Path, top_n: int = 50) -> None:
    totals = X_raw_numeric.sum(axis=0, skipna=True).sort_values(ascending=False)
    nonzero_counts = (X_raw_numeric.fillna(0) != 0).sum(axis=0).sort_values(ascending=False)
    summary = pd.DataFrame({"feature": totals.index, "total_frequency": totals.values})
    summary["nonzero_sample_count"] = summary["feature"].map(nonzero_counts)
    summary.to_csv(output_dir.parent / "metrics" / "feature_frequency_summary.csv", index=False)

    _plot_counts(totals, output_dir / "top_features_by_total_frequency.png", "Top Features by Total Frequency", "Feature", "Total frequency", top_n)

    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.histplot(totals.values, bins=50, color="#DD6B20")
    else:
        plt.hist(totals.values, bins=50, color="#DD6B20", alpha=0.8)
    plt.title("Feature Total Frequency Summary")
    plt.xlabel("Total frequency across samples")
    plt.ylabel("Features")
    _finish_plot(output_dir / "feature_frequency_summary.png")


def plot_pca_samples(
    X: pd.DataFrame,
    labels: Sequence[str],
    output_path: Path,
    title: str,
    predicted_labels: Optional[Sequence[str]] = None,
) -> None:
    if X.shape[0] < 3 or X.shape[1] < 2:
        return
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "label": labels})
    if predicted_labels is not None:
        plot_df["predicted_label"] = predicted_labels

    plt.figure(figsize=(9, 7))
    if sns is not None:
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="label", s=35, linewidth=0, alpha=0.85)
        if plot_df["label"].nunique() > 15:
            plt.legend([], [], frameon=False)
        else:
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    else:
        plt.scatter(plot_df["PC1"], plot_df["PC2"], s=35, alpha=0.85)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    _finish_plot(output_path)


def _plot_history_pair(history_df: pd.DataFrame, train_col: str, val_col: str, ylabel: str, output_path: Path) -> None:
    if train_col not in history_df.columns or val_col not in history_df.columns:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df[train_col], label=f"Training {ylabel.lower()}", linewidth=2)
    plt.plot(history_df["epoch"], history_df[val_col], label=f"Validation {ylabel.lower()}", linewidth=2)
    plt.title(f"Training vs Validation {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    _finish_plot(output_path)


def _plot_counts(counts: pd.Series, output_path: Path, title: str, xlabel: str, ylabel: str, top_n: int = 50) -> None:
    plot_counts_series = counts.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(max(10, min(22, len(plot_counts_series) * 0.45)), 6))
    plt.bar(plot_counts_series.index.astype(str), plot_counts_series.values, color="#3C78D8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=60, ha="right")
    _finish_plot(output_path)


def _class_name(encoded_label, class_names: Sequence[str]) -> str:
    try:
        return str(class_names[int(encoded_label)])
    except Exception:
        return str(encoded_label)

