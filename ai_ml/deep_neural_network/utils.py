"""Utility functions for the TSR key-frequency DNN pipeline.

The helpers in this module deliberately keep preprocessing state explicit.
That makes it harder to leak information from validation/test sets into the
training transformation and easier to reload the exact same pipeline later.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


LABEL_LIKE_COLUMNS = {
    "label",
    "labels",
    "class",
    "classes",
    "target",
    "targets",
    "y",
    "residue_type",
    "residue",
    "ligand",
    "ligand_name",
    "carb_name",
    "carbohydrate",
}


@dataclass
class PreprocessingReport:
    """Human-readable preprocessing diagnostics saved with every run."""

    input_rows: int
    rows_after_duplicate_removal: int
    rows_after_min_class_filter: int
    input_columns: int
    initial_feature_columns: int
    non_numeric_removed: List[str]
    all_missing_removed: List[str]
    zero_variance_removed: List[str]
    rare_features_removed: List[str]
    selected_feature_count: int
    duplicate_rows_removed: int
    missing_values_before_fill: int
    classes_removed_by_min_count: Dict[str, int]
    warnings: List[str]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow when TensorFlow is available."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Deterministic kernels are not available for every TensorFlow op, but
        # enabling this flag improves reproducibility when the runtime supports it.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create and return the standard output directories."""

    paths = {
        "root": output_dir,
        "config": output_dir / "config",
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
        "models": output_dir / "models",
        "predictions": output_dir / "predictions",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(data: Mapping, path: Path) -> None:
    """Save JSON with a stable, readable layout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_input_dataframe(input_csv: Path, label_from: str) -> pd.DataFrame:
    """Load the CSV and validate that the protein identifier column exists."""

    df = pd.read_csv(input_csv)
    if label_from not in df.columns:
        available = ", ".join(map(str, df.columns[:20]))
        raise ValueError(
            f"Column '{label_from}' was not found in {input_csv}. "
            f"Available columns include: {available}"
        )
    return df


def extract_labels_from_protein(values: pd.Series) -> pd.Series:
    """Extract class labels from the final underscore-separated token."""

    labels = values.astype(str).str.strip().str.split("_").str[-1].str.strip()
    invalid = values.isna() | (labels == "")
    if invalid.any():
        bad_examples = values[invalid].head(5).tolist()
        raise ValueError(f"Could not extract labels from these protein values: {bad_examples}")
    return labels


def initial_feature_columns(df: pd.DataFrame, label_from: str) -> List[str]:
    """Choose candidate feature columns before numeric validation.

    The intended input format is protein,k1,k2,...,kn. If metadata columns are
    present, label-like names are excluded here and other nonnumeric columns are
    removed during numeric validation with a saved warning.
    """

    excluded = {label_from}
    excluded.update(col for col in df.columns if str(col).strip().lower() in LABEL_LIKE_COLUMNS)
    return [col for col in df.columns if col not in excluded]


def remove_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows while preserving the first occurrence."""

    duplicate_mask = df.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        df = df.loc[~duplicate_mask].copy()
    return df, duplicate_count


def filter_min_class_count(
    df: pd.DataFrame,
    labels: pd.Series,
    min_class_count: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """Remove classes below a user-defined sample threshold."""

    counts = labels.value_counts()
    too_small = counts[counts < min_class_count]
    removed = {str(label): int(count) for label, count in too_small.items()}
    if removed:
        keep_mask = labels.isin(counts[counts >= min_class_count].index)
        df = df.loc[keep_mask].copy()
        labels = labels.loc[keep_mask].copy()
    return df, labels, removed


def coerce_features_to_numeric(
    df: pd.DataFrame,
    candidate_features: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], List[str], int]:
    """Convert candidate feature columns to numeric and drop unusable columns."""

    numeric = df.loc[:, candidate_features].apply(pd.to_numeric, errors="coerce")
    missing_before_fill = int(numeric.isna().sum().sum())

    original_non_null = df.loc[:, candidate_features].notna().sum()
    numeric_non_null = numeric.notna().sum()
    non_numeric_removed = [
        col
        for col in candidate_features
        if original_non_null[col] > 0 and numeric_non_null[col] == 0
    ]
    all_missing_removed = [col for col in candidate_features if original_non_null[col] == 0]

    keep_columns = [
        col
        for col in candidate_features
        if col not in set(non_numeric_removed) and col not in set(all_missing_removed)
    ]
    return numeric.loc[:, keep_columns], non_numeric_removed, all_missing_removed, missing_before_fill


def stratified_train_val_test_split(
    df: pd.DataFrame,
    labels_encoded: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
    identifiers: Optional[Sequence[str]] = None,
    split_strategy: str = "row_stratified",
    group_split_candidates: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation, and test indexes using row or grouped splitting."""

    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("--test_size and --val_size must be positive and sum to less than 1.")

    if split_strategy not in {"row_stratified", "pdb_grouped", "pdb_chain_grouped"}:
        raise ValueError(
            "--split_strategy must be one of: row_stratified, pdb_grouped, pdb_chain_grouped"
        )

    if split_strategy != "row_stratified":
        if identifiers is None:
            raise ValueError("Grouped splitting requires sample identifiers.")
        groups = extract_groups_from_identifiers(identifiers, split_strategy)
        return grouped_train_val_test_split(
            labels_encoded=labels_encoded,
            groups=groups,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            n_candidates=group_split_candidates,
        )

    all_idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_encoded,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=labels_encoded[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def extract_groups_from_identifiers(
    identifiers: Sequence[str],
    split_strategy: str,
) -> np.ndarray:
    """Extract PDB or PDB-chain groups from underscore-delimited identifiers."""

    required_tokens = 1 if split_strategy == "pdb_grouped" else 2
    groups = []
    invalid = []
    for value in identifiers:
        identifier = str(value).strip()
        tokens = identifier.split("_")
        if not identifier or len(tokens) < required_tokens or any(
            not token.strip() for token in tokens[:required_tokens]
        ):
            invalid.append(identifier)
            continue
        groups.append("_".join(token.strip() for token in tokens[:required_tokens]))
    if invalid:
        raise ValueError(
            f"Could not extract {split_strategy} groups from identifiers such as: {invalid[:5]}"
        )
    return np.asarray(groups, dtype=object)


def grouped_train_val_test_split(
    labels_encoded: np.ndarray,
    groups: Sequence[str],
    test_size: float,
    val_size: float,
    random_state: int,
    n_candidates: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose disjoint group partitions that approximate class-stratified fractions."""

    labels_encoded = np.asarray(labels_encoded)
    groups = np.asarray(groups, dtype=object)
    if len(labels_encoded) != len(groups):
        raise ValueError("labels_encoded and groups must have the same length.")
    if len(np.unique(groups)) < 3:
        raise ValueError("Grouped train/validation/test splitting requires at least three groups.")
    if n_candidates < 1:
        raise ValueError("--group_split_candidates must be at least 1.")

    all_idx = np.arange(len(labels_encoded))
    train_val_idx, test_idx = _best_group_holdout(
        indices=all_idx,
        labels=labels_encoded,
        groups=groups,
        holdout_fraction=test_size,
        random_state=random_state,
        n_candidates=n_candidates,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = _best_group_holdout(
        indices=train_val_idx,
        labels=labels_encoded,
        groups=groups,
        holdout_fraction=relative_val_size,
        random_state=random_state + 1,
        n_candidates=n_candidates,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _best_group_holdout(
    *,
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    holdout_fraction: float,
    random_state: int,
    n_candidates: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select the best of deterministic random group holdouts."""

    subset_labels = labels[indices]
    subset_groups = groups[indices]
    classes, total_class_counts = np.unique(subset_labels, return_counts=True)
    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=holdout_fraction,
        random_state=random_state,
    )
    best = None
    for train_relative, holdout_relative in splitter.split(
        np.zeros(len(indices)),
        subset_labels,
        subset_groups,
    ):
        train_counts = np.bincount(
            subset_labels[train_relative],
            minlength=int(classes.max()) + 1,
        )[classes]
        if np.any(train_counts == 0):
            continue
        holdout_counts = np.bincount(
            subset_labels[holdout_relative],
            minlength=int(classes.max()) + 1,
        )[classes]
        observed_fraction = len(holdout_relative) / len(indices)
        size_error = abs(observed_fraction - holdout_fraction)
        class_fraction_error = float(
            np.mean(np.abs(holdout_counts / total_class_counts - holdout_fraction))
        )
        missing_fraction = float(np.mean(holdout_counts == 0))
        score = 4.0 * size_error + class_fraction_error + 0.25 * missing_fraction
        candidate = (score, size_error, class_fraction_error, train_relative, holdout_relative)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    if best is None:
        raise ValueError(
            "Could not create a grouped holdout while retaining every class in training. "
            "Some classes may occur in only one group; increase --min_class_count, merge rare "
            "classes, or choose a broader training dataset."
        )
    return indices[best[3]], indices[best[4]]


def save_split_assignments(
    *,
    identifiers: Sequence[str],
    labels: Sequence[str],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    split_strategy: str,
    assignments_path: Path,
    report_path: Path,
) -> Dict:
    """Save row-level split assignments and grouped-split diagnostics."""

    identifiers = pd.Series(identifiers, dtype=str).reset_index(drop=True)
    labels = pd.Series(labels, dtype=str).reset_index(drop=True)
    split = np.empty(len(identifiers), dtype=object)
    split[np.asarray(train_idx)] = "train"
    split[np.asarray(val_idx)] = "validation"
    split[np.asarray(test_idx)] = "test"

    if split_strategy == "row_stratified":
        groups = np.asarray([f"row_{index}" for index in range(len(identifiers))], dtype=object)
        group_scope = "row"
    else:
        groups = extract_groups_from_identifiers(identifiers, split_strategy)
        group_scope = "PDB" if split_strategy == "pdb_grouped" else "PDB-chain"

    assignments = pd.DataFrame(
        {
            "row_index": np.arange(len(identifiers)),
            "identifier": identifiers,
            "label": labels,
            "group": groups,
            "split": split,
        }
    )
    assignments.to_csv(assignments_path, index=False)

    split_names = ["train", "validation", "test"]
    group_sets = {
        name: set(assignments.loc[assignments["split"] == name, "group"])
        for name in split_names
    }
    label_sets = {
        name: set(assignments.loc[assignments["split"] == name, "label"])
        for name in split_names
    }
    all_labels = set(labels)
    class_group_counts = assignments.groupby("label")["group"].nunique()
    report = {
        "split_strategy": split_strategy,
        "group_scope": group_scope,
        "total_rows": int(len(assignments)),
        "total_groups": int(assignments["group"].nunique()),
        "rows_by_split": {
            name: int((assignments["split"] == name).sum()) for name in split_names
        },
        "groups_by_split": {name: int(len(group_sets[name])) for name in split_names},
        "group_overlap": {
            "train_validation": int(len(group_sets["train"] & group_sets["validation"])),
            "train_test": int(len(group_sets["train"] & group_sets["test"])),
            "validation_test": int(len(group_sets["validation"] & group_sets["test"])),
        },
        "classes_by_split": {name: int(len(label_sets[name])) for name in split_names},
        "classes_missing_by_split": {
            name: sorted(all_labels - label_sets[name]) for name in split_names
        },
        "classes_with_fewer_than_three_groups": sorted(
            class_group_counts[class_group_counts < 3].index.astype(str).tolist()
        ),
    }
    save_json(report, report_path)

    if split_strategy != "row_stratified" and any(report["group_overlap"].values()):
        raise RuntimeError("Grouped split validation failed because a group crosses partitions.")
    missing_test = report["classes_missing_by_split"]["test"]
    if missing_test:
        warnings.warn(
            f"{len(missing_test)} classes are absent from the test partition. "
            "See split_group_report.json for the class list."
        )
    return report


def save_split_statistics(
    labels_by_split: Mapping[str, Sequence[str]],
    output_path: Path,
) -> pd.DataFrame:
    """Save per-class counts and percentages for every split."""

    rows = []
    for split_name, labels in labels_by_split.items():
        series = pd.Series(labels, name="label")
        counts = series.value_counts().sort_index()
        total = len(series)
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "percent": float(count / total * 100.0) if total else 0.0,
                }
            )
    stats = pd.DataFrame(rows)
    stats.to_csv(output_path, index=False)
    return stats


def fit_preprocessor(
    X_train_raw: pd.DataFrame,
    use_log1p: bool,
    scaler_name: str,
    rare_feature_min_total: float = 0.0,
) -> Tuple[Dict, pd.DataFrame]:
    """Fit preprocessing state on training data only and transform train data."""

    warnings_list: List[str] = []
    train = X_train_raw.copy()

    medians = train.median(axis=0, skipna=True).fillna(0.0)
    train = train.fillna(medians)

    if use_log1p:
        min_value = float(train.min().min()) if train.shape[1] else 0.0
        if min_value < 0:
            warnings_list.append(
                "Negative feature values were clipped to 0 before log1p transformation."
            )
            train = train.clip(lower=0)
        train = np.log1p(train)

    zero_variance = train.columns[train.var(axis=0) <= 0].tolist()
    train = train.drop(columns=zero_variance)

    rare_features: List[str] = []
    if rare_feature_min_total and rare_feature_min_total > 0:
        totals = train.sum(axis=0)
        rare_features = totals[totals < rare_feature_min_total].index.tolist()
        train = train.drop(columns=rare_features)

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "none":
        scaler = None
    else:
        raise ValueError("--scaler must be one of: standard, robust, none")

    selected_features = train.columns.tolist()
    if not selected_features:
        raise ValueError("No usable feature columns remain after preprocessing.")

    if scaler is not None:
        scaled = scaler.fit_transform(train)
        train = pd.DataFrame(scaled, index=train.index, columns=selected_features)

    state = {
        "medians": medians.to_dict(),
        "use_log1p": bool(use_log1p),
        "scaler_name": scaler_name,
        "scaler": scaler,
        "selected_features": selected_features,
        "zero_variance_removed": zero_variance,
        "rare_features_removed": rare_features,
        "warnings": warnings_list,
    }
    return state, train


def transform_with_preprocessor(X_raw: pd.DataFrame, state: Mapping) -> pd.DataFrame:
    """Apply saved preprocessing state to validation, test, or new data."""

    medians = pd.Series(state["medians"], dtype=float)
    selected_features = list(state["selected_features"])

    # Reindex first so missing training columns in a future inference CSV become
    # NaN and are filled with the training median. Extra columns are ignored.
    X = X_raw.reindex(columns=medians.index).copy()
    X = X.fillna(medians).fillna(0.0)

    if state["use_log1p"]:
        X = X.clip(lower=0)
        X = np.log1p(X)

    X = X.reindex(columns=selected_features)
    scaler = state.get("scaler")
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), index=X.index, columns=selected_features)
    return X


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Compute sklearn balanced class weights for Keras."""

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def apply_training_balance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: str,
    random_state: int,
    smote_k_neighbors: int,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[Dict[int, float]], Dict]:
    """Apply the requested imbalance strategy to the training set only."""

    method = method.lower()
    before = pd.Series(y_train).value_counts().sort_index().to_dict()
    class_weights = None
    notes: List[str] = []

    if method == "none":
        X_balanced, y_balanced = X_train, y_train
    elif method == "class_weight":
        X_balanced, y_balanced = X_train, y_train
        class_weights = compute_class_weights(y_train)
    elif method in {"smote", "smoteenn", "random_oversample"}:
        try:
            from imblearn.combine import SMOTEENN
            from imblearn.over_sampling import RandomOverSampler, SMOTE
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for SMOTE, SMOTEENN, and random_oversample. "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        if method == "random_oversample":
            sampler = RandomOverSampler(random_state=random_state)
        else:
            min_count = int(pd.Series(y_train).value_counts().min())
            effective_k = min(smote_k_neighbors, min_count - 1)
            if effective_k < 1:
                raise ValueError(
                    "SMOTE requires at least two samples in every training class. "
                    "Increase --min_class_count or use --balance_method class_weight."
                )
            if effective_k != smote_k_neighbors:
                notes.append(
                    f"SMOTE k_neighbors was reduced from {smote_k_neighbors} to {effective_k} "
                    "because at least one class is small."
                )
                warnings.warn(notes[-1])
            smote = SMOTE(random_state=random_state, k_neighbors=effective_k)
            sampler = (
                SMOTEENN(random_state=random_state, smote=smote)
                if method == "smoteenn"
                else smote
            )
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        X_balanced = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_balanced = np.asarray(y_resampled)
    else:
        raise ValueError(
            "--balance_method must be one of: none, class_weight, smote, smoteenn, random_oversample"
        )

    after = pd.Series(y_balanced).value_counts().sort_index().to_dict()
    balance_report = {
        "method": method,
        "before": {int(k): int(v) for k, v in before.items()},
        "after": {int(k): int(v) for k, v in after.items()},
        "notes": notes,
    }
    return X_balanced, y_balanced, class_weights, balance_report


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    split_name: str,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Compute summary metrics, classification report, and confusion matrix."""

    y_pred = np.argmax(y_prob, axis=1)
    labels = np.arange(len(class_names))

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }
    metrics.update(_safe_multiclass_auc(y_true, y_prob, labels))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    return metrics, report_df, cm_df


def _safe_multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC and PR-AUC when the split contains enough classes."""

    results = {
        "roc_auc_macro_ovr": np.nan,
        "roc_auc_weighted_ovr": np.nan,
        "pr_auc_macro": np.nan,
        "pr_auc_weighted": np.nan,
    }
    if len(np.unique(y_true)) < 2:
        return results

    try:
        y_bin = LabelBinarizer().fit(labels).transform(y_true)
        if y_bin.shape[1] == 1:
            y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])
        results["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        )
        results["roc_auc_weighted_ovr"] = float(
            roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
        )
        results["pr_auc_macro"] = float(average_precision_score(y_bin, y_prob, average="macro"))
        results["pr_auc_weighted"] = float(
            average_precision_score(y_bin, y_prob, average="weighted")
        )
    except Exception as exc:
        warnings.warn(f"Could not compute multiclass AUC metrics: {exc}")
    return results


def save_predictions(
    proteins: Sequence[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> pd.DataFrame:
    """Save per-sample predictions with one probability column per class."""

    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    pred_df = pd.DataFrame(
        {
            "protein": proteins,
            "true_label": [class_names[i] for i in y_true],
            "predicted_label": [class_names[i] for i in y_pred],
            "predicted_probability": confidence,
            "is_correct": y_true == y_pred,
        }
    )
    for idx, class_name in enumerate(class_names):
        safe_name = str(class_name).replace(" ", "_")
        pred_df[f"probability_{safe_name}"] = y_prob[:, idx]
    pred_df.to_csv(output_path, index=False)
    return pred_df


def make_preprocessing_report(
    *,
    input_rows: int,
    rows_after_duplicate_removal: int,
    rows_after_min_class_filter: int,
    input_columns: int,
    initial_feature_count: int,
    non_numeric_removed: List[str],
    all_missing_removed: List[str],
    duplicate_rows_removed: int,
    missing_values_before_fill: int,
    classes_removed_by_min_count: Dict[str, int],
    preprocessor_state: Mapping,
) -> PreprocessingReport:
    """Build the report object saved as JSON."""

    warnings_list = list(preprocessor_state.get("warnings", []))
    if non_numeric_removed:
        warnings_list.append(
            f"Removed {len(non_numeric_removed)} nonnumeric candidate feature columns."
        )
    if classes_removed_by_min_count:
        warnings_list.append(
            f"Removed {len(classes_removed_by_min_count)} classes below --min_class_count."
        )

    return PreprocessingReport(
        input_rows=input_rows,
        rows_after_duplicate_removal=rows_after_duplicate_removal,
        rows_after_min_class_filter=rows_after_min_class_filter,
        input_columns=input_columns,
        initial_feature_columns=initial_feature_count,
        non_numeric_removed=non_numeric_removed,
        all_missing_removed=all_missing_removed,
        zero_variance_removed=list(preprocessor_state.get("zero_variance_removed", [])),
        rare_features_removed=list(preprocessor_state.get("rare_features_removed", [])),
        selected_feature_count=len(preprocessor_state.get("selected_features", [])),
        duplicate_rows_removed=duplicate_rows_removed,
        missing_values_before_fill=missing_values_before_fill,
        classes_removed_by_min_count=classes_removed_by_min_count,
        warnings=warnings_list,
    )


def save_artifact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path):
    return joblib.load(path)


def report_to_dict(report: PreprocessingReport) -> Dict:
    return asdict(report)


def print_device_information() -> Dict:
    """Print and return TensorFlow device information."""

    info = {"tensorflow_available": False, "gpus": [], "cpus": []}
    try:
        import tensorflow as tf

        info["tensorflow_available"] = True
        info["gpus"] = [device.name for device in tf.config.list_physical_devices("GPU")]
        info["cpus"] = [device.name for device in tf.config.list_physical_devices("CPU")]
        print("TensorFlow version:", tf.__version__)
        print("GPUs:", info["gpus"] if info["gpus"] else "none detected")
        print("CPUs:", info["cpus"] if info["cpus"] else "none detected")
    except Exception as exc:
        print(f"TensorFlow device information is unavailable: {exc}")
    return info
