#!/usr/bin/env python3
"""
Fast clustermap generation for TSR carbohydrate distance matrices.

This script keeps the original clustermap workflow:
1. read the generalized CSV produced by the TSR pipeline,
2. extract residue types from names like "..._..._<residue>",
3. cluster a distance matrix,
4. calculate Adjusted Rand Index,
5. save the clustermap image and clustered row names.

For small and medium matrices it can run exact hierarchical clustering. For
large matrices it switches to a representative plot plus full-row MiniBatchKMeans
assignments, because exact hierarchical clustering and full heatmap rendering are
O(n^2) memory/time operations and become impractical for very large n.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.spatial as sp
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster


@dataclass(frozen=True)
class ClusterResult:
    row_names: pd.Index
    row_order: np.ndarray
    cluster_labels: np.ndarray
    mode_used: str


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def estimate_exact_memory_gb(n_items: int, dtype_bytes: int) -> float:
    """Approximate peak memory for exact mode.

    The input square matrix is n*n. Hierarchical clustering also needs a
    condensed distance vector of n*(n-1)/2 doubles and a linkage matrix.
    Pandas/object overhead is not included, so real usage will be higher.
    """
    matrix = n_items * n_items * dtype_bytes
    condensed = n_items * (n_items - 1) // 2 * 8
    linkage = max(n_items - 1, 0) * 4 * 8
    return (matrix + condensed + linkage) / (1024**3)


def load_and_process_data(csv_file: str | Path, dtype: str = "float32") -> tuple[pd.DataFrame, pd.Series]:
    """Load the original generalized CSV format used by the old script."""
    csv_file = Path(csv_file)
    log(f"Reading input CSV: {csv_file}")

    df = pd.read_csv(csv_file, header=None, low_memory=False)
    split_first_column = df[0].astype(str).str.split(";", expand=True)
    item_names = split_first_column[0].astype(str)
    residue_split = item_names.str.rsplit("_", n=1, expand=True)

    if residue_split.shape[1] < 2 or residue_split[1].eq("").any():
        raise ValueError(
            "Could not extract residue type from column 0. Expected each name to end "
            "with an underscore residue token, like 'part1_part2_RESIDUE'."
        )

    residue_types = residue_split[1].reset_index(drop=True)

    numeric_part = df.drop(0, axis=1)
    merged_df = pd.concat([split_first_column, numeric_part], axis=1)

    processed_df = (
        merged_df.set_index(merged_df[0])
        .T.reset_index(drop=True)
        .rename_axis(None, axis=1)
    )
    processed_df = processed_df.drop(index=0).reset_index(drop=True)
    processed_df.index = processed_df.columns

    processed_df = processed_df.apply(pd.to_numeric, errors="coerce").astype(dtype, copy=False)
    if processed_df.isna().any().any():
        missing = int(processed_df.isna().sum().sum())
        raise ValueError(f"Input matrix contains {missing} non-numeric or missing values.")

    if processed_df.shape[0] != processed_df.shape[1]:
        raise ValueError(
            f"Expected a square distance matrix after processing, got {processed_df.shape}."
        )

    if len(residue_types) != processed_df.shape[0]:
        residue_types = residue_types.iloc[: processed_df.shape[0]].reset_index(drop=True)

    log(f"Processed DataFrame shape: {processed_df.shape}")
    log(f"Unique residue types: {sorted(residue_types.dropna().unique().tolist())}")
    return processed_df, residue_types


def compute_linkage_from_distance_matrix(
    matrix: np.ndarray,
    method: str,
    use_fastcluster: bool,
) -> np.ndarray:
    """Compute hierarchical linkage from a square distance matrix."""
    condensed = sp.distance.squareform(matrix, checks=False)

    if use_fastcluster:
        try:
            import fastcluster  # type: ignore

            log("Using fastcluster.linkage with preserve_input=False")
            return fastcluster.linkage(condensed, method=method, preserve_input=False)
        except ImportError:
            log("fastcluster is not installed; falling back to scipy.cluster.hierarchy.linkage")

    log("Using scipy.cluster.hierarchy.linkage")
    return hierarchy.linkage(condensed, method=method)


def make_palette(values: Iterable[object], palette_name: str) -> dict[object, tuple[float, float, float]]:
    unique_values = sorted(set(values))
    palette = sns.color_palette(palette_name, max(len(unique_values), 1))
    return {value: color for value, color in zip(unique_values, palette)}


def comb2(values: np.ndarray) -> np.ndarray:
    return values * (values - 1) / 2.0


def adjusted_rand_index(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    """Calculate Adjusted Rand Index without importing scikit-learn."""
    contingency = pd.crosstab(pd.Series(labels_true), pd.Series(labels_pred)).to_numpy(dtype=np.float64)
    n_samples = contingency.sum()
    if n_samples < 2:
        return 1.0

    sum_comb_contingency = comb2(contingency).sum()
    sum_comb_rows = comb2(contingency.sum(axis=1)).sum()
    sum_comb_cols = comb2(contingency.sum(axis=0)).sum()
    total_pairs = comb2(np.array([n_samples], dtype=np.float64))[0]

    expected_index = sum_comb_rows * sum_comb_cols / total_pairs
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0
    return float((sum_comb_contingency - expected_index) / denominator)


def save_order_csv(row_names: pd.Index, row_order: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Keep the same public CSV shape as the original script: one Drugs column.
    pd.DataFrame({"Drugs": row_names}).to_csv(output_csv, index=False)
    log(f"Saved clustered row names: {output_csv}")


def save_assignments_csv(
    row_names: pd.Index,
    residue_types: pd.Series,
    cluster_labels: np.ndarray,
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Drugs": row_names,
            "ResidueType": residue_types.to_numpy(),
            "ClusterLabel": cluster_labels,
        }
    ).to_csv(output_csv, index=False)
    log(f"Saved all-row cluster assignments: {output_csv}")


def plot_fast_heatmap(
    matrix: np.ndarray,
    row_order: np.ndarray,
    residue_types: pd.Series,
    cluster_labels: np.ndarray,
    row_names: pd.Index,
    output_file: Path,
    title: str,
    max_plot_items: int,
) -> None:
    """Render an ordered heatmap without seaborn's expensive dendrogram artist."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if len(row_order) > max_plot_items:
        log(f"Plot has {len(row_order)} rows; downsampling display to {max_plot_items} rows")
        take = np.linspace(0, len(row_order) - 1, num=max_plot_items, dtype=int)
        plot_order = row_order[take]
    else:
        plot_order = row_order

    ordered_matrix = matrix[np.ix_(plot_order, plot_order)]
    ordered_residues = residue_types.iloc[plot_order].to_numpy()
    ordered_clusters = cluster_labels[plot_order]

    residue_palette = make_palette(ordered_residues, "husl")
    cluster_palette = make_palette(ordered_clusters, "Set2")
    residue_rgb = np.array([residue_palette[value] for value in ordered_residues], dtype=float)
    cluster_rgb = np.array([cluster_palette[value] for value in ordered_clusters], dtype=float)

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[0.35, 10],
        height_ratios=[0.35, 10],
    )

    ax_residue = fig.add_subplot(grid[1, 0])
    ax_cluster = fig.add_subplot(grid[0, 1])
    ax_heatmap = fig.add_subplot(grid[1, 1])

    ax_residue.imshow(residue_rgb.reshape(len(plot_order), 1, 3), aspect="auto")
    ax_residue.set_xticks([])
    ax_residue.set_yticks([])
    ax_residue.set_ylabel("Residue")

    ax_cluster.imshow(cluster_rgb.reshape(1, len(plot_order), 3), aspect="auto")
    ax_cluster.set_xticks([])
    ax_cluster.set_yticks([])
    ax_cluster.set_title(title)

    image = ax_heatmap.imshow(ordered_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xlabel(f"{len(plot_order)} plotted rows")
    ax_heatmap.set_ylabel(f"{len(plot_order)} plotted rows")

    cbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.025, pad=0.01)
    cbar.set_label("Distance (%)")

    if len(plot_order) <= 80:
        labels = row_names[plot_order]
        ax_heatmap.set_xticks(np.arange(len(plot_order)))
        ax_heatmap.set_yticks(np.arange(len(plot_order)))
        ax_heatmap.set_xticklabels(labels, rotation=90, fontsize=6)
        ax_heatmap.set_yticklabels(labels, fontsize=6)

    fig.savefig(output_file, dpi=220)
    plt.close(fig)
    log(f"Saved heatmap: {output_file}")


def plot_seaborn_clustermap(
    matrix_df: pd.DataFrame,
    linkage: np.ndarray,
    residue_types: pd.Series,
    cluster_labels: np.ndarray,
    output_file: Path,
) -> np.ndarray:
    """Render the old-style seaborn clustermap for smaller matrices."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    residue_colors = make_palette(residue_types, "husl")
    cluster_colors = make_palette(cluster_labels, "Set2")
    row_colors_residue = [residue_colors[residue_types.iloc[i]] for i in range(len(residue_types))]
    row_colors_cluster = [cluster_colors[cluster_labels[i]] for i in range(len(cluster_labels))]

    g = sns.clustermap(
        matrix_df,
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage,
        col_linkage=linkage,
        row_colors=[row_colors_residue, row_colors_cluster],
        cbar_kws={"shrink": 0.5},
        cbar_pos=(0.1, 0.83, 0.02, 0.18),
        figsize=(10, 8),
    )
    g.savefig(output_file, bbox_inches="tight", dpi=220)
    row_order = np.asarray(g.dendrogram_row.reordered_ind, dtype=int)
    plt.close()
    log(f"Saved seaborn clustermap: {output_file}")
    return row_order


def run_exact(
    data: pd.DataFrame,
    residue_types: pd.Series,
    output_file: Path,
    output_csv: Path,
    assignments_csv: Path,
    num_clusters: int | None,
    method: str,
    use_fastcluster: bool,
    seaborn_max_items: int,
    max_plot_items: int,
) -> ClusterResult:
    n_items = data.shape[0]
    matrix = data.to_numpy(dtype=np.float32, copy=True) * 100.0
    np.fill_diagonal(matrix, 0.0)

    log(f"Exact mode estimated matrix+linkage memory: {estimate_exact_memory_gb(n_items, 4):.2f} GB")
    linkage = compute_linkage_from_distance_matrix(matrix, method=method, use_fastcluster=use_fastcluster)

    if num_clusters is None:
        num_clusters = len(set(residue_types))
    cluster_labels = fcluster(linkage, t=num_clusters, criterion="maxclust")
    ari = adjusted_rand_index(residue_types, cluster_labels)
    log(f"Adjusted Rand Index: {ari:.6f}")

    if n_items <= seaborn_max_items:
        row_order = plot_seaborn_clustermap(data.astype(float) * 100.0, linkage, residue_types, cluster_labels, output_file)
    else:
        row_order = hierarchy.leaves_list(linkage)
        plot_fast_heatmap(
            matrix,
            row_order,
            residue_types,
            cluster_labels,
            data.index,
            output_file,
            title="Exact hierarchical clustering",
            max_plot_items=max_plot_items,
        )

    row_names = data.index[row_order]
    save_order_csv(row_names, row_order, output_csv)
    save_assignments_csv(data.index, residue_types, cluster_labels, assignments_csv)
    return ClusterResult(row_names, row_order, cluster_labels, "exact")


def stratified_sample_indices(
    residue_types: pd.Series,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_items = len(residue_types)
    if sample_size >= n_items:
        return np.arange(n_items)

    sample_parts: list[np.ndarray] = []
    grouped = residue_types.reset_index().groupby(residue_types.to_numpy(), sort=False)
    for _, group in grouped:
        group_indices = group["index"].to_numpy(dtype=int)
        group_take = max(1, round(sample_size * len(group_indices) / n_items))
        group_take = min(group_take, len(group_indices))
        sample_parts.append(rng.choice(group_indices, size=group_take, replace=False))

    sample_indices = np.concatenate(sample_parts)
    if len(sample_indices) > sample_size:
        sample_indices = rng.choice(sample_indices, size=sample_size, replace=False)
    elif len(sample_indices) < sample_size:
        missing = sample_size - len(sample_indices)
        remaining = np.setdiff1d(np.arange(n_items), sample_indices, assume_unique=False)
        if len(remaining) > 0:
            sample_indices = np.concatenate(
                [sample_indices, rng.choice(remaining, size=min(missing, len(remaining)), replace=False)]
            )

    return np.sort(sample_indices.astype(int))


def random_feature_indices(n_features: int, feature_sample_size: int, random_state: int) -> np.ndarray:
    """Choose a reproducible subset of distance columns for scalable assignment."""
    if feature_sample_size <= 0 or feature_sample_size >= n_features:
        return np.arange(n_features)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_features, size=feature_sample_size, replace=False).astype(int))


def predict_nearest_center(matrix: np.ndarray, centers: np.ndarray, chunk_size: int) -> np.ndarray:
    """Assign each row to the nearest center without building a huge distance matrix."""
    labels = np.empty(matrix.shape[0], dtype=np.int32)
    center_norm = np.einsum("ij,ij->i", centers, centers)

    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        chunk = matrix[start:stop]
        chunk_norm = np.einsum("ij,ij->i", chunk, chunk)[:, None]
        distances = chunk_norm + center_norm[None, :] - 2.0 * chunk @ centers.T
        labels[start:stop] = np.argmin(distances, axis=1)

    return labels


def minibatch_kmeans_numpy(
    matrix: np.ndarray,
    n_clusters: int,
    batch_size: int,
    max_iter: int,
    random_state: int,
) -> np.ndarray:
    """Small NumPy MiniBatchKMeans replacement that avoids sklearn/threadpoolctl.

    This is used on HPC systems where scikit-learn's threadpool detection can
    fail against the site OpenBLAS library before clustering even starts.
    """
    rng = np.random.default_rng(random_state)
    n_rows = matrix.shape[0]
    if n_clusters <= 0:
        raise ValueError("--num_clusters must be positive")
    if n_clusters > n_rows:
        raise ValueError(f"Cannot create {n_clusters} clusters from only {n_rows} rows")

    init_indices = rng.choice(n_rows, size=n_clusters, replace=False)
    centers = matrix[init_indices].astype(np.float32, copy=True)
    counts = np.zeros(n_clusters, dtype=np.int64)
    batch_size = min(max(batch_size, n_clusters), n_rows)

    for iteration in range(max_iter):
        order = rng.permutation(n_rows)
        inertia = 0.0

        for start in range(0, n_rows, batch_size):
            batch_indices = order[start : start + batch_size]
            batch = matrix[batch_indices]
            batch_labels = predict_nearest_center(batch, centers, chunk_size=batch_size)

            for cluster_id in range(n_clusters):
                members = batch[batch_labels == cluster_id]
                if len(members) == 0:
                    continue
                old_count = counts[cluster_id]
                new_count = old_count + len(members)
                centers[cluster_id] = (
                    centers[cluster_id] * old_count + members.sum(axis=0)
                ) / new_count
                counts[cluster_id] = new_count

            nearest = centers[batch_labels]
            diff = batch - nearest
            inertia += float(np.einsum("ij,ij->", diff, diff))

        log(f"NumPy MiniBatchKMeans iteration {iteration + 1}/{max_iter}, inertia={inertia:.3f}")

    return predict_nearest_center(matrix, centers, chunk_size=batch_size) + 1


def assign_large_mode_clusters(
    matrix: np.ndarray,
    n_clusters: int,
    batch_size: int,
    random_state: int,
    backend: str,
    max_iter: int,
) -> np.ndarray:
    """Assign all rows to clusters using the selected large-mode backend."""
    if backend == "sklearn":
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=10,
            random_state=random_state,
            reassignment_ratio=0.01,
        )
        return kmeans.fit_predict(matrix) + 1

    return minibatch_kmeans_numpy(
        matrix=matrix,
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
    )


def run_large(
    data: pd.DataFrame,
    residue_types: pd.Series,
    output_file: Path,
    output_csv: Path,
    assignments_csv: Path,
    num_clusters: int | None,
    method: str,
    use_fastcluster: bool,
    sample_size: int,
    feature_sample_size: int,
    batch_size: int,
    assignment_backend: str,
    kmeans_max_iter: int,
    random_state: int,
    max_plot_items: int,
) -> ClusterResult:
    """Run scalable approximate clustering and a representative exact heatmap."""
    n_items = data.shape[0]
    if num_clusters is None:
        num_clusters = len(set(residue_types))

    matrix = data.to_numpy(dtype=np.float32, copy=True) * 100.0
    np.fill_diagonal(matrix, 0.0)

    log(
        "Large mode: mini-batch k-means creates full-row assignments; "
        "hierarchical clustering is run only on a representative sample for plotting."
    )
    feature_indices = random_feature_indices(matrix.shape[1], feature_sample_size, random_state)
    kmeans_matrix = matrix[:, feature_indices]
    log(f"Assignment backend: {assignment_backend}")
    log(f"Mini-batch k-means feature columns: {kmeans_matrix.shape[1]} of {matrix.shape[1]}")

    cluster_labels = assign_large_mode_clusters(
        matrix=kmeans_matrix,
        n_clusters=num_clusters,
        batch_size=batch_size,
        random_state=random_state,
        backend=assignment_backend,
        max_iter=kmeans_max_iter,
    )
    ari = adjusted_rand_index(residue_types, cluster_labels)
    log(f"Approximate all-row Adjusted Rand Index: {ari:.6f}")
    save_assignments_csv(data.index, residue_types, cluster_labels, assignments_csv)

    sample_indices = stratified_sample_indices(residue_types, sample_size, random_state)
    log(f"Representative clustermap sample size: {len(sample_indices)} of {n_items}")
    sample_matrix = matrix[np.ix_(sample_indices, sample_indices)]
    sample_linkage = compute_linkage_from_distance_matrix(
        sample_matrix,
        method=method,
        use_fastcluster=use_fastcluster,
    )
    sample_order_local = hierarchy.leaves_list(sample_linkage)
    row_order = sample_indices[sample_order_local]

    plot_fast_heatmap(
        matrix,
        row_order,
        residue_types,
        cluster_labels,
        data.index,
        output_file,
        title="Representative hierarchical clustermap",
        max_plot_items=max_plot_items,
    )

    row_names = data.index[row_order]
    save_order_csv(row_names, row_order, output_csv)
    return ClusterResult(row_names, row_order, cluster_labels, "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exact or scalable clustermap outputs from a TSR distance matrix. "
            "Use --mode exact for full hierarchical clustering; use --mode large for "
            "representative clustering when the matrix is too large."
        )
    )
    parser.add_argument("-p", "--input_path", required=True, help="Generalized CSV input path")
    parser.add_argument("--output_dir", default="outputs", help="Directory for images, logs, and CSV outputs")
    parser.add_argument("--plot_file", default="clustermap.png", help="Output image filename")
    parser.add_argument("--order_csv", default="clustermap.csv", help="Clustered row-name CSV filename")
    parser.add_argument("--assignments_csv", default="cluster_assignments.csv", help="All-row assignment CSV filename")
    parser.add_argument("--mode", choices=["auto", "exact", "large"], default="auto")
    parser.add_argument("--max_exact_items", type=int, default=5000, help="Auto mode uses exact at or below this n")
    parser.add_argument("--sample_size", type=int, default=2500, help="Representative sample size for large mode")
    parser.add_argument(
        "--feature_sample_size",
        type=int,
        default=4096,
        help="Distance columns used for MiniBatchKMeans in large mode; 0 means all columns",
    )
    parser.add_argument("--batch_size", type=int, default=4096, help="Mini-batch k-means batch size for large mode")
    parser.add_argument(
        "--assignment_backend",
        choices=["numpy", "sklearn"],
        default="numpy",
        help="Backend for large-mode assignments; numpy avoids sklearn/threadpoolctl HPC issues",
    )
    parser.add_argument("--kmeans_max_iter", type=int, default=20, help="NumPy mini-batch k-means iterations")
    parser.add_argument("--num_clusters", type=int, default=None, help="Cluster count; default is number of residue types")
    parser.add_argument("--method", default="average", help="Hierarchical linkage method")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--no_fastcluster", action="store_true", help="Disable fastcluster even if installed")
    parser.add_argument("--seaborn_max_items", type=int, default=1500, help="Use seaborn clustermap up to this n")
    parser.add_argument("--max_plot_items", type=int, default=3000, help="Maximum rows/cols drawn in heatmap image")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    log(f"Pandas version: {pd.__version__}")
    log(f"Seaborn version: {sns.__version__}")
    log(f"Matplotlib version: {matplotlib.__version__}")
    log(f"Scipy version: {scipy.__version__}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.plot_file
    order_csv = output_dir / args.order_csv
    assignments_csv = output_dir / args.assignments_csv

    data, residue_types = load_and_process_data(args.input_path, dtype=args.dtype)
    n_items = data.shape[0]

    if args.mode == "auto":
        mode = "exact" if n_items <= args.max_exact_items else "large"
    else:
        mode = args.mode

    log(f"Selected mode: {mode} (n={n_items})")
    if mode == "exact":
        result = run_exact(
            data=data,
            residue_types=residue_types,
            output_file=output_file,
            output_csv=order_csv,
            assignments_csv=assignments_csv,
            num_clusters=args.num_clusters,
            method=args.method,
            use_fastcluster=not args.no_fastcluster,
            seaborn_max_items=args.seaborn_max_items,
            max_plot_items=args.max_plot_items,
        )
    else:
        result = run_large(
            data=data,
            residue_types=residue_types,
            output_file=output_file,
            output_csv=order_csv,
            assignments_csv=assignments_csv,
            num_clusters=args.num_clusters,
            method=args.method,
            use_fastcluster=not args.no_fastcluster,
            sample_size=min(args.sample_size, n_items),
            feature_sample_size=args.feature_sample_size,
            batch_size=args.batch_size,
            assignment_backend=args.assignment_backend,
            kmeans_max_iter=args.kmeans_max_iter,
            random_state=args.random_state,
            max_plot_items=args.max_plot_items,
        )

    elapsed = time.time() - start_time
    log(f"Mode used: {result.mode_used}")
    log(f"Rows in clustermap order CSV: {len(result.row_names)}")
    log(f"Finished in {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
