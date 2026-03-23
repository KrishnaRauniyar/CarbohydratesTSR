import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.cluster import hierarchy
from sklearn.metrics import adjusted_rand_score
import argparse
import scipy.spatial as sp
from scipy.cluster.hierarchy import fcluster
import matplotlib.colors as mcolors

print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Scipy version:", scipy.__version__)

def load_and_process_data(csv_file):
    df = pd.read_csv(csv_file, header=None)
    df1 = df[0].str.split(';', expand=True)
    residue_split = df1[0].str.split('_', expand=True)
    residue_types = residue_split[3]
    df = df.drop(0, axis=1)
    merged_df = pd.concat([df1, df], axis=1)
    processed_df = merged_df.set_index(merged_df[0]).T.reset_index(drop=True).rename_axis(None, axis=1)
    processed_df = processed_df.drop(index=0).reset_index(drop=True)
    processed_df.index = processed_df.columns
    print("Processed DataFrame shape:", processed_df.shape)
    print("Residue Types:", residue_types.tolist())
    return processed_df, residue_types

def plot_clustermap(data, residue_types, output_file, num_clusters=None):
    distance_matrix = 100 * data.astype(float)  # in percentage

    # Compute linkage for clustering
    linkage = hierarchy.linkage(sp.distance.squareform(distance_matrix.values), method='average')

    # Extract cluster labels
    if num_clusters is None:
        num_clusters = len(set(residue_types))
    cluster_labels = fcluster(linkage, t=num_clusters, criterion='maxclust')
    print("Cluster Labels:", cluster_labels.tolist())

    # Create color mappings for true and predicted labels
    unique_residues = sorted(set(residue_types))
    unique_clusters = sorted(set(cluster_labels))
    palette_residues = sns.color_palette("husl", len(unique_residues))
    palette_clusters = sns.color_palette("Set2", len(unique_clusters))
    
    residue_colors = {
        residue: color for residue, color in zip(unique_residues, palette_residues)
    }
    cluster_colors = {
        cluster: color for cluster, color in zip(unique_clusters, palette_clusters)
    }
    
    row_colors_residue = [
        residue_colors[residue_types[i]] for i in range(len(residue_types))
    ]
    row_colors_cluster = [
        cluster_colors[cluster_labels[i]] for i in range(len(cluster_labels))
    ]

    # Calculate ARI
    ari = adjusted_rand_score(residue_types, cluster_labels)
    print(f"Adjusted Rand Index: {ari}")

    # Create clustermap with row colors, no title, no legends
    g = sns.clustermap(
        distance_matrix,
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage,
        col_linkage=linkage,
        row_colors=[row_colors_residue, row_colors_cluster],
        cbar_kws={"shrink": 0.5},
        cbar_pos=(0.1, 0.83, 0.02, 0.18),
        figsize=(10, 8)
    )

    # Save the plot
    g.savefig(output_file, bbox_inches='tight')
    plt.close()

    # Get row order and names
    row_order = g.dendrogram_row.reordered_ind
    row_names = data.index[row_order]
    print("Row Order after clustering:", row_order)
    print("Row Names after clustering:", row_names.tolist())

    return row_names, row_order, cluster_labels

def save_row_names(row_names, output_csv):
    row_names_df = pd.DataFrame({'Drugs': row_names})
    row_names_df.to_csv(output_csv, index=False)

def calculate_ari(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari

def main(csv_file, plot_file, output_csv):
    data, residue_types = load_and_process_data(csv_file)
    
    # Plot clustermap and get cluster labels
    row_names, row_order, cluster_labels = plot_clustermap(data, residue_types, plot_file)
    
    # Save row names
    save_row_names(row_names, output_csv)

    # Align residue types with clustered data
    true_labels = residue_types.iloc[row_order].values
    print("True Labels:", true_labels.tolist())
    print("Predicted Labels:", cluster_labels.tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot heatmap clusters and calculate ARI")
    parser.add_argument('-p', '--input_path', type=str, required=True, help="Generalized CSV input path")
    args = parser.parse_args()
    main(args.input_path, 'clustermap.png', 'clustermap.csv')