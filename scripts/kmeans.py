import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_data(input_file):
    """Load preprocessed data."""
    return pd.read_csv(input_file)


def apply_kmeans(data, n_clusters, random_state):
    """Apply K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans


def evaluate_clustering(data, labels):
    """Evaluates the clustering performance using multiple metrics."""
    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
    ch_index = calinski_harabasz_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    return silhouette, ch_index, db_index


def visualize_evaluation_results(results, output_plot_file):
    """Visualize clustering evaluation results for different values of k."""
    ks = list(results.keys())
    silhouette_scores = [results[k]["Silhouette Score"] for k in ks]
    ch_scores = [results[k]["Calinski-Harabasz Index"] for k in ks]
    db_scores = [results[k]["Davies-Bouldin Index"] for k in ks]

    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 5))

    # Silhouette Score
    plt.subplot(1, 3, 1)
    plt.plot(ks, silhouette_scores, marker='o', label="Silhouette")
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.legend()

    # Calinski-Harabasz Index
    plt.subplot(1, 3, 2)
    plt.plot(ks, ch_scores, marker='o', label="CH Index", color='g')
    plt.title("Calinski-Harabasz Index")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.legend()

    # Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(ks, db_scores, marker='o', label="DB Index", color='r')
    plt.title("Davies-Bouldin Index")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.legend()

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(output_plot_file, dpi=300)
    plt.close()


def save_metrics(metrics, metrics_file):
    """Saves clustering metrics to a JSON file."""
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)


def visualize_pca(data, labels, plot_file):
    """Visualizes the clustering results after PCA dimensionality reduction."""
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels, cmap='viridis', s=10)

    plt.colorbar(scatter, ax=ax, label="Cluster Labels")
    plt.title("K-Means Clustering (PCA Reduced to 3D)")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()


def apply_pca(data, n_components=3):
    """Apply PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca, pca


def main(input_file, output_file, plot_file, metrics_file, n_clusters, random_state, n_components=3):
    """Main function to perform clustering, evaluation, and visualization."""
    # Load the dataset
    data = load_data(input_file)
    features = data.drop(columns=['stroke'], errors='ignore')  # Exclude target column if present

    # Apply PCA to reduce dimensionality
    features_pca, pca = apply_pca(features, n_components=n_components)
    print(f"Explained Variance Ratio with {n_components} components: {sum(pca.explained_variance_ratio_):.4f}")

    # Tune n_clusters (k) by evaluating clustering metrics for a range of k values
    max_k = n_clusters
    evaluation_results = {}
    for k in range(2, max_k + 1):
        clusters, kmeans = apply_kmeans(features_pca, n_clusters=k, random_state=random_state)
        silhouette, ch_index, db_index = evaluate_clustering(features_pca, clusters)
        evaluation_results[k] = {
            "Silhouette Score": silhouette,
            "Calinski-Harabasz Index": ch_index,
            "Davies-Bouldin Index": db_index
        }
        print(f"k={k}: Silhouette={silhouette:.4f}, CH Index={ch_index:.4f}, DB Index={db_index:.4f}")

    # Save evaluation results to a JSON file
    save_metrics(evaluation_results, metrics_file)

    # Visualize the evaluation results and determine the best k
    evaluation_plot_file = plot_file.replace(".png", "_evaluation.png")
    visualize_evaluation_results(evaluation_results, evaluation_plot_file)

    # Select the best k (using Silhouette Score here, but can be adjusted)
    best_k = max(evaluation_results, key=lambda k: evaluation_results[k]["Silhouette Score"])
    print(f"Best k determined by Silhouette Score: {best_k}")

    # Apply K-Means clustering with the best k
    clusters, kmeans = apply_kmeans(features_pca, n_clusters=best_k, random_state=random_state)

    # Save the clustering results
    data['Cluster'] = clusters
    data.to_csv(output_file, index=False)

    # Visualize clustering results with PCA (already applied)
    visualize_pca(features_pca, clusters, plot_file)


if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    plot_file = snakemake.output[1]
    metrics_file = snakemake.output[2]
    RANDOM_SEED = snakemake.params.random_seed
    n_clusters = 10
    n_components = 3

    main(input_file, output_file, plot_file, metrics_file, n_clusters, RANDOM_SEED, n_components)

