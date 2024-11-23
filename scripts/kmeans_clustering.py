import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_data(input_file):
    """Load preprocessed data."""
    return pd.read_csv(input_file)


def apply_kmeans(data, n_clusters):
    """Apply K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans


def evaluate_silhouette_score(data, clusters):
    """Evaluate clustering using Silhouette Score."""
    return silhouette_score(data, clusters)


def plot_pca(data, clusters, output_dir):
    """Visualize clusters using PCA."""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=clusters, palette='viridis', s=60)
    plt.title('PCA Visualization of Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(output_dir, "pca_kmeans.png"))
    plt.close()


def plot_tsne(data, clusters, output_dir):
    """Visualize clusters using t-SNE."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=clusters, palette='viridis', s=60)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(output_dir, "tsne_kmeans.png"))
    plt.close()


def save_cluster_labels(data, clusters, output_file):
    """Save the dataset with cluster labels to a CSV file."""
    data['Cluster'] = clusters
    data.to_csv(output_file, index=False)


def save_evaluation_metrics(metrics, output_dir):
    """Save clustering evaluation metrics to a file."""
    metrics_file = os.path.join(output_dir, "clustering_evaluation.txt")
    with open(metrics_file, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")


def plot_elbow_method(data, max_k, output_dir):
    """Plot the elbow method to determine optimal k."""
    distortions = []
    for k in range(2, max_k + 1):
        _, kmeans = apply_kmeans(data, n_clusters=k)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(output_dir, "elbow_method.png"))
    plt.close()


def summarize_evaluations(results, output_dir):
    """Summarize evaluations and save the best k based on Silhouette Score."""
    best_k = max(results, key=results.get)
    summary_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Best k based on Silhouette Score: {best_k}\n")
        for k, score in results.items():
            f.write(f"k={k}: Silhouette Score = {score:.4f}\n")
    return best_k, results.get(best_k)


def main(input_file, output_dir, max_k=10):
    """Main function to perform clustering, evaluation, and visualization."""
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    data = load_data(input_file)
    features = data.drop(columns=['stroke'], errors='ignore')  # Exclude target column if present

    # Determine "elbow"
    plot_elbow_method(features, max_k, output_dir)
    
    # Find best k value for max_k range
    results = {}
    for k in range(2, max_k + 1):
        clusters, _ = apply_kmeans(data, n_clusters=k)
        score = evaluate_silhouette_score(data, clusters)
        results[k] = score
    best_k, silhouette_score = summarize_evaluations(results, output_dir)
    print("Best k and Silhouette score:", best_k, silhouette_score)

    # Apply K-Means clustering on best k
    clusters, kmeans_model = apply_kmeans(features, best_k)

    # Save the cluster labels
    save_cluster_labels(data, clusters, os.path.join(output_dir, "kmeans_clusters.csv"))

    # Visualize clustering
    plot_pca(features, clusters, output_dir)
    plot_tsne(features, clusters, output_dir)


if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_dir = snakemake.output[0]
    main(input_file, output_dir, 10)
