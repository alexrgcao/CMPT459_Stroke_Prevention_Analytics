import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import adjusted_rand_score
from kneed import KneeLocator
import json
import os

def load_data(input_file):
    data = pd.read_csv(input_file)
    if input_file == "output/outlier_detection/isolation_forest_inliers_detection_results.csv":
        data = data.drop(columns=['Outlier'])
    return data

def preprocess_data(data):
    
    features = data.drop(columns=['stroke'])
    pca = PCA(n_components=10)
    features_pca = pca.fit_transform(features)
    
    return features_pca

def metrics_heatmap(data, min_samples_range, target):
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    eps_values = []
    distance_list = []
    ari_list = []
    
    for min_samples in min_samples_range:
        eps, distance = k_distance(data, min_samples)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        ari = adjusted_rand_score(target, labels)
        if len(set(labels)) > 1:
            silhouette = silhouette_score(data, labels)
            ch_index = calinski_harabasz_score(data, labels)
            db_index = davies_bouldin_score(data, labels)
        else:
            silhouette, ch_index, db_index = -1, -1, -1
        silhouette_scores.append((min_samples, silhouette))
        ch_scores.append((min_samples, ch_index))
        db_scores.append((min_samples, db_index))
        eps_values.append(eps)
        distance_list.append(distance)
        ari_list.append((min_samples, ari))

    return np.array(silhouette_scores), np.array(ch_scores), np.array(db_scores), np.array(eps_values), np.array(distance_list), np.array(ari_list)

def plot_heatmap(metrics, min_samples_range, plot_file):
    silhouette_scores, _, db_scores, _, _, ari = metrics
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(min_samples_range, silhouette_scores[:, 1], label="Silhouette Score", marker="o")
    plt.plot(min_samples_range, ari[:, 1], label="Adjusted Rand Index", marker="o")
    plt.plot(min_samples_range, db_scores[:, 1], label="Davies-Bouldin Index", marker="o")
    
    #plt.gca().invert_yaxis()
    
    plt.xlabel("min_samples")
    plt.ylabel("Metric Value")
    plt.title("Clustering Metrics vs min_samples")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

def dbscan_clustering(data, eps=1.18, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

def k_distance(data, min_samples):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    distances = np.sort(distances[:, -1])

    kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
    best_eps = distances[kneedle.knee]
    
    return best_eps, distances

def plot_k_distance_graph(distances, knee_point, plot_file):
    plt.figure(figsize=(8, 6))
    plt.plot(distances, label="k-distance")
    plt.axvline(x=knee_point, color='r', linestyle='--', label=f'Optimal eps')
    plt.xlabel("Points sorted by distance to k-th nearest neighbor")
    plt.ylabel("k-th Nearest Neighbor Distance")
    plt.title("k-Distance Graph")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_clustering(data, labels):
    
    if len(set(labels)) == 1:
        return None, None, None
    
    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
    ch_index = calinski_harabasz_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    return silhouette, ch_index, db_index

def visualize_pca(data, labels, plot_file):
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels, cmap='viridis', s=10)
    
    plt.colorbar(scatter, ax=ax, label="Cluster Labels")
    plt.title("DBSCAN Clustering (PCA Reduced to 3D)")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(metrics, metrics_file):
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def main(input_file, output_file, plot_file, metrics_file, k_distance_file, metrics_heatmap_file):
    data = load_data(input_file)
    data_pca = preprocess_data(data)

    min_samples_range = range(5, 15)
    metrics = metrics_heatmap(data_pca, min_samples_range, data['stroke'])
    plot_heatmap(metrics, min_samples_range, metrics_heatmap_file)

    db_index = metrics[2]
    best_idx = np.argmin(db_index[:, 1])
    best_min_samples = int(db_index[best_idx, 0])
    best_eps = metrics[3][best_idx]
    best_eps_distance = metrics[4][best_idx]
    plot_k_distance_graph(best_eps_distance, best_eps, k_distance_file)
    print(f"Best min_samples: {best_min_samples} (Davies-Bouldin Index: {db_index[best_idx, 1]:.4f})")

    labels = dbscan_clustering(data_pca, eps=best_eps, min_samples=best_min_samples)
    ari = adjusted_rand_score(data['stroke'], labels)
    print(f"Adjusted Rand Index (ARI): {ari}")

    silhouette, ch_index, db_index = evaluate_clustering(data_pca, labels)
    print(f'Silhouette: {silhouette}, Calinski-Harabasz Index: {ch_index}, Davies-Bouldin Index: {db_index}')

    metrics = {
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Index": ch_index,
        "Davies-Bouldin Index": db_index,
        "Adjusted Rand Index": ari,
        "Number of Clusters": len(np.unique(labels)) - (1 if -1 in labels else 0),
        "Noise Points": list(labels).count(-1),
        "Best eps": best_eps,
        "Best min_samples": best_min_samples,
    }
    
    save_metrics(metrics, metrics_file)
    
    visualize_pca(data_pca, labels, plot_file)
    
    data['Cluster'] = labels
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    plot_file = snakemake.output[1]
    metrics_file = snakemake.output[2]
    k_distance_file = snakemake.output[3]
    metrics_heatmap_file = snakemake.output[4]
    main(input_file, output_file, plot_file, metrics_file, k_distance_file, metrics_heatmap_file)
