import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import json
import os

def load_data(input_file):
    return pd.read_csv(input_file)

def preprocess_data(data):
    data = data.drop(columns=['ever_married', 'Residence_type', 'gender'], errors='ignore')
    
    #scaler = StandardScaler()
    #data_scaled = scaler.fit_transform(data)
    
    return data

def dbscan_clustering(data, eps=1.18, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples,)
    labels = dbscan.fit_predict(data)
    return labels

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

def main(input_file, output_file, plot_file, metrics_file):
    data = load_data(input_file)
    data_scaled = preprocess_data(data)
    
    labels = dbscan_clustering(data_scaled)
    #print("Number of clusters (excluding noise):", len(np.unique(labels)) - (1 if -1 in labels else 0))
    #print("Cluster labels:", np.unique(labels))
    
    silhouette, ch_index, db_index = evaluate_clustering(data_scaled, labels)
    print(f'Silhouette: {silhouette}, Calinski-Harabasz Index: {ch_index}, Davies-Bouldin Index: {db_index}')

    metrics = {
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Index": ch_index,
        "Davies-Bouldin Index": db_index,
        "Number of Clusters": len(np.unique(labels)) - (1 if -1 in labels else 0),
        "Noise Points": list(labels).count(-1)
    }
    
    save_metrics(metrics, metrics_file)
    
    visualize_pca(data_scaled, labels, plot_file)
    
    data['Cluster'] = labels
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    plot_file = snakemake.output[1]
    metrics_file = snakemake.output[2]
    main(input_file, output_file, plot_file, metrics_file)
