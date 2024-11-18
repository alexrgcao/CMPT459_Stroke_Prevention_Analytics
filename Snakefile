#rule all:
#    input:
#        "data/processed/healthcare-dataset-stroke-data-processed.csv"

# Preprocess Rule
rule preprocess:
    input:
        "data/raw/healthcare-dataset-stroke-data.csv"
    output:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"
    script:
        "scripts/preprocess.py"

# EDA Rule
rule eda:
    input:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"
    output:
        directory("eda_results/")
    script:
        "scripts/eda.py"

# Clustering Rule
rule kmeans_clustering:
    input:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"
    output:
        directory("clustering_results/kmeans"),
        cluster_labels = "clustering_results/kmeans/kmeans_clusters.csv",
        pca_plot = "clustering_results/kmeans/pca_kmeans.png",
        tsne_plot = "clustering_results/kmeans/tsne_kmeans.png"
    script:
        "scripts/kmeans_clustering.py"

