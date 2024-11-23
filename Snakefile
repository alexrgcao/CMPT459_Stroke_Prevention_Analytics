RANDOM_SEED = 42
TARGET = "stroke"

# Preprocess Rule
rule preprocess:
    input:
        "data/raw/healthcare-dataset-stroke-data.csv"
    output:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"
    params:
        random_seed=RANDOM_SEED
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

#SMOTE Oversampling Rule
rule smote:
    input:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"
    output:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    params:
        random_seed=RANDOM_SEED
    script:
        "scripts/oversampling_smote.py"

#Outlier Detection Rule
rule isolation_forest:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/outlier_detection/isolation_forest_outlier_detection_results.csv",
        "output/outlier_detection/isolation_forest_inliers_detection_results.csv",
        "output/outlier_detection/isolation_forest_outlier_detection_plot.png"
    params:
        random_seed=RANDOM_SEED
    script:
        "scripts/isolation_forest.py"

#Clustering DBSCAN Rule
rule dbscan:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/clustering/dbscan_clustering_results.csv",
        "output/clustering/dbscan_clustering_plot.png",
        "output/clustering/dbscan_clustering_results_metrics.json",
    script:
        "scripts/dbscan.py"

#Clustering DBSCAN Rule
rule dbscan_with_outliers_removal:
    input:
        "output/outlier_detection/isolation_forest_inliers_detection_results.csv"
    output:
        "output/clustering/dbscan_clustering_results_with_outliers_removal.csv",
        "output/clustering/dbscan_clustering_plot_with_outliers_removal.png",
        "output/clustering/dbscan_clustering_results_metrics_with_outliers_removal.json",
    script:
        "scripts/dbscan.py"

#Feature Selection Rule
rule lasso_feature_selection:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/feature_selection/lasso_reduced_features.csv"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/lasso.py"

# Random Forest Classifier Rule
rule random_forest_classifier:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/classification/random_forest_results_cv_scores.csv",
        "output/classification/random_forest_results_scalar_metrics.csv",
        "output/classification/random_forest_plot.png"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/random_forest.py"

# GridSearchCV Rule
rule grid_search_cv:
    input:
        "output/feature_selection/lasso_reduced_features.csv"
    output:
        "output/hyperparameter_tuning/grid_search_cv_results.csv"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/gridsearchcv.py"

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

