RANDOM_SEED = 42
TARGET = "stroke"

rule all:
    input:
        # Preprocess
        "data/processed/healthcare-dataset-stroke-data-processed.csv",
        # EDA
        "eda_results/.done",
        # SMOTE
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv",
        # Outlier Detection
        "output/outlier_detection/isolation_forest/isolation_forest_outlier_detection_results.csv",
        "output/outlier_detection/isolation_forest/isolation_forest_inliers_detection_results.csv",
        "output/outlier_detection/isolation_forest/isolation_forest_outlier_detection_plot.png",
        "output/outlier_detection/local_outlier_factor/lof_outlier_detection_results.csv",
        "output/outlier_detection/local_outlier_factor/lof_inliers_detection_results.csv",
        "output/outlier_detection/local_outlier_factor/lof_outlier_detection_plot.png",
        "output/outlier_detection/local_outlier_factor/lof_tuning_results.json",
        # Clustering
        "output/clustering/dbscan/dbscan_clustering_results.csv",
        "output/clustering/dbscan/dbscan_clustering_plot.png",
        "output/clustering/dbscan/dbscan_clustering_results_metrics.json",
        "output/clustering/dbscan/dbscan_k_distance_plot.png",
        "output/clustering/dbscan/dbscan_metrics_heatmap.png",
        "output/clustering/dbscan/dbscan_clustering_results_with_outliers_removal.csv",
        "output/clustering/dbscan/dbscan_clustering_plot_with_outliers_removal.png",
        "output/clustering/dbscan/dbscan_clustering_results_metrics_with_outliers_removal.json",
        "output/clustering/dbscan/dbscan_k_distance_plot_with_outliers_removal.png",
        "output/clustering/dbscan/dbscan_metrics_heatmap_with_outliers_removal.png",
        "output/clustering/kmeans/kmeans_clustering_results.csv",
        "output/clustering/kmeans/kmeans_clustering_plot.png",
        "output/clustering/kmeans/kmeans_clustering_results_metrics.json",
        "output/clustering/kmeans/kmeans_clustering_results_with_outliers_removal.csv",
        "output/clustering/kmeans/kmeans_clustering_plot_with_outliers_removal.png",
        "output/clustering/kmeans/kmeans_clustering_results_metrics_with_outliers_removal.json",
        # Feature Selection
        "output/feature_selection/mutual_information_reduced_features.csv",
        "output/feature_selection/rfe_reduced_features.csv",
        # Classification
        "output/classification/random_forest/random_forest_results_cv_scores.csv",
        "output/classification/random_forest/random_forest_results_scalar_metrics.csv",
        "output/classification/random_forest/random_forest_plot.png",
        "output/classification/random_forest/random_forest_cm.png",
        "output/classification/random_forest/random_forest_roc_plot.png",
        "output/classification/random_forest/random_forest_lc_plot.png",
        "output/classification/svm/svm_results_scalar_metrics.csv",
        "output/classification/svm/svm_confusion_matrix.png",
        "output/classification/svm/svm_roc_curve.png",
        # Hyperparameter Tuning
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_results.json",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_rf_plot.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_mi_results.json",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_halving_plot.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_confusion_matrix.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_roc_curve.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_learning_curve.png",
        "output/hyperparameter_tuning/random_search/random_search_results.csv",
        "output/hyperparameter_tuning/random_search/random_search_confusion_matrix.png",
        "output/hyperparameter_tuning/random_search/random_search_roc_curve.png"


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
        directory("eda_results/"),
        "eda_results/.done"
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
        "output/outlier_detection/isolation_forest/isolation_forest_outlier_detection_results.csv",
        "output/outlier_detection/isolation_forest/isolation_forest_inliers_detection_results.csv",
        "output/outlier_detection/isolation_forest/isolation_forest_outlier_detection_plot.png",
    params:
        random_seed=RANDOM_SEED,
        anomaly_scores_plot = 'output/outlier_detection/isolation_forest/isolation_forest_anomaly_scores_plot',
        tuning_summary_plot = 'output/outlier_detection/isolation_forest/isolation_forest_tuning_summary_plot'
    script:
        "scripts/isolation_forest.py"

rule local_outlier_factor:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/outlier_detection/local_outlier_factor/lof_outlier_detection_results.csv",
        "output/outlier_detection/local_outlier_factor/lof_inliers_detection_results.csv",
        "output/outlier_detection/local_outlier_factor/lof_outlier_detection_plot.png",
        "output/outlier_detection/local_outlier_factor/lof_tuning_results.json"
    script:
        "scripts/local_outlier_factor.py"

#Clustering DBSCAN Rule
rule dbscan:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/clustering/dbscan/dbscan_clustering_results.csv",
        "output/clustering/dbscan/dbscan_clustering_plot.png",
        "output/clustering/dbscan/dbscan_clustering_results_metrics.json",
        "output/clustering/dbscan/dbscan_k_distance_plot.png",
        "output/clustering/dbscan/dbscan_metrics_heatmap.png"
    script:
        "scripts/dbscan.py"

#Clustering DBSCAN Rule
rule dbscan_with_outliers_removal:
    input:
        "output/outlier_detection/isolation_forest_inliers_detection_results.csv"
    output:
        "output/clustering/dbscan/dbscan_clustering_results_with_outliers_removal.csv",
        "output/clustering/dbscan/dbscan_clustering_plot_with_outliers_removal.png",
        "output/clustering/dbscan/dbscan_clustering_results_metrics_with_outliers_removal.json",
        "output/clustering/dbscan/dbscan_k_distance_plot_with_outliers_removal.png",
        "output/clustering/dbscan/dbscan_metrics_heatmap_with_outliers_removal.png"
    script:
        "scripts/dbscan.py"

# Clustering KMeans Rule
rule kmeans:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/clustering/kmeans/kmeans_clustering_results.csv",
        "output/clustering/kmeans/kmeans_clustering_plot.png",
        "output/clustering/kmeans/kmeans_clustering_results_metrics.json"
    params:
        random_seed=RANDOM_SEED
    script:
        "scripts/kmeans.py"

# Clustering KMeans Rule
rule kmeans_with_outliers_removal:
    input:
        "output/outlier_detection/local_outlier_factor/lof_inliers_detection_results.csv"
    output:
        "output/clustering/kmeans/kmeans_clustering_results_with_outliers_removal.csv",
        "output/clustering/kmeans/kmeans_clustering_plot_with_outliers_removal.png",
        "output/clustering/kmeans/kmeans_clustering_results_metrics_with_outliers_removal.json",
    params:
        random_seed=RANDOM_SEED
    script:
        "scripts/kmeans.py"

#Feature Selection Rule
rule mutual_information_feature_selection:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/feature_selection/mutual_information_reduced_features.csv"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/mutual_information.py"

# RFE Feature Selection Rule
rule recursive_feature_elimination:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/feature_selection/rfe_reduced_features.csv"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/recursive_feature_elimination.py"

# Random Forest Classifier Rule
rule random_forest_classifier:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/classification/random_forest/random_forest_results_cv_scores.csv",
        "output/classification/random_forest/random_forest_results_scalar_metrics.csv",
        "output/classification/random_forest/random_forest_plot.png",
        "output/classification/random_forest/random_forest_cm.png",
        "output/classification/random_forest/random_forest_roc_plot.png",
        "output/classification/random_forest/random_forest_lc_plot.png"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/random_forest.py"

# SVM Classifier Rule
rule svm_classifier:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/classification/svm/svm_results_scalar_metrics.csv",
        "output/classification/svm/svm_confusion_matrix.png",
        "output/classification/svm/svm_roc_curve.png"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/support_vector_machine.py"


# GridSearchCV Rule
rule grid_search_cv:
    input:
        "output/feature_selection/mutual_information_reduced_features.csv"
    output:
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_results.json",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_rf_plot.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_mi_results.json",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_halving_plot.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_confusion_matrix.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_roc_curve.png",
        "output/hyperparameter_tuning/grid_search_cv/grid_search_cv_learning_curve.png"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/gridsearchcv.py"

# RandomSearch Hyperparameter Tuning Rule:
rule random_search:
    input:
        "data/processed/healthcare-dataset-stroke-data-oversampled.csv"
    output:
        "output/hyperparameter_tuning/random_search/random_search_results.csv",
        "output/hyperparameter_tuning/random_search/random_search_confusion_matrix.png",
        "output/hyperparameter_tuning/random_search/random_search_roc_curve.png"
    params:
        target=TARGET,
        random_seed=RANDOM_SEED
    script:
        "scripts/random_search.py"
