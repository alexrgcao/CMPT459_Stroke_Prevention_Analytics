import json
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


def load_data(input_file):
    """Load preprocessed data."""
    return pd.read_csv(input_file)


def detect_outliers_lof(data, n_neighbors, contamination):
    """Detect outliers using Local Outlier Factor."""
    lof = LocalOutlierFactor(n_neighbors=int(n_neighbors), contamination=contamination, novelty=False)
    outlier_labels = lof.fit_predict(data)
    data = data.copy()
    data['Outlier'] = outlier_labels
    lof_scores = lof.negative_outlier_factor_
    return data, lof_scores


def tune_lof(data, neighbor_values, contamination_values):
    """Tune LOF hyperparameters and evaluate results."""
    results = []
    for n_neighbors in neighbor_values:
        for contamination in contamination_values:
            labels, lof_scores = detect_outliers_lof(data, n_neighbors=int(n_neighbors), contamination=contamination)
            n_outliers = (labels['Outlier'] == -1).sum()
            results.append({
                "n_neighbors": int(n_neighbors),
                "contamination": contamination,
                "n_outliers": n_outliers
            })
    return pd.DataFrame(results)


def save_tuning_results_to_json(results, json_file):
    """Saves LOF tuning results to a JSON file."""
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    results_list = results.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
    with open(json_file, 'w') as f:
        json.dump(results_list, f, indent=4)


def visualize_outliers(data, labels, plot_file):
    """Visualize the data with outliers highlighted."""
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)

    inliers = data_pca[labels == 1]
    outliers = data_pca[labels == -1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c="blue", s=10, label="Inliers")
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c="red", s=20, label="Outliers")
    ax.set_title("Outlier Detection with LOF (PCA Reduced to 3D)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend()
    plt.savefig(plot_file, dpi=300)
    plt.close()


def main(input_file, outlier_output_file ,inlier_output_file, outlier_plot_file, tuning_output_file, neighbor_values, contamination_values):
    """Main function to tune LOF and visualize outliers."""
    # Load the dataset
    data = load_data(input_file)
    features = data.drop(columns=['stroke'], errors='ignore')  # Exclude irrelevant columns

    # Tune LOF hyperparameters
    results = tune_lof(features, neighbor_values, contamination_values)
    save_tuning_results_to_json(results, tuning_output_file)
    print(results)

    # Select the best hyperparameters (e.g., maximum number of outliers detected)
    best_row = results.loc[results["n_outliers"].idxmax()]
    best_n_neighbors = int(best_row["n_neighbors"])  # Ensure it's an integer
    best_contamination = best_row["contamination"]
    data_with_outliers, lof_scores = detect_outliers_lof(features, n_neighbors=best_n_neighbors, contamination=best_contamination)
    print(f"Best Hyperparameters: n_neighbors={best_n_neighbors}, contamination={best_contamination}")

    # Save data
    outliers_data = data_with_outliers[data_with_outliers['Outlier'] == -1]
    inliers_data = data_with_outliers[data_with_outliers['Outlier'] == 1]
    outliers_data.to_csv(outlier_output_file, index=False)
    inliers_data.to_csv(inlier_output_file, index=False)

    # Visualize the outliers with the best hyperparameters
    visualize_outliers(features, data_with_outliers['Outlier'], outlier_plot_file)


if __name__ == "__main__":
    input_file = snakemake.input[0]
    outlier_output_file = snakemake.output[0]
    inlier_output_file = snakemake.output[1]
    outlier_plot_file = snakemake.output[2]
    tuning_output_file = snakemake.output[3]
    neighbor_values = [10, 20, 30, 40]
    contamination_values = [0.05, 0.1, 0.15, 0.2]

    main(input_file, outlier_output_file, inlier_output_file, outlier_plot_file, tuning_output_file, neighbor_values, contamination_values)
