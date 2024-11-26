from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import ParameterGrid
import numpy as np
import seaborn as sns
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

def evaluate_isolation_forest(data, param_grid, plot_dir, true_labels=None):
    results = []
    os.makedirs(plot_dir, exist_ok=True)

    for params in ParameterGrid(param_grid):
        iso_forest = IsolationForest(**params, random_state=RANDOM_SEED)
        iso_forest.fit(data)
        scores = iso_forest.decision_function(data)
        labels = iso_forest.predict(data)

        avg_inlier_score = np.mean(scores[labels == 1])
        proportion_outliers = (labels == -1).mean()

        result = {
            "params": params,
            "avg_inlier_score": avg_inlier_score,
            "proportion_outliers": proportion_outliers,
        }

        if true_labels is not None:
            precision = precision_score(true_labels, labels, pos_label=-1, zero_division=0)
            recall = recall_score(true_labels, labels, pos_label=-1, zero_division=0)
            f1 = f1_score(true_labels, labels, pos_label=-1, zero_division=0)
            try:
                roc_auc = roc_auc_score(true_labels, scores)
            except ValueError:
                roc_auc = np.nan

            result.update({
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            })

        results.append(result)

        plt.figure(figsize=(10, 6))
        sns.histplot(scores, bins=50, kde=True, color='blue', label='Anomaly Scores', alpha=0.6)
        plt.axvline(0, color='red', linestyle='--', label='Decision Boundary')
        plt.title(f"Anomaly Score Distribution\nParams: {json.dumps(params)}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.legend()
        plot_filename = os.path.join(
            plot_dir,
            f"anomaly_scores_contam_{params['contamination']}_estim_{params['n_estimators']}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return results

def isolation_forest(data, best_params):
    iso_forest = IsolationForest(**best_params, random_state=RANDOM_SEED)
    labels = iso_forest.fit_predict(data)
    data['Outlier'] = labels
    return data

def plot_tuning_summary(results_df, param_name, plot_file):
    plt.figure(figsize=(12, 8))
    
    sns.set_style("whitegrid")
    
    if 'f1_score' in results_df.columns:
        scatter = sns.scatterplot(
            x=param_name, 
            y="f1_score", 
            hue="contamination",  # Now accessible directly
            size="n_estimators",
            data=results_df,
            palette="viridis",
            sizes=(20, 200),
            alpha=0.7
        )
        plt.ylabel("F1 Score")
        plt.title(f"Tuning Summary for {param_name}")
    else:
        sns.lineplot(
            x=param_name, 
            y="avg_inlier_score", 
            data=results_df, 
            marker='o', label="Average Inlier Score"
        )
        sns.lineplot(
            x=param_name, 
            y="proportion_outliers", 
            data=results_df, 
            marker='o', label="Proportion of Outliers"
        )
        plt.ylabel("Score / Proportion")
        plt.title(f"Tuning Summary for {param_name}")

    plt.xlabel(param_name)
    plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_3d(data, labels, output_plot_file):
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(data)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Outlier'] = labels

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        pca_df['PC1'], 
        pca_df['PC2'], 
        pca_df['PC3'], 
        c=pca_df['Outlier'], 
        cmap='coolwarm',
        s=20,
        alpha=0.6
    )

    ax.set_title('Isolation Forest Outliers (3D PCA)', fontsize=15)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)

    legend = ax.legend(*scatter.legend_elements(), title="Outlier")
    ax.add_artist(legend)

    plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
    plt.close()

def main(input_file, outlier_output_file, inlier_output_file, output_plot_file, anomaly_scores_plot_dir, tuning_plot_file, true_labels_file=None):
    data = pd.read_csv(input_file)
    
    if true_labels_file:
        true_labels = pd.read_csv(true_labels_file)['label'].values  # Adjust based on your label column
    else:
        true_labels = None

    features = data.drop(columns=['stroke'])  # Adjust if there are other non-feature columns

    features.fillna(features.mean(), inplace=True)

    param_grid = {
        "n_estimators": [50, 55],
        "max_samples": ['auto', 0.6, 0.7],
        "contamination": [0.001, 0.005],
        "max_features": [1.0, 0.95],
        "bootstrap": [True, False]
    }

    results = evaluate_isolation_forest(
        features, 
        param_grid, 
        anomaly_scores_plot_dir,
        true_labels=true_labels
    )
    
    results_df = pd.DataFrame(results)
    
    params_df = results_df['params'].apply(pd.Series)
    results_df = pd.concat([results_df.drop(columns=['params']), params_df], axis=1)

    print("Flattened Results DataFrame:")
    print(results_df.head())

    isolation_forest_tuning_summary_csv = f'{tuning_plot_file}_evaluation_results.csv'
    results_df.to_csv(isolation_forest_tuning_summary_csv, index=False)

    plot_tuning_summary(results_df, "contamination", tuning_plot_file)

    if true_labels is not None:
        best_result = results_df.loc[results_df['f1_score'].idxmax()]
    else:
        best_result = results_df.loc[results_df['avg_inlier_score'].idxmax()]
    best_params = best_result.to_dict()
    param_keys = ["n_estimators", "max_samples", "contamination", "max_features", "bootstrap"]
    best_params = {k: best_params[k] for k in param_keys}
    print(f"Best Parameters: {best_params}")

    iso_forest_best = IsolationForest(**best_params, random_state=RANDOM_SEED)
    iso_forest_best.fit(features)
    labels = iso_forest_best.predict(features)

    data['Outlier'] = labels

    outliers_data = data[data['Outlier'] == -1]
    inliers_data = data[data['Outlier'] == 1]

    outliers_data.to_csv(outlier_output_file, index=False)
    inliers_data.to_csv(inlier_output_file, index=False)

    visualize_3d(features, data['Outlier'], output_plot_file)



if __name__ == "__main__":
    input_file = snakemake.input[0]
    outlier_output_file = snakemake.output[0]
    inlier_output_file = snakemake.output[1]
    RANDOM_SEED = snakemake.params.random_seed
    output_plot_file = snakemake.output[2]
    anomaly_scores_plot_file = snakemake.params.anomaly_scores_plot
    tuning_plot_file = snakemake.params.tuning_summary_plot
    main(input_file, outlier_output_file, inlier_output_file, output_plot_file, anomaly_scores_plot_file, tuning_plot_file)