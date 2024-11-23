from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dbscan

RANDOM_SEED = 42

def isolation_forest(data):
    iso_forest = IsolationForest(random_state=RANDOM_SEED)
    outlier_labels = iso_forest.fit_predict(data)
    data['Outlier'] = outlier_labels
    return data

def visualize_3d(data, labels, output_plot_file):
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(data)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='coolwarm', s=10)
    ax.set_title('Isolation Forest Outliers (3D PCA)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.colorbar(scatter, ax=ax, label='Outlier Label')
    plt.savefig(output_plot_file)
    #plt.show()

def main(input_file, outlier_output_file, inlier_output_file, output_plot_file):
    data = pd.read_csv(input_file)
    
    data_with_outliers = isolation_forest(data)

    outliers_data = data_with_outliers[data_with_outliers['Outlier'] == -1]
    inliers_data = data_with_outliers[data_with_outliers['Outlier'] == 1]

    outliers_data.to_csv(outlier_output_file, index=False)
    inliers_data.to_csv(inlier_output_file, index=False)

    visualize_3d(data.drop(columns=['Outlier']), data_with_outliers['Outlier'], output_plot_file)

if __name__ == "__main__":
    input_file = snakemake.input[0]
    outlier_output_file = snakemake.output[0]
    inlier_output_file = snakemake.output[1]
    RANDOM_SEED = snakemake.params.random_seed
    output_plot_file = snakemake.output[2]
    main(input_file, outlier_output_file, inlier_output_file, output_plot_file)