import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(input_file):
    return pd.read_csv(input_file)


def plot_histograms(data, output_dir):
    """Plot histograms for all numerical features."""
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
        plt.close()

'''
def plot_categorical_distributions(data, output_dir):
    """Plot count plots for all categorical features."""
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(12, 10))
        sns.countplot(x=data[col])
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, f"count_{col}.png"), bbox_inches='tight')
        plt.close()
'''

def plot_correlation_heatmap(data, output_dir):
    """Plot a correlation heatmap for numerical features."""
    numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns

    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()


def plot_boxplot(data, output_dir):
    """Plot boxplots for all numerical features vs. the stroke target."""
    numerical_cols = data.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        if col != 'stroke':  # Avoid plotting stroke against itself
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='stroke', y=col, data=data)
            plt.title(f'Box Plot of {col} vs Stroke')
            plt.savefig(os.path.join(output_dir, f"boxplot_{col}_stroke.png"))
            plt.close()


def main(input_file, output_dir):
    data = load_data(input_file)

    os.makedirs(output_dir, exist_ok=True)

    # Generate and save plots
    plot_histograms(data, output_dir)
    #plot_categorical_distributions(data, output_dir)
    plot_correlation_heatmap(data, output_dir)
    plot_boxplot(data, output_dir)

    with open(os.path.join(output_dir, ".done"), "w") as f:
      f.write("EDA completed successfully.")


if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_dir = snakemake.output[0]
    main(input_file, output_dir)