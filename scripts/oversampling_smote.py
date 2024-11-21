from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42

def smote_oversample(data, output_file):
    
    X = data.drop(columns=['stroke'])
    y = data['stroke']

    smote = SMOTE(random_state=RANDOM_SEED)
    X_smote, y_smote = smote.fit_resample(X, y)

    smote_counts = pd.Series(y_smote).value_counts()
    print(smote_counts)

    data_smote = pd.concat([X_smote, y_smote], axis=1)

    data_smote.to_csv(output_file, index=False)


def plot_distribution(data):
    plt.bar(data.index, data.values, tick_label=['No Stroke', 'Stroke'])
    plt.title('Class Distribution After SMOTE')
    plt.ylabel('Count')
    #plt.show()

def main(input_file, output_file):
    data = pd.read_csv(input_file)
    smote_oversample(data, output_file)
    
    oversampled_data = pd.read_csv(output_file)
    smote_counts = oversampled_data['stroke'].value_counts()
    plot_distribution(smote_counts)


if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    RANDOM_SEED = snakemake.params.random_seed
    main(input_file, output_file)