from sklearn.feature_selection import mutual_info_classif, SelectKBest
import numpy as np
import pandas as pd

RANDOM_SEED = 42

def apply_mi(data, target, k=10):
    
    mi_scores = mutual_info_classif(data, target, random_state=RANDOM_SEED)
    
    selector = SelectKBest(mutual_info_classif, k=k)
    reduced_data = selector.fit_transform(data, target)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = data.columns[selected_indices].tolist()
    
    print("Mutual Information Scores:")
    for feature, score in zip(data.columns, mi_scores):
        print(f"{feature}: {score:.4f}")
    
    print(f"Selected Features by MI: {selected_features}")
    
    return pd.DataFrame(reduced_data, columns=selected_features), selected_features

def main(input_file, target_column, output_file):
    
    data = pd.read_csv(input_file)
    
    target = data[target_column]
    features = data.drop(columns=[target_column])
    
    reduced_data, selected_features = apply_mi(features, target)
    
    reduced_data[target_column] = target.reset_index(drop=True)
    
    reduced_data.to_csv(output_file, index=False)
    print(f"Reduced dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = snakemake.input[0]
    target_column = snakemake.params.target
    RANDOM_SEED = snakemake.params.random_seed
    output_file = snakemake.output[0]
    main(input_file, target_column, output_file)