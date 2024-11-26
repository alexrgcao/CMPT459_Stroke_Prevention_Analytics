from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd


def apply_rfe(data, target, random_state):
    """Apply Recursive Feature Elimination (RFE) with Random Forest."""
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    rfe = RFE(estimator=model, n_features_to_select=10, step=1)
    rfe.fit(data, target)

    selected_features = data.columns[rfe.support_]
    print(f"Selected Features by RFE: {list(selected_features)}")

    reduced_data = data[selected_features]
    return reduced_data, list(selected_features)

def main(input_file, target_column, output_file, random_state):
    """Main function to apply RFE and save the reduced dataset."""
    # Load the dataset
    data = pd.read_csv(input_file)

    # Separate features and target
    target = data[target_column]
    features = data.drop(columns=[target_column])

    # Scale the features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Apply RFE
    reduced_data, selected_features = apply_rfe(features_scaled, target, random_state=random_state)

    # Save the reduced dataset
    reduced_data[target_column] = target
    reduced_data.to_csv(output_file, index=False)
    print(f"Reduced dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = snakemake.input[0]
    target_column = snakemake.params.target
    RANDOM_SEED = snakemake.params.random_seed
    output_file = snakemake.output[0]
    main(input_file, target_column, output_file, RANDOM_SEED)
