from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

RANDOM_SEED = 42

def apply_lasso(data, target, random_state):
    
    alphas = np.logspace(-4, 0, 200)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    lasso = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=5000,
        tol=1e-5,
        selection="random",
        random_state=random_state
    )
    
    lasso.fit(data, target)
    
    importance = np.abs(lasso.coef_)
    selected_features = data.columns[importance > 0]
    print(f"Selected Features by Lasso: {list(selected_features)}")
    
    reduced_data = data[selected_features]
    return reduced_data, list(selected_features)

def main(input_file, target_column, output_file):

    data = pd.read_csv(input_file)
    
    target = data[target_column]
    features = data.drop(columns=[target_column])
    
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    reduced_data, selected_features = apply_lasso(features_scaled, target, random_state=RANDOM_SEED)
    
    reduced_data[target_column] = target
    reduced_data.to_csv(output_file, index=False)
    print(f"Reduced dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = snakemake.input[0]
    target_column = snakemake.params.target
    RANDOM_SEED = snakemake.params.random_seed
    output_file = snakemake.output[0]
    main(input_file, target_column, output_file)
