import pandas as pd
import numpy as np
import os

def load_data(input_file):
    return pd.read_csv(input_file)

def clean_gender(data):
    data = data.loc[data["gender"] != "Other"].copy()
    data.loc[:, "gender"] = data["gender"].map({"Female": 0, "Male": 1})
    return data

def clean_age(data):
    data["age"] = data["age"].astype(float)
    return data

def clean_bmi(data):
    data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce")
    data["bmi"].fillna(data["bmi"].mean(), inplace=True)
    data["bmi"] = data["bmi"].astype(float)
    return data

def clean_smoking_status(data):
    data["smoking_status"].replace("Unknown", np.nan, inplace=True)
    probabilities = data["smoking_status"].value_counts(normalize=True).loc[["never smoked", "formerly smoked", "smokes"]]
    data["smoking_status"] = data["smoking_status"].apply(
        lambda x: np.random.choice(["never smoked", "formerly smoked", "smokes"], p=probabilities) if pd.isnull(x) else x
    )
    return data

def remove_feature(data, col):
    data = data.drop(data.columns[col], axis=1)
    return data

def main(input_file, output_file):
    data = load_data(input_file)
    data = remove_feature(data, 0)
    data = clean_gender(data)
    data = clean_age(data)
    data = clean_bmi(data)
    data = clean_smoking_status(data)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    data.to_csv(output_file, index=False)

# Run only if this script is called directly (not imported)
if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    main(input_file, output_file)