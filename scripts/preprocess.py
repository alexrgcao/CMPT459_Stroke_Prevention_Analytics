import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import os

def load_data(input_file):
    return pd.read_csv(input_file)

def clean_feature(data):
    data.drop(columns=['id'], inplace=True)

    data = data.loc[data["gender"] != "Other"].copy()
    data.loc[:, "gender"] = data["gender"].map({"Female": 0, "Male": 1})

    data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})

    data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})

    data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce")
    data["bmi"].fillna(data["bmi"].mean(), inplace=True)
    data["bmi"] = data["bmi"].astype(float)

    data["smoking_status"].replace("Unknown", np.nan, inplace=True)
    probabilities = data["smoking_status"].value_counts(normalize=True).loc[["never smoked", "formerly smoked", "smokes"]]
    data["smoking_status"] = data["smoking_status"].apply(
        lambda x: np.random.choice(["never smoked", "formerly smoked", "smokes"], p=probabilities) if pd.isnull(x) else x
    )

    return data

def scale_feature_bmi(data):
    data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce")
    data["bmi"].fillna(data["bmi"].mean(), inplace=True)
    Q1 = data["bmi"].quantile(0.25)
    Q3 = data["bmi"].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data["bmi"] >= Q1 - 1.5 * IQR) & (data["bmi"] <= Q3 + 1.5 * IQR)]
    scaler = RobustScaler()
    data["bmi"] = scaler.fit_transform(data[["bmi"]])

    return data

def bin_glucose_levels(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data["glucose_level_cluster"] = kmeans.fit_predict(data[["avg_glucose_level"]])
    return data

def scale_numerical_features(data, cols):
    scaler = RobustScaler()
    data[cols] = scaler.fit_transform(data[cols])
    return data

def one_hot_encode(data):
    data["work_type"] = data["work_type"].replace('Never_worked', 'children')
    data = pd.get_dummies(data, columns=['work_type'], drop_first=True)

    data = pd.get_dummies(data, columns=["smoking_status"], drop_first=True)

    return data

def main(input_file, output_file):
    data = load_data(input_file)
    data = clean_feature(data)
    data = scale_feature_bmi(data)
    data = bin_glucose_levels(data)
    data = scale_numerical_features(data, ["age", "avg_glucose_level"])
    data = one_hot_encode(data)
    data = data.apply(lambda col: col.astype(int) if col.dtypes == 'bool' else col)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    main(input_file, output_file)