rule all:
    input:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"

# Preprocess Rule
rule preprocess:
    input:
        "data/raw/healthcare-dataset-stroke-data.csv"
    output:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"
    script:
        "scripts/preprocess.py"

#EDA Rule
rule eda:
    input:
        "data/processed/healthcare-dataset-stroke-data-processed.csv"
    output:
        directory("eda_results/")
    script:
        "scripts/eda.py"