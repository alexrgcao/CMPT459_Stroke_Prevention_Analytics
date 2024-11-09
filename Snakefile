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