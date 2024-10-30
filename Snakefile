# Rule 1: Data Preprocessing
rule preprocess:
    input:
        "data/heart_stroke_data.csv"
    output:
        "output/preprocessed_data.csv"
    script:
        "scripts/preprocess.py"