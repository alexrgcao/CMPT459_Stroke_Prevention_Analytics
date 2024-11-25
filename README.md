# CMPT459 Stroke Prevention Analytics

## Overview

This project focuses on analyzing the **Stroke Prediction Dataset** to extract patterns regarding stroke likelihood using various data mining techniques. Using techniques like clustering, classification, and exploratory data analysis (EDA), we hope to get insights into factors influencing stroke risk. The dataset was sourced from Kaggle, consisting of both numerical and categorical features, making it suitable for this task.

---

## Project Proposal

### Objective
Stroke has been identified as the second leading cause of death globally by the World Health Organization. Because of this, it is important to gain an early insight on the chances an individual can get a stroke to allow for intervention and prevention. In this project, through performing multiple data mining tasks, we wish to achieve the ability to predict the chance of someone having a stroke based on their various health-related attributes.

### Dataset Description
- **Name**: Stroke Prediction Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Samples**: 5110 observations
- **Features**: 11 clinical and demographic attributes
  - **Target Feature**: `stroke` (1 for stroke, 0 for no stroke)
  - **Feature Types**: 7 numerical and 5 categorical variables
- **License**: Permissive for academic and research purposes

### Justification
- **Usability**: High-quality dataset with a usability score of 10.00 on Kaggle.
- **Structure**: CSV format with clear feature descriptions.
- **Relevance**: Directly addresses a critical healthcare issue.
- **Size**: Sufficient data for robust training, testing, and validation.
- **Diversity**: Balanced mix of feature types enables multiple analytical methods.

---

## Repository Structure

```
├── clustering_results/          # Results from clustering analysis (e.g., KMeans, DBSCAN)
├── data/                        # Raw and processed dataset files
├── eda_results/                 # EDA results (latest iteration)
├── eda_results_prev/            # Previous EDA versions for reference
├── output/                      # Outputs from models and evaluations (e.g., oversampling, classification results)
├── scripts/                     # Python scripts for preprocessing and analysis
├── EDA_Report.pdf               # Comprehensive EDA summary (PDF)
├── Snakefile                    # Workflow automation script (Snakemake)
├── README.md                    # Project documentation
├── requirement.txt              # Python dependencies
```

---

## Requirements

### Python Dependencies
Install the required libraries using:
```bash
pip install -r requirement.txt
```

### Key Libraries
- **Data Analysis**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Modeling**: `sklearn`

---

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/stroke-prediction-project.git
   cd stroke-prediction-project
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

3. **Execute Workflow**:
   Use Snakemake to automate the analysis pipeline:
   ```bash
   snakemake --cores 4
   ```

4. **View Outputs**:
   - Clustering results: `clustering_results/`
   - EDA summary: `EDA_Report.pdf`
   - Model evaluations: `output/`

---

## Acknowledgments
- Dataset by [Federico Soriano on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Inspired by the need for actionable insights in global healthcare.

## License
This project is licensed under a permissive license for academic and research purposes.

---
