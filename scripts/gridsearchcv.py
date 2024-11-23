from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import pandas as pd

RANDOM_SEED = 42

def grid_search_rf(X_train, y_train, random_state):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'class_weight': ['balanced'],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_tuned_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability predictions
    
    print("Tuned Random Forest Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Tuned AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.2f}")
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "auc_roc": roc_auc_score(y_test, y_pred_proba)
    }

def main(input_file, target_column, output_file):
    
    data = pd.read_csv(input_file)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    best_rf_model, best_params = grid_search_rf(X_train, y_train, random_state=RANDOM_SEED)
    
    test_metrics = evaluate_tuned_model(best_rf_model, X_test, y_test)
    
    results = {
        "best_params": best_params,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "auc_roc": test_metrics["auc_roc"]
        }
    }
    pd.DataFrame.from_dict(results, orient="index").to_csv(output_file, header=False)
    print(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    input_file = snakemake.input[0]          # Input CSV file
    target_column = snakemake.params.target  # Target column name
    RANDOM_SEED = snakemake.params.random_seed
    output_file = snakemake.output[0]       # Output CSV file for evaluation results
    main(input_file, target_column, output_file)
