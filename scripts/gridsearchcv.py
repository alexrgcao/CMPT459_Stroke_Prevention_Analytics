import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, HalvingGridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix


RANDOM_SEED = 42

def save_metrics(metrics, metrics_file):
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_feature_importances(model, feature_names, output_file):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances - Random Forest")
    plt.bar(range(len(importances)), importances[sorted_indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_test, y_test_pred, output_file):
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_file)
    plt.show()

def plot_roc_curve(y_test, y_test_pred_prob, output_file):
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.show()

def plot_learning_curve(model, X_train, y_train, output_file):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Accuracy", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
    plt.plot(train_sizes, test_mean, label="Cross-Validation Accuracy", color="green")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green", alpha=0.2)
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(output_file)
    plt.show()

def select_features_with_mi(X, y, feature_names, k=10):
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    return X_selected, selected_feature_names, mi_scores

def train_random_forest(X_train, y_train, random_state, cv_folds=5):
    rf = RandomForestClassifier(random_state=random_state, oob_score=True)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    halving_cv = HalvingGridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        factor=2,
        resource='n_samples',
        max_resources='auto',
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )

    halving_cv.fit(X_train, y_train)
    best_rf = halving_cv.best_estimator_
    print(f"Best Random Forest Parameters: {halving_cv.best_params_}")
    print(f"Best CV Accuracy: {halving_cv.best_score_:.4f}")

    return best_rf, halving_cv.cv_results_

def plot_multiple_halving_results(halving_results, output_file):
    plt.figure(figsize=(12, 8))

    for k, cv_results in halving_results:
        results = pd.DataFrame(cv_results)
        plt.plot(results['mean_test_score'], label=f'k={k}')

    plt.title("HalvingGridSearchCV Results for Different k Values")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Test Score")
    plt.legend(title="Number of Features (k)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def main(input_file, target_column, final_metrics_file, rf_plot_file, mi_metrics_file, halving_plot_file, confusion_matrix_file, roc_curve_file, learning_curve_file):
    data = pd.read_csv(input_file)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    feature_names = X.columns.tolist()

    final_results = {}
    halving_results = []
    best_model_info = {"accuracy": 0, "model": None, "X_test": None, "y_test": None, "selected_features": None}

    k_values = [8, 9, 10]

    for k in k_values:
        X_selected, selected_feature_names, mi_scores = select_features_with_mi(X, y, feature_names, k=k)
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
        
        best_rf, rf_cv_results = train_random_forest(X_train, y_train, random_state=RANDOM_SEED)
        halving_results.append((k, rf_cv_results))

        rf_pred_train = best_rf.predict(X_train)
        rf_pred_test = best_rf.predict(X_test)
        rf_pred_prob_test = best_rf.predict_proba(X_test)

        train_accuracy = accuracy_score(y_train, rf_pred_train)
        test_accuracy = accuracy_score(y_test, rf_pred_test)

        rf_metrics = {
            "Best Parameters": best_rf.get_params(),
            "Best CV Accuracy": max(rf_cv_results['mean_test_score']),
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Classification Report": classification_report(y_test, rf_pred_test, output_dict=True)
        }

        final_results[k] = rf_metrics

        if test_accuracy > best_model_info["accuracy"]:
            best_model_info.update({
                "accuracy": test_accuracy,
                "model": best_rf,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": rf_pred_test,
                "y_pred_prob": rf_pred_prob_test,
                "selected_features": selected_feature_names,
                "X_train": X_train,
                "y_train": y_train
            })

    best_rf = best_model_info["model"]
    plot_feature_importances(best_rf, best_model_info["selected_features"], rf_plot_file)
    plot_confusion_matrix(best_model_info["y_test"], best_model_info["y_pred"], confusion_matrix_file)
    plot_roc_curve(best_model_info["y_test"], best_model_info["y_pred_prob"], roc_curve_file)
    plot_learning_curve(best_rf, best_model_info["X_train"], best_model_info["y_train"], learning_curve_file)

    mi_metrics = {
        "Feature Names": feature_names,
        "MI Scores": mi_scores.tolist(),
        "Results for each k": final_results
    }
    save_metrics(mi_metrics, mi_metrics_file)

    plot_multiple_halving_results(halving_results, halving_plot_file)

    final_metrics = {
        "MI-Based Feature Selection": mi_metrics,
        "Random Forest": rf_metrics
    }
    save_metrics(final_metrics, final_metrics_file)
    print(f"All metrics saved to {final_metrics_file}")


if __name__ == "__main__":
    input_file = snakemake.input[0]
    target_column = snakemake.params.target
    RANDOM_SEED = snakemake.params.random_seed
    final_metrics_file = snakemake.output[0]
    rf_plot_file = snakemake.output[1]
    mi_metrics_file = snakemake.output[2]
    halving_plot_file = snakemake.output[3]
    confusion_matrix_file = snakemake.output[4]
    roc_curve_file = snakemake.output[5]
    learning_curve_file = snakemake.output[6]

    main(input_file, target_column, final_metrics_file, rf_plot_file, mi_metrics_file, halving_plot_file, confusion_matrix_file, roc_curve_file, learning_curve_file)
