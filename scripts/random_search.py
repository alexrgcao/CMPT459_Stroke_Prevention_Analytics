import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform


def random_search_svm(X_train, y_train, random_state):
    """Perform Random Search for hyperparameter tuning on SVM."""
    param_distributions = {
        'C': uniform(loc=0.1, scale=10.0),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'probability': [True],
    }

    random_search = RandomizedSearchCV(
        estimator=SVC(random_state=random_state),
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
    )

    random_search.fit(X_train, y_train)

    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


def evaluate_tuned_model(model, X_test, y_test):
    """Evaluate the tuned SVM model on the test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Tuned SVM Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Tuned AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    return y_pred, y_pred_proba, {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
    }


def plot_confusion_matrix(y_true, y_pred, output_file):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"],
                yticklabels=["No Stroke", "Stroke"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_file)
    plt.show()
    print(f"Confusion matrix saved to {output_file}")


def plot_roc_curve(y_test, y_pred_proba, output_file):
    """Plot and save ROC curve."""
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    plt.title("ROC Curve")
    plt.savefig(output_file)
    plt.show()
    print(f"ROC curve saved to {output_file}")


def main(input_file, target_column, random_seed, output_metrics_file, confusion_matrix_file, roc_curve_file):
    """Main function to load data, tune SVM, and evaluate the model."""
    # Load dataset
    data = pd.read_csv(input_file)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )

    # Perform Random Search for SVM hyperparameter tuning
    best_svm_model, best_params = random_search_svm(X_train, y_train, random_seed)

    # Evaluate the tuned model
    y_pred, y_pred_proba, metrics = evaluate_tuned_model(best_svm_model, X_test, y_test)

    # Save evaluation metrics
    pd.DataFrame([metrics]).to_csv(output_metrics_file, index=False)
    print(f"Metrics saved to {output_metrics_file}")

    # Visualize confusion matrix
    plot_confusion_matrix(y_test, y_pred, confusion_matrix_file)

    # Visualize ROC curve
    plot_roc_curve(y_test, y_pred_proba, roc_curve_file)


if __name__ == "__main__":
    input_file = snakemake.input[0]
    target_column = snakemake.params.target
    random_seed = snakemake.params.random_seed
    output_metrics_file = snakemake.output[0]
    confusion_matrix_file = snakemake.output[1]
    roc_curve_file = snakemake.output[2]

    main(input_file, target_column, random_seed, output_metrics_file, confusion_matrix_file, roc_curve_file)
