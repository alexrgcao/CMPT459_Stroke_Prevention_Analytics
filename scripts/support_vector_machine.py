from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    RocCurveDisplay,
    confusion_matrix,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(input_file, target_column):
    """Load dataset and split features and target."""
    data = pd.read_csv(input_file)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def train_svm(X_train, y_train, kernel="rbf", random_state=42, cv_folds=5):
    """Train SVM model with cross-validation and return the trained model and metrics."""
    svm = SVC(
        kernel=kernel,
        probability=True,
        random_state=random_state,
        C=1.0,
        gamma="scale",
    )

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation: {np.std(cv_scores):.4f}")

    # Train the model on the entire training set
    svm.fit(X_train, y_train)

    return svm, {
        "cv_scores": cv_scores,
        "mean_cv_accuracy": np.mean(cv_scores),
        "std_cv_accuracy": np.std(cv_scores),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the model on train and test sets."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred),
        "auc_roc": roc_auc_score(y_test, y_test_proba),
    }

    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    return metrics


def save_metrics_to_csv(metrics, output_file):
    """Save metrics to a CSV file."""
    pd.DataFrame([metrics]).to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")


def plot_confusion_matrix(y_true, y_pred, output_file):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_file)
    print(f"Confusion matrix saved to {output_file}")


def plot_roc_curve(model, X_test, y_test, output_file):
    """Plot and save ROC curve."""
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(output_file)
    print(f"ROC curve saved to {output_file}")


def main(input_file, target_column, scalar_metrics_file, confusion_matrix_file, roc_curve_file, random_seed):
    """Main function to load data, train SVM, and evaluate the model."""
    # Load data
    X, y = load_data(input_file, target_column)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_seed)

    # Train SVM with cross-validation
    svm_model, cross_val_metrics = train_svm(X_train, y_train, random_state=random_seed)

    # Evaluate on train and test sets
    test_metrics = evaluate_model(svm_model, X_train, y_train, X_test, y_test)

    # Combine and save metrics
    combined_metrics = {**cross_val_metrics, **test_metrics}
    save_metrics_to_csv(combined_metrics, scalar_metrics_file)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, svm_model.predict(X_test), confusion_matrix_file)

    # Plot ROC curve
    plot_roc_curve(svm_model, X_test, y_test, roc_curve_file)


if __name__ == "__main__":
    input_file = snakemake.input[0]
    target_column = snakemake.params.target
    scalar_metrics_file = snakemake.output[0]
    confusion_matrix_file = snakemake.output[1]
    roc_curve_file = snakemake.output[2]
    random_seed = snakemake.params.random_seed

    main(input_file, target_column, scalar_metrics_file, confusion_matrix_file, roc_curve_file, random_seed)
