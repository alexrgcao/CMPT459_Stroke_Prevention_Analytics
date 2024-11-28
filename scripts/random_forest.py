from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42

def train_with_cross_val(X_train, y_train, X_test, y_test, random_state, cv_folds=5):
    rf = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=random_state
    )
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation: {np.std(cv_scores):.2f}")
    
    rf.fit(X_train, y_train)
    print(f"OOB Score: {rf.oob_score_:.2f}")
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return rf, y_test, y_test_pred, {
        "cv_scores": cv_scores,
        "mean_accuracy": np.mean(cv_scores),
        "std_accuracy": np.std(cv_scores),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

def plot_feature_importances(model, feature_names, output_file):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[sorted_indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    #plt.show()

def plot_confusion_matrix(y_test, y_test_pred, output_file):
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_file)
    #plt.show()

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
    #plt.show()

def main(input_file, target_column, output_file, plot_file, output_results_scalar_file, cm_file, roc_file, lc_file):
    data = pd.read_csv(input_file)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    
    rf_model, y_test, y_test_pred, metrics = train_with_cross_val(X_train, y_train, X_test, y_test, random_state=RANDOM_SEED)
    
    cv_df = pd.DataFrame({"cv_scores": metrics["cv_scores"]})
    cv_df.to_csv(output_file, index=False)
    
    scalar_metrics = pd.DataFrame([{
        "mean_accuracy": metrics["mean_accuracy"],
        "std_accuracy": metrics["std_accuracy"],
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"]
    }])
    scalar_metrics.to_csv(output_results_scalar_file, index=False)
    
    plot_feature_importances(rf_model, X.columns, plot_file)
    plot_confusion_matrix(y_test, y_test_pred, cm_file)
    plot_roc_curve(y_test, rf_model.predict_proba(X_test), roc_file)
    plot_learning_curve(rf_model, X_train, y_train, lc_file)
    print(f"All plots and metrics have been saved.")

if __name__ == "__main__":
    input_file = snakemake.input[0]
    target_column = snakemake.params.target
    RANDOM_SEED = snakemake.params.random_seed
    output_file = snakemake.output[0]
    output_results_scalar_file = snakemake.output[1]
    plot_file = snakemake.output[2]
    cm_file = snakemake.output[3]
    roc_file = snakemake.output[4]
    lc_file = snakemake.output[5]
    main(input_file, target_column, output_file, plot_file, output_results_scalar_file, cm_file, roc_file, lc_file)
