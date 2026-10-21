"""
Project 2: Breast Cancer Detection (binary classification)
Compares Logistic Regression, Random Forest and Gradient Boosting using
5-fold cross-validation, then evaluates the best on a held-out test set.
Dataset (Wisconsin Diagnostic) is bundled with scikit-learn.

Run: python breast_cancer_detection.py
"""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0 = malignant, 1 = benign
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    cv = {}
    for name, m in models.items():
        s = cross_val_score(m, Xtr, ytr, cv=5, scoring="roc_auc")
        cv[name] = s.mean()
        print(f"{name:<20} CV ROC-AUC: {s.mean():.4f} (+/- {s.std():.4f})")

    best_name = max(cv, key=cv.get)
    best = models[best_name].fit(Xtr, ytr)
    proba = best.predict_proba(Xte)[:, 1]
    pred = best.predict(Xte)

    print(f"\nBest model: {best_name}")
    print(f"Test ROC-AUC: {roc_auc_score(yte, proba):.4f}\n")
    print(classification_report(yte, pred, target_names=["malignant", "benign"]))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))
    joblib.dump(best, OUT / "cancer_model.joblib")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    RocCurveDisplay.from_predictions(yte, proba, ax=ax[0], name=best_name)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax[0].set_title("ROC curve (test set)")

    # feature importance (tree models) or |coefficients| (logistic regression)
    est = best[-1] if hasattr(best, "steps") else best
    imp = (pd.Series(est.feature_importances_, index=X.columns)
           if hasattr(est, "feature_importances_")
           else pd.Series(abs(est.coef_[0]), index=X.columns))
    imp.sort_values().tail(10).plot.barh(ax=ax[1], color="steelblue")
    ax[1].set_title("Top 10 most influential features")
    plt.tight_layout()
    plt.savefig(OUT / "results.png", dpi=110)
    print("Saved results.png")


if __name__ == "__main__":
    main()
