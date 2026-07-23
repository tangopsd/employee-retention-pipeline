"""
Employee Retention Prediction Pipeline
========================================
Cleans HR data, explores key churn drivers, and trains/compares three
classification models (Logistic Regression, Decision Tree, Random Forest)
to predict which employees are likely to leave.

Run: python src/pipeline.py
Outputs: results.json (metrics for the dashboard) + PNGs in dashboard/assets/
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "HR_comma_sep.csv")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "assets")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "dashboard", "results.json")

os.makedirs(ASSETS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize column names (fix inconsistent original naming)
    df = df.rename(
        columns={
            "Work_accident": "work_accident",
            "average_montly_hours": "average_monthly_hours",
            "time_spend_company": "tenure",
            "Department": "department",
        }
    )

    n_before = len(df)
    df = df.drop_duplicates(keep="first")
    n_after = len(df)
    print(f"Dropped {n_before - n_after} exact duplicate rows ({n_before} -> {n_after}).")

    return df


def get_tenure_outlier_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """IQR-based outlier bounds for tenure, used to bound the training set."""
    p25, p75 = df["tenure"].quantile([0.25, 0.75])
    iqr = p75 - p25
    lower = p25 - 1.5 * iqr
    upper = p75 + 1.5 * iqr
    return lower, upper


# ---------------------------------------------------------------------------
# 2. EDA plots (saved as PNGs the dashboard reads directly)
# ---------------------------------------------------------------------------
def make_eda_plots(df: pd.DataFrame) -> None:
    # Attrition split
    plt.figure(figsize=(5, 5))
    counts = df["left"].value_counts(normalize=True).sort_index()
    plt.pie(
        counts,
        labels=["Stayed", "Left"],
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"],
    )
    plt.title("Employee Attrition Split")
    plt.savefig(os.path.join(ASSETS_DIR, "attrition_split.png"), bbox_inches="tight", dpi=110)
    plt.close()

    # Monthly hours vs. number of projects, split by attrition
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(
        data=df, x="average_monthly_hours", y="number_project", hue="left", orient="h", ax=ax[0]
    )
    ax[0].invert_yaxis()
    ax[0].set_title("Monthly Hours by Number of Projects")
    sns.histplot(
        data=df, x="number_project", hue="left", multiple="dodge", shrink=0.8, ax=ax[1]
    )
    ax[1].set_title("Employee Count by Number of Projects")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "hours_by_projects.png"), bbox_inches="tight", dpi=110)
    plt.close()

    # Satisfaction vs monthly hours
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="average_monthly_hours", y="satisfaction_level", hue="left", alpha=0.4
    )
    plt.axvline(x=166.67, color="#DD4444", ls="--", label="166.67 hrs/mo (full-time avg)")
    plt.title("Satisfaction Level vs. Monthly Hours Worked")
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, "satisfaction_vs_hours.png"), bbox_inches="tight", dpi=110)
    plt.close()

    # Tenure histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="tenure", hue="left", multiple="dodge", shrink=0.8)
    plt.title("Tenure Distribution by Attrition")
    plt.savefig(os.path.join(ASSETS_DIR, "tenure_hist.png"), bbox_inches="tight", dpi=110)
    plt.close()

    # Salary by tenure bucket
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    short = df[df["tenure"] < 7]
    long_ = df[df["tenure"] >= 7]
    order = ["low", "medium", "high"]
    sns.histplot(
        data=short, x="tenure", hue="salary", hue_order=order, discrete=True,
        multiple="dodge", shrink=0.7, ax=ax[0],
    )
    ax[0].set_title("Salary by Tenure: Short-Tenured Employees (<7 yrs)")
    sns.histplot(
        data=long_, x="tenure", hue="salary", hue_order=order, discrete=True,
        multiple="dodge", shrink=0.7, ax=ax[1],
    )
    ax[1].set_title("Salary by Tenure: Long-Tenured Employees (7+ yrs)")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "salary_by_tenure.png"), bbox_inches="tight", dpi=110)
    plt.close()

    # Department breakdown
    plt.figure(figsize=(11, 6))
    pd.crosstab(df["department"], df["left"]).plot(
        kind="bar", color=["#4C72B0", "#DD8452"], ax=plt.gca()
    )
    plt.title("Employees Who Left vs. Stayed, by Department")
    plt.ylabel("Employee count")
    plt.xlabel("Department")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Stayed", "Left"])
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "department_breakdown.png"), bbox_inches="tight", dpi=110)
    plt.close()

    # Correlation heatmap
    numeric_cols = [
        "satisfaction_level", "last_evaluation", "number_project",
        "average_monthly_hours", "tenure",
    ]
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="crest")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(ASSETS_DIR, "correlation_heatmap.png"), bbox_inches="tight", dpi=110)
    plt.close()

    print(f"Saved {len(os.listdir(ASSETS_DIR))} EDA plots to {ASSETS_DIR}")


# ---------------------------------------------------------------------------
# 3. Encoding
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    df_enc["salary"] = (
        df_enc["salary"].astype("category")
        .cat.set_categories(["low", "medium", "high"])
        .cat.codes
    )
    # BUG FIX: original notebook passed drop_first="False" (a truthy string),
    # which silently behaved like drop_first=True. Using the real boolean here.
    df_enc = pd.get_dummies(df_enc, columns=["department"], drop_first=False)
    return df_enc


# ---------------------------------------------------------------------------
# 4. Modeling — compare 3 models instead of just Logistic Regression
# ---------------------------------------------------------------------------
def train_and_evaluate(df_enc: pd.DataFrame, lower: float, upper: float) -> dict:
    df_model = df_enc[(df_enc["tenure"] >= lower) & (df_enc["tenure"] <= upper)]

    y = df_model["left"]
    X = df_model.drop("left", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=300, max_depth=10, n_jobs=-1
        ),
    }

    results = {}
    best_name, best_f1 = None, -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(
            y_test, y_pred,
            target_names=["Predicted would stay", "Predicted would leave"],
            output_dict=True,
        )
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision_leave": round(report["Predicted would leave"]["precision"], 4),
            "recall_leave": round(report["Predicted would leave"]["recall"], 4),
            "f1_leave": round(report["Predicted would leave"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "confusion_matrix": cm.tolist(),
        }

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(5, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Stayed", "Left"]
        )
        disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
        ax.set_title(f"{name} — Confusion Matrix")
        fname = name.lower().replace(" ", "_") + "_cm.png"
        plt.savefig(os.path.join(ASSETS_DIR, fname), bbox_inches="tight", dpi=110)
        plt.close()
        results[name]["cm_plot"] = fname

        f1 = results[name]["f1_leave"]
        if f1 > best_f1:
            best_f1, best_name = f1, name

        print(f"{name}: accuracy={results[name]['accuracy']}, "
              f"recall(leave)={results[name]['recall_leave']}, "
              f"f1(leave)={results[name]['f1_leave']}")

    # Feature importance for the best tree-based model (if applicable)
    feature_importance = None
    if best_name in ("Random Forest", "Decision Tree"):
        model = models[best_name]
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
            ascending=False
        ).head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, color="#4C72B0")
        plt.title(f"Top 10 Feature Importances — {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), bbox_inches="tight", dpi=110)
        plt.close()
        feature_importance = importances.round(4).to_dict()

    results["_best_model"] = best_name
    results["_feature_importance"] = feature_importance
    results["_n_train"] = len(X_train)
    results["_n_test"] = len(X_test)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_and_clean_data(DATA_PATH)
    lower, upper = get_tenure_outlier_bounds(df)
    print(f"Tenure outlier bounds: [{lower}, {upper}]")

    make_eda_plots(df)

    df_enc = encode_features(df)
    results = train_and_evaluate(df_enc, lower, upper)

    summary = {
        "dataset_size": len(df),
        "attrition_rate": round(df["left"].mean(), 4),
        "tenure_outlier_bounds": [lower, upper],
        "models": results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {RESULTS_PATH}")
    print(f"Best model by F1 (leave class): {results['_best_model']}")


if __name__ == "__main__":
    main()
