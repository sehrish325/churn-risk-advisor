import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

sns.set_theme(style="whitegrid")

print("Week 2 libraries imported successfully!")

import os

if "data" not in globals():
    if os.path.exists("telco_churn_cleaned.csv"):
        data = pd.read_csv("telco_churn_cleaned.csv")
    else:
        raise FileNotFoundError(
            "Cleaned dataset nahi mili. Week 1 ka Step 10 run karein "
            "ya cleaned CSV notebook mein upload karein."
        )

# Check the dataset
print("Dataset shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

print("\nChurn distribution:")
print(data["Churn"].value_counts(dropna=False))

display(data.head(10))

data = data.copy()

data["Churn"] = pd.to_numeric(data["Churn"], errors="coerce")
data = data.dropna(subset=["Churn"])
data["Churn"] = data["Churn"].astype(int)

X = data.drop(columns=["Churn"])
y = data["Churn"]

print("Features:", X.shape)
print("Target:", y.shape)
print("\nTarget classes:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining churn distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting churn distribution:")
print(y_test.value_counts(normalize=True).round(3))


numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric features:", numeric_features)
print("\nCategorical features:", categorical_features)


numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully!")


logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed!")


rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")


extra_trees_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

extra_trees_model.fit(X_train, y_train)

print("Extra Trees training completed!")


models = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "Extra Trees": extra_trees_model
}

results = []
probabilities = {}

for name, model in models.items():

    # Probability of churn (class 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Default threshold = 0.50
    y_pred = (y_prob >= 0.50).astype(int)

    probabilities[name] = y_prob

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "F2 Score": fbeta_score(
            y_test, y_pred, beta=2, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results).set_index("Model")

display(results_df.round(4))


metrics_to_plot = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "F2 Score",
    "ROC-AUC"
]

results_df[metrics_to_plot].plot(
    kind="bar",
    figsize=(13, 6)
)

plt.title("Week 2: Machine Learning Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


best_model_name = results_df["F2 Score"].idxmax()
best_model = models[best_model_name]

best_prob = probabilities[best_model_name]
best_pred = (best_prob >= 0.50).astype(int)

print("Best model by F2-score:", best_model_name)

print("\nClassification report:")
print(classification_report(
    y_test,
    best_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.title(f"Confusion Matrix — {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

for name, y_prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

for name, y_prob in probabilities.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.plot(
        recall,
        precision,
        label=name
    )

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()


results_df.to_csv("week2_model_comparison.csv")

# Save actual and predicted probabilities for analysis
predictions_df = pd.DataFrame({
    "Actual_Churn": y_test.to_numpy()
})

for name, y_prob in probabilities.items():
    predictions_df[f"{name}_Churn_Probability"] = y_prob

predictions_df.to_csv(
    "week2_test_predictions.csv",
    index=False
)

print("Week 2 results saved:")
print("- week2_model_comparison.csv")
print("- week2_test_predictions.csv")
