import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    precision_recall_curve
)

sns.set_theme(style="whitegrid")
print("Week 3 libraries imported successfully!")


import os
import glob

if "data" not in globals():
    if os.path.exists("telco_churn_cleaned.csv"):
        data = pd.read_csv("telco_churn_cleaned.csv")
    else:
        files = glob.glob(
            "/kaggle/input/**/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            recursive=True
        )
        if not files:
            raise FileNotFoundError(
                "Dataset nahi mili. Week 1 ka cleaned CSV upload karein "
                "ya Kaggle mein dataset add karein."
            )

        data = pd.read_csv(files[0])
        data.drop(columns=["customerID"], errors="ignore", inplace=True)
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"], errors="coerce"
        )
        data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

data = data.copy()
data["Churn"] = pd.to_numeric(data["Churn"], errors="coerce")
data = data.dropna(subset=["Churn"])
data["Churn"] = data["Churn"].astype(int)

print("Dataset shape:", data.shape)
print("\nTarget distribution:")
print(data["Churn"].value_counts())

display(data.head(10))


X = data.drop(columns=["Churn"])
y = data["Churn"]

# First: separate final test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second: create validation set from remaining 80%
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Final test:", X_test.shape)


numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("Preprocessing pipeline ready!")


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score

# Create F2 scoring function
f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

# Create Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ))
])

# Hyperparameter search space
rf_params = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 8, 12, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.7]
}

# Randomized search with cross-validation
rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_params,
    n_iter=12,
    scoring=f2_scorer,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    error_score="raise"
)

# Train and tune the model
rf_search.fit(X_train, y_train)

# Display results
print("Best Random Forest parameters:")
print(rf_search.best_params_)

print(
    "Best cross-validation F2-score:",
    round(rf_search.best_score_, 4)
)


from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score

# Create F2 scorer
f2_scorer = make_scorer(
    fbeta_score,
    beta=2,
    zero_division=0
)

# Extra Trees pipeline
et_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", ExtraTreesClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ))
])

# Hyperparameter search space
et_params = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 8, 12, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.7]
}

# Randomized search with valid F2 scoring
et_search = RandomizedSearchCV(
    estimator=et_pipeline,
    param_distributions=et_params,
    n_iter=12,
    scoring=f2_scorer,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    error_score="raise"
)

# Train and tune the model
et_search.fit(X_train, y_train)

# Display results
print("Best Extra Trees parameters:")
print(et_search.best_params_)

print(
    "Best cross-validation F2-score:",
    round(et_search.best_score_, 4)
)

lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

lr_model.fit(X_train, y_train)

candidate_models = {
    "Logistic Regression": lr_model,
    "Tuned Random Forest": rf_search.best_estimator_,
    "Tuned Extra Trees": et_search.best_estimator_
}

validation_results = []

for name, model in candidate_models.items():
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.50).astype(int)

    validation_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "F2": fbeta_score(y_val, pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_val, prob)
    })

validation_df = pd.DataFrame(validation_results).set_index("Model")
display(validation_df.round(4))


validation_df[["Precision", "Recall", "F1", "F2", "ROC-AUC"]].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Week 3: Tuned Model Comparison on Validation Data")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


best_model_name = validation_df["F2"].idxmax()
best_model = candidate_models[best_model_name]

val_prob = best_model.predict_proba(X_val)[:, 1]

threshold_rows = []

for threshold in np.arange(0.10, 0.81, 0.01):
    val_pred = (val_prob >= threshold).astype(int)

    threshold_rows.append({
        "Threshold": round(float(threshold), 2),
        "Precision": precision_score(y_val, val_pred, zero_division=0),
        "Recall": recall_score(y_val, val_pred, zero_division=0),
        "F1": f1_score(y_val, val_pred, zero_division=0),
        "F2": fbeta_score(y_val, val_pred, beta=2, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_rows)

eligible = threshold_df[threshold_df["Recall"] >= 0.85]

if not eligible.empty:
    best_threshold = float(
        eligible.sort_values(
            ["F2", "Precision"],
            ascending=False
        ).iloc[0]["Threshold"]
    )
else:
    best_threshold = float(
        threshold_df.sort_values(
            ["Recall", "F2"],
            ascending=False
        ).iloc[0]["Threshold"]
    )

print("Selected model:", best_model_name)
print("Selected threshold:", round(best_threshold, 2))

display(threshold_df.sort_values("F2", ascending=False).head(10))


plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)
plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)
plt.plot(
    threshold_df["Threshold"],
    threshold_df["F2"],
    label="F2-score",
    linewidth=2
)

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    label=f"Selected threshold = {best_threshold:.2f}"
)

plt.title("Threshold Optimization on Validation Data")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()


cluster_features = [
    col for col in ["tenure", "MonthlyCharges", "TotalCharges"]
    if col in X_train.columns
]

if len(cluster_features) < 2:
    raise ValueError(
        "Clustering ke liye kam az kam 2 numerical features zaroori hain."
    )

cluster_data = data[cluster_features].copy()

cluster_imputer = SimpleImputer(strategy="median")
cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_imputer.fit_transform(cluster_data)
)

print("Clustering features:", cluster_features)
print("Scaled clustering data:", cluster_scaled.shape)


inertias = []
silhouette_scores = []
k_values = range(2, 8)

for k in k_values:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = km.fit_predict(cluster_scaled)

    inertias.append(km.inertia_)
    silhouette_scores.append(
        silhouette_score(cluster_scaled, labels)
    )

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_values), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_values), silhouette_scores, marker="o")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

print("Silhouette scores:")
for k, score in zip(k_values, silhouette_scores):
    print(f"k={k}: {score:.4f}")


chosen_k = 3

kmeans = KMeans(
    n_clusters=chosen_k,
    random_state=42,
    n_init=10
)

data["CustomerSegment"] = kmeans.fit_predict(cluster_scaled)

print("Customer segment sizes:")
display(data["CustomerSegment"].value_counts().sort_index())

print("\nSegment profiles:")
display(
    data.groupby("CustomerSegment")[
        cluster_features + ["Churn"]
    ].mean().round(2)
)


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=data,
    x="tenure",
    y="MonthlyCharges",
    hue="CustomerSegment",
    palette="Set2",
    alpha=0.7,
    s=45
)

plt.title("Customer Segmentation: Tenure vs Monthly Charges")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.legend(title="Segment")
plt.tight_layout()
plt.show()


test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("Final selected model:", best_model_name)
print("Selected threshold:", round(best_threshold, 2))

print("\nFinal test classification report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

final_metrics = {
    "Model": best_model_name,
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, test_pred),
    "Precision": precision_score(y_test, test_pred, zero_division=0),
    "Recall": recall_score(y_test, test_pred, zero_division=0),
    "F1": f1_score(y_test, test_pred, zero_division=0),
    "F2": fbeta_score(y_test, test_pred, beta=2, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, test_prob)
}

display(pd.DataFrame([final_metrics]).round(4))

cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.title("Final Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


validation_df.to_csv("week3_validation_model_comparison.csv")
threshold_df.to_csv("week3_threshold_analysis.csv", index=False)
pd.DataFrame([final_metrics]).to_csv(
    "week3_final_test_metrics.csv",
    index=False
)

data.to_csv("week3_customer_segments.csv", index=False)

print("Week 3 outputs saved:")
print("- week3_validation_model_comparison.csv")
print("- week3_threshold_analysis.csv")
print("- week3_final_test_metrics.csv")
print("- week3_customer_segments.csv")
