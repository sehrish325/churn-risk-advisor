import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries loaded successfully!")

file_path = "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

display(df.head(10))


print("Dataset dimensions:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null values:")
df.info()

print("\nStatistical summary:")
display(df.describe(include="all").T)

missing_report = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

display(missing_report)

print("Duplicate rows:", df.duplicated().sum())

print("Churn counts:")
display(df["Churn"].value_counts(dropna=False))

print("\nChurn percentage:")
display(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

if "Contract" in df.columns and "Churn" in df.columns:
    churn_rate = (
        df.assign(
            Churn_Flag=(
                df["Churn"].astype(str).str.strip().str.lower()
                .eq("yes").astype(int)
            )
        )
        .groupby("Contract")["Churn_Flag"]
        .mean()
        .mul(100)
        .round(2)
    )

    display(churn_rate.rename("Churn Rate (%)").to_frame())

    churn_rate.plot(kind="bar", figsize=(8, 4))
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Contract Type")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
else:
    print("Check whether the columns Contract and Churn exist.")

# Step 2: Prepare data for machine learning

import pandas as pd
from sklearn.model_selection import train_test_split

# Check the column names
print("Dataset columns:")
print(df.columns.tolist())

# Check the target column
print("\nPossible churn columns:")
print([col for col in df.columns if "churn" in col.lower()])


# Step 3: Prepare training and testing data

import pandas as pd
from sklearn.model_selection import train_test_split

# Make a copy of the dataset
data = df.copy()

# Remove customer ID because it does not help predict churn
data = data.drop(columns=["customerID"])

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

# Remove rows with missing values
data = data.dropna()

# Separate features (X) and target (y)
X = data.drop(columns=["Churn"])
y = data["Churn"].map({"Yes": 1, "No": 0})

# Convert text columns into numeric columns
X = pd.get_dummies(X, drop_first=True)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("\nTraining target distribution:")
print(y_train.value_counts())
print("\nMissing values in training data:", X_train.isnull().sum().sum())
print("Missing values in testing data:", X_test.isnull().sum().sum())

# Step 4: Train the Churn Prediction Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create the model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Training Completed!")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# CUSTOMER CHURN PREDICTION
# Class-weighted Logistic Regression + Recall-focused threshold
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Prepare target labels
# ------------------------------------------------------------
# Supports binary targets encoded as 0/1 or labeled Churn/No Churn.
# Assumption: 1 means Churn for numeric targets.

def encode_target(y):
    y = pd.Series(y).reset_index(drop=True)

    if set(y.dropna().unique()).issubset({0, 1}):
        return y.astype(int)

    normalized = y.astype(str).str.strip().str.lower()

    mapping = {
        "no churn": 0,
        "churn": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1
    }

    if set(normalized.unique()).issubset(mapping.keys()):
        return normalized.map(mapping).astype(int)

    raise ValueError(
        "Target must be encoded as 0/1 or use "
        "'Churn' and 'No Churn' labels. Check your target labels."
    )


y_train_bin = encode_target(y_train)
y_test_bin = encode_target(y_test)

# Preserve existing feature data and row alignment.
# The split below uses ONLY the original training data.

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train_bin,
    test_size=0.20,
    random_state=42,
    stratify=y_train_bin
)

print("Model fitting rows:", len(y_fit))
print("Validation rows:", len(y_val))
print("Final test rows:", len(y_test_bin))

# ------------------------------------------------------------
# 2. Build class-weighted Logistic Regression pipeline
# ------------------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("logistic", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=42
    ))
])

# Fit only on the fitting subset.
model.fit(X_fit, y_fit)

# ------------------------------------------------------------
# 3. Tune the threshold using validation data only
# ------------------------------------------------------------

val_prob = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.81, 0.01)
results = []

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )
    f2 = fbeta_score(
        y_val, val_pred, beta=2, zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2
    })

threshold_results = pd.DataFrame(results)

# F2 emphasizes recall more strongly than precision.
# Select the threshold using validation F2, NOT test results.
best_row = threshold_results.loc[
    threshold_results["f2"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nBest validation threshold:", round(best_threshold, 2))
print("Validation precision:", round(best_row["precision"], 4))
print("Validation recall:", round(best_row["recall"], 4))
print("Validation F1-score:", round(best_row["f1"], 4))
print("Validation F2-score:", round(best_row["f2"], 4))

# Show the 10 best thresholds by validation F2.
print("\nTop 10 thresholds on validation data:")
print(
    threshold_results.sort_values(
        "f2", ascending=False
    ).head(10).round(4).to_string(index=False)
)

# ------------------------------------------------------------
# 4. Final evaluation on the untouched test set
# ------------------------------------------------------------

# No fitting or threshold selection occurs on test data.
test_prob = model.predict_proba(X_test)[:, 1]

# Default threshold baseline
baseline_pred = (test_prob >= 0.50).astype(int)

# Recall-focused tuned threshold
tuned_pred = (test_prob >= best_threshold).astype(int)

def evaluate_model(y_true, y_pred, probabilities, title):
    print(f"\n{'=' * 55}")
    print(title)
    print("=" * 55)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F2-score:  {fbeta_score(y_true, y_pred, beta=2, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


evaluate_model(
    y_test_bin,
    baseline_pred,
    test_prob,
    "BASELINE: Class-weighted Logistic Regression (threshold 0.50)"
)

evaluate_model(
    y_test_bin,
    tuned_pred,
    test_prob,
    f"TUNED: Recall-focused Logistic Regression (threshold {best_threshold:.2f})"
)

# ------------------------------------------------------------
# 5. Compare baseline and tuned test performance
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "F2-score"],
    "Baseline": [
        accuracy_score(y_test_bin, baseline_pred),
        precision_score(y_test_bin, baseline_pred, zero_division=0),
        recall_score(y_test_bin, baseline_pred, zero_division=0),
        f1_score(y_test_bin, baseline_pred, zero_division=0),
        fbeta_score(y_test_bin, baseline_pred, beta=2, zero_division=0)
    ],
    "Tuned": [
        accuracy_score(y_test_bin, tuned_pred),
        precision_score(y_test_bin, tuned_pred, zero_division=0),
        recall_score(y_test_bin, tuned_pred, zero_division=0),
        f1_score(y_test_bin, tuned_pred, zero_division=0),
        fbeta_score(y_test_bin, tuned_pred, beta=2, zero_division=0)
    ]
})

print("\nFINAL PERFORMANCE COMPARISON")
print(comparison.round(4).to_string(index=False))



# Choose a minimum acceptable validation recall.
# Change this value based on your project's requirements.
minimum_recall = 0.85

eligible = threshold_results[
    threshold_results["recall"] >= minimum_recall
].copy()

if eligible.empty:
    print("No threshold meets the requested recall.")
else:
    # Among thresholds meeting the recall requirement,
    # choose the one with the highest validation precision.
    selected = eligible.sort_values(
        ["precision", "f1"],
        ascending=False
    ).iloc[0]

    print("Selected threshold:", round(selected["threshold"], 2))
    print("Validation precision:", round(selected["precision"], 4))
    print("Validation recall:", round(selected["recall"], 4))
    print("Validation F1-score:", round(selected["f1"], 4))
    print("Validation F2-score:", round(selected["f2"], 4))


# ============================================================
# FINAL CUSTOMER CHURN MODEL
# Logistic Regression + Recall-Focused Threshold
# ============================================================

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# STEP 1: Select threshold using validation data only
# ------------------------------------------------------------

minimum_recall = 0.85

eligible = threshold_results[
    threshold_results["recall"] >= minimum_recall
].copy()

if eligible.empty:
    raise ValueError(
        "No threshold achieves the required validation recall. "
        "Try a lower minimum_recall."
    )

selected = eligible.sort_values(
    ["precision", "f1"],
    ascending=False
).iloc[0]

best_threshold = float(selected["threshold"])

print("SELECTED VALIDATION THRESHOLD")
print("-" * 45)
print(f"Threshold: {best_threshold:.2f}")
print(f"Validation Precision: {selected['precision']:.4f}")
print(f"Validation Recall:    {selected['recall']:.4f}")
print(f"Validation F1-score:  {selected['f1']:.4f}")
print(f"Validation F2-score:  {selected['f2']:.4f}")

# ------------------------------------------------------------
# STEP 2: Create a fresh model
# ------------------------------------------------------------

final_model = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("logistic", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=42
    ))
])

# ------------------------------------------------------------
# STEP 3: Train on ALL original training data
# ------------------------------------------------------------

final_model.fit(X_train, y_train_bin)

print("\nFinal model trained on all training rows:", len(X_train))

# ------------------------------------------------------------
# STEP 4: Predict probabilities on untouched test data
# ------------------------------------------------------------

test_prob_final = final_model.predict_proba(X_test)[:, 1]

# Apply the threshold selected on validation data
test_pred_final = (
    test_prob_final >= best_threshold
).astype(int)

# ------------------------------------------------------------
# STEP 5: Evaluate final model
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("FINAL MODEL TEST RESULTS")
print("=" * 55)

print(f"Selected Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test_bin, test_pred_final):.4f}")
print(f"Precision: {precision_score(y_test_bin, test_pred_final, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test_bin, test_pred_final, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test_bin, test_pred_final, zero_division=0):.4f}")
print(f"F2-score:  {fbeta_score(y_test_bin, test_pred_final, beta=2, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test_bin, test_prob_final):.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test_bin,
    test_pred_final,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0
))

print("Confusion Matrix:")
cm = confusion_matrix(
    y_test_bin,
    test_pred_final,
    labels=[0, 1]
)
print(cm)

print("\nConfusion Matrix Interpretation:")
print("True Negatives :", cm[0, 0])
print("False Positives:", cm[0, 1])
print("False Negatives:", cm[1, 0])
print("True Positives :", cm[1, 1])


# ============================================================
# CUSTOMER CHURN PREDICTION
# Model Comparison + Graphical Visualization
# Dataset: Telco Customer Churn
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

RANDOM_STATE = 42
MINIMUM_RECALL = 0.85

sns.set_theme(style="whitegrid")

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = (
    "/kaggle/input/datasets/blastchar/"
    "telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

if not os.path.exists(DATA_PATH):
    # Search the Kaggle input directory if the exact path differs.
    import glob
    matches = glob.glob(
        "/kaggle/input/**/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        recursive=True
    )
    if not matches:
        raise FileNotFoundError(
            "Dataset CSV not found. Check the Kaggle dataset path."
        )
    DATA_PATH = matches[0]

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# 2. DATA CLEANING
# ============================================================

# Remove customerID because it is an identifier, not a useful
# predictive feature for generalizing to new customers.
df = df.drop(columns=["customerID"], errors="ignore")

# TotalCharges is often stored as text because some rows contain
# blank strings. Convert it safely to numeric.
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

# Encode target: No = 0, Yes = 1
df["Churn"] = (
    df["Churn"].astype(str).str.strip().str.lower()
    .map({"no": 0, "yes": 1})
)

if df["Churn"].isna().any():
    raise ValueError("Unexpected or missing values found in Churn.")

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# ============================================================
# 3. TARGET DISTRIBUTION VISUALIZATION
# ============================================================

class_counts = df["Churn"].map({
    0: "No Churn",
    1: "Churn"
}).value_counts().reindex(["No Churn", "Churn"])

plt.figure(figsize=(7, 5))
ax = sns.barplot(
    x=class_counts.index,
    y=class_counts.values
)
ax.bar_label(ax.containers[0])
plt.title("Customer Churn Distribution")
plt.xlabel("Customer Class")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

print("\nTarget distribution:")
print(class_counts)
print(
    "Churn percentage:",
    round(df["Churn"].mean() * 100, 2), "%"
)

# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

# First reserve the final test set.
# No model or threshold selection will use this test set.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Create a validation set from training data only.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("\nData split:")
print("Fitting rows:", len(X_fit))
print("Validation rows:", len(X_val))
print("Final test rows:", len(X_test))

# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_cols = X_fit.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X_fit.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_cols),
        ("categorical", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}

# ============================================================
# 7. TRAIN MODELS AND SELECT THRESHOLDS ON VALIDATION DATA
# ============================================================

threshold_grid = np.arange(0.10, 0.81, 0.01)

validation_results = []
validation_probabilities = {}
validation_threshold_tables = {}
fitted_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_fit, y_fit)
    fitted_models[name] = model

    probabilities = model.predict_proba(X_val)[:, 1]
    validation_probabilities[name] = probabilities

    threshold_rows = []

    for threshold in threshold_grid:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        threshold_rows.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "F2": fbeta_score(
                y_val, predictions, beta=2, zero_division=0
            )
        })

    threshold_df = pd.DataFrame(threshold_rows)
    validation_threshold_tables[name] = threshold_df

    # Prefer thresholds meeting the minimum recall target.
    eligible = threshold_df[
        threshold_df["Recall"] >= MINIMUM_RECALL
    ]

    if eligible.empty:
        # Fallback: highest recall, then highest F2.
        max_recall = threshold_df["Recall"].max()
        eligible = threshold_df[
            threshold_df["Recall"] == max_recall
        ]

    # Within eligible thresholds, maximize F2.
    best = eligible.sort_values(
        ["F2", "Precision"],
        ascending=False
    ).iloc[0]

    best_threshold = float(best["Threshold"])
    val_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    validation_results.append({
        "Model": name,
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(y_val, val_predictions),
        "Precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_val, val_predictions, zero_division=0
        ),
        "F2-score": fbeta_score(
            y_val, val_predictions, beta=2, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
        "PR-AUC": average_precision_score(y_val, probabilities)
    })

validation_table = pd.DataFrame(
    validation_results
).sort_values(
    ["F2-score", "Recall"],
    ascending=False
).reset_index(drop=True)

print("\nVALIDATION MODEL COMPARISON")
print(validation_table.round(4).to_string(index=False))

# ============================================================
# 8. GRAPH 1: MODEL PERFORMANCE COMPARISON
# ============================================================

metric_columns = [
    "Precision", "Recall", "F1-score", "F2-score", "ROC-AUC"
]

plot_data = validation_table.set_index("Model")[
    metric_columns
]

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
    width=0.8
)

plt.title("Model Performance Comparison (Validation Data)")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.xticks(rotation=15, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# 9. GRAPH 2: RECALL VS THRESHOLD
# ============================================================

plt.figure(figsize=(10, 6))

for name, probabilities in validation_probabilities.items():

    recalls = [
        recall_score(
            y_val,
            (probabilities >= threshold).astype(int),
            zero_division=0
        )
        for threshold in threshold_grid
    ]

    plt.plot(
        threshold_grid,
        recalls,
        label=name
    )

plt.axhline(
    MINIMUM_RECALL,
    linestyle="--",
    label="Minimum recall target"
)

plt.title("Recall vs Classification Threshold")
plt.xlabel("Classification Threshold")
plt.ylabel("Recall")
plt.ylim(0, 1.02)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 10. SELECT BEST MODEL USING VALIDATION F2
# ============================================================

best_model_name = validation_table.iloc[0]["Model"]
best_threshold = float(
    validation_table.iloc[0]["Threshold"]
)

print("\nSelected model:", best_model_name)
print("Selected threshold:", round(best_threshold, 2))

# Refit the selected model on all training data.
# Test data remains untouched until final evaluation.
final_model = clone(models[best_model_name])
final_model.fit(X_train, y_train)

# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL TEST RESULTS")
print("=" * 50)
print("Selected model:", best_model_name)
print("Threshold:", round(best_threshold, 2))

print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"F2-score:  {fbeta_score(y_test, test_predictions, beta=2, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probabilities):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, test_probabilities):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# ============================================================
# 12. GRAPH 3: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title(f"Final Test Confusion Matrix: {best_model_name}")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# ============================================================
# 13. GRAPH 4: ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    test_probabilities
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"{best_model_name} (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.title("ROC Curve - Final Test Data")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 14. GRAPH 5: PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    test_probabilities
)

average_precision = average_precision_score(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.axhline(
    y_test.mean(),
    linestyle="--",
    label="Churn prevalence baseline"
)

plt.title("Precision-Recall Curve - Final Test Data")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 15. GRAPH 6: FEATURE IMPORTANCE / COEFFICIENTS
# ============================================================

try:
    fitted_preprocessor = final_model.named_steps["preprocessor"]
    fitted_classifier = final_model.named_steps["classifier"]

    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    if hasattr(fitted_classifier, "coef_"):

        coefficients = fitted_classifier.coef_[0]

        feature_table = pd.DataFrame({
            "Feature": feature_names,
            "Value": coefficients
        })

        feature_table["Magnitude"] = (
            feature_table["Value"].abs()
        )

        top_features = feature_table.nlargest(
            20, "Magnitude"
        ).sort_values("Value")

        plt.figure(figsize=(10, 8))

        plt.barh(
            top_features["Feature"],
            top_features["Value"]
        )

        plt.axvline(0, linestyle="--")
        plt.title("Top 20 Logistic Regression Coefficients")
        plt.xlabel("Coefficient (positive = higher churn log-odds)")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    elif hasattr(fitted_classifier, "feature_importances_"):

        feature_importances = (
            fitted_classifier.feature_importances_
        )

        feature_table = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        }).nlargest(20, "Importance").sort_values("Importance")

        plt.figure(figsize=(10, 8))

        plt.barh(
            feature_table["Feature"],
            feature_table["Importance"]
        )

        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

except Exception as error:
    print("Feature importance graph unavailable:", error)

# ============================================================
# 16. SAVE VALIDATION RESULTS
# ============================================================

validation_table.to_csv(
    "churn_validation_model_comparison.csv",
    index=False
)

print("\nValidation comparison saved successfully.")
print("All visualizations generated.")
