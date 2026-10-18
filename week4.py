import joblib
import os

# Create output directory
os.makedirs("/kaggle/working", exist_ok=True)

# Use the optimized threshold from Week 3
if "best_threshold" not in globals():
    best_threshold = 0.50

# Save model and its required information
model_artifacts = {
    "model": best_model,
    "threshold": best_threshold,
    "feature_columns": list(X_train.columns)
}

joblib.dump(
    model_artifacts,
    "/kaggle/working/churn_model.pkl"
)

print("Model saved successfully!")
print("File: /kaggle/working/churn_model.pkl")
print("Threshold:", best_threshold)
print("Number of input features:", len(X_train.columns))




import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="/kaggle/working/.virtual_documents/__notebook_source__.ipynb",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# 2. Load trained model
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "churn_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    artifacts = load_model()
    model = artifacts["model"]
    threshold = artifacts["threshold"]
    feature_columns = artifacts["feature_columns"]
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -----------------------------
# 3. Application heading
# -----------------------------
st.title("📊 Churn Risk Advisor")
st.subheader("Customer Churn Prediction System")

st.write(
    "Enter customer details below to estimate the probability "
    "that the customer will leave the service."
)

st.info(
    f"Prediction threshold selected in Week 3: {threshold:.2f}"
)

# -----------------------------
# 4. Customer input form
# -----------------------------
with st.form("customer_form"):

    st.header("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        use_container_width=True
    )

# -----------------------------
# 5. Make prediction
# -----------------------------
if submitted:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Match the exact training feature names and order
    input_df = pd.DataFrame([customer_data])
    input_df = input_df.reindex(columns=feature_columns)

    # Predict probability for Churn = 1
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    churn_index = classes.index(1)
    churn_probability = probabilities[churn_index]

    # Apply the threshold selected in Week 3
    prediction = int(churn_probability >= threshold)

    st.divider()
    st.header("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

        st.progress(float(churn_probability))

    with result_col2:
        if prediction == 1:
            st.error("⚠️ Prediction: Customer is at risk of churning")
        else:
            st.success("✅ Prediction: Customer is not flagged for churn")

    # Risk category for easier interpretation
    if churn_probability >= 0.60:
        risk_level = "High"
        st.warning("High-risk customer: consider proactive retention.")
    elif churn_probability >= 0.30:
        risk_level = "Medium"
        st.info("Medium-risk customer: monitor engagement.")
    else:
        risk_level = "Low"
        st.success("Low-risk customer: maintain service quality.")

    st.subheader("Recommended Action")

    if risk_level == "High":
        st.write(
            "- Contact the customer to understand concerns.\n"
            "- Review contract and pricing options.\n"
            "- Offer suitable support or retention incentives."
        )
    elif risk_level == "Medium":
        st.write(
            "- Monitor customer satisfaction.\n"
            "- Explain available plans and support options.\n"
            "- Review service usage and engagement."
        )
    else:
        st.write(
            "- Continue providing reliable service.\n"
            "- Maintain customer communication.\n"
            "- Review risk again when new data becomes available."
        )

    st.caption(
        "This is a machine-learning estimate, not a guarantee "
        "of future customer behavior. Risk bands are illustrative."
    )



import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "/kaggle/working/churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

artifacts = load_model()

model = artifacts["model"]
threshold = artifacts["threshold"]
feature_columns = artifacts["feature_columns"]

print("Model loaded successfully!")
print("Threshold:", threshold)
print("Features:", len(feature_columns))



import os

print("Files in /kaggle/working:")
print(os.listdir("/kaggle/working"))

model_path = "/kaggle/working/churn_model.pkl"

if os.path.exists(model_path):
    print("Model file found!")
else:
    print("Model file not found. Save the Week 3 model first.")


import joblib

model_path = "/kaggle/working/churn_model.pkl"

artifacts = joblib.load(model_path)

model = artifacts["model"]
threshold = artifacts["threshold"]
feature_columns = artifacts["feature_columns"]

print("Model loaded successfully!")
print("Threshold:", threshold)
print("Number of features:", len(feature_columns))
print("Feature names:", feature_columns)



import pandas as pd

# Use one customer record from the test dataset
sample = X_test.iloc[[0]].copy()

probabilities = model.predict_proba(sample)[0]
classes = list(model.classes_)

churn_index = classes.index(1)
churn_probability = probabilities[churn_index]

prediction = int(churn_probability >= threshold)

print("Churn Probability:", round(churn_probability * 100, 2), "%")
print("Prediction:", "Churn" if prediction == 1 else "No Churn")
print("Threshold:", threshold)
