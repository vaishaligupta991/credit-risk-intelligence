import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦")

@st.cache_resource
def load_bundle():
    return joblib.load("loan_model.joblib")

bundle = load_bundle()
model = bundle["model"]
columns = bundle["columns"]
cut_medium = bundle["cut_medium"]
cut_high = bundle["cut_high"]

st.title("Loan Default Predictor")
st.caption("Random Forest trained on a public Kaggle loan dataset. "
           "A portfolio demo, not a real credit decision.")

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 18, 69, 35)
    income = st.number_input("Annual income", 15000, 149999, 70000, step=1000)
    loan_amount = st.number_input("Loan amount", 5000, 249999, 120000, step=1000)
    credit_score = st.slider("Credit score", 300, 849, 600)
    months_employed = st.slider("Months employed", 0, 119, 48)
    num_credit_lines = st.slider("Number of credit lines", 1, 4, 2)
    interest_rate = st.slider("Interest rate (%)", 2.0, 25.0, 12.0, step=0.1)
    loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
    dti = st.slider("Debt-to-income ratio", 0.10, 0.90, 0.40, step=0.01)
with c2:
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment = st.selectbox("Employment type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
    mortgage = st.selectbox("Has mortgage", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["Yes", "No"])
    purpose = st.selectbox("Loan purpose", ["Home", "Auto", "Education", "Business", "Other"])
    cosigner = st.selectbox("Has co-signer", ["Yes", "No"])

if st.button("Predict default risk", type="primary"):
    row = pd.DataFrame([{
        "Age": age, "Income": income, "LoanAmount": loan_amount,
        "CreditScore": credit_score, "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti, "Education": education,
        "EmploymentType": employment, "MaritalStatus": marital,
        "HasMortgage": mortgage, "HasDependents": dependents,
        "LoanPurpose": purpose, "HasCoSigner": cosigner,
    }])
    X = pd.get_dummies(row).reindex(columns=columns, fill_value=0).astype(float)
    prob = float(model.predict_proba(X)[0, 1])

    if prob >= cut_high:
        tier = "High"
    elif prob >= cut_medium:
        tier = "Medium"
    else:
        tier = "Low"

    st.metric("Probability of default", f"{prob:.1%}")
    if tier == "High":
        st.error("Risk tier: High. Send for manual review.")
    elif tier == "Medium":
        st.warning("Risk tier: Medium.")
    else:
        st.success("Risk tier: Low.")
    st.progress(min(prob, 1.0))
    st.caption("Tiers use the same cutoffs as the Power BI dashboard: "
               "the top 20% of predicted probabilities is High, the next 30% is Medium.")
