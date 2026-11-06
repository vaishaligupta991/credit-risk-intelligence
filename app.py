import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg, #0B2A5B 0%, #1F5FBF 100%); padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #FFFFFF; padding: 0;}
.hero p {margin: 8px 0 0 0; color: #CFE0FA; font-size: 1.05rem;}
.card {background: #FFFFFF; border: 1px solid #D9E2F3; border-radius: 16px; padding: 28px; color: #0B2A5B;}
.label {font-size: 0.95rem; color: #5B6B8C; margin-bottom: 4px;}
.big {font-size: 3.6rem; font-weight: 700; line-height: 1.05;}
.badge {display: inline-block; padding: 6px 18px; border-radius: 999px; color: #FFFFFF; font-weight: 600; margin-top: 12px;}
.bar {position: relative; height: 14px; border-radius: 7px; margin-top: 34px;}
.marker {position: absolute; top: -7px; width: 4px; height: 28px; background: #0B2A5B; border-radius: 2px;}
.scale {display: flex; justify-content: space-between; font-size: 0.8rem; color: #5B6B8C; margin-top: 10px;}
.note {font-size: 0.85rem; color: #5B6B8C; margin-top: 18px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return joblib.load("loan_model.joblib")


bundle = load_bundle()
model = bundle["model"]
columns = bundle["columns"]
cut_medium = bundle["cut_medium"]
cut_high = bundle["cut_high"]

st.markdown(
    '<div class="hero"><h1>🏦 Loan Default Predictor</h1>'
    "<p>Adjust the borrower profile and watch the default risk update instantly.</p></div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.subheader("Borrower details")
    a, b = st.columns(2)
    with a:
        age = st.slider("Age", 18, 69, 35)
        income = st.slider("Annual income", 15000, 149999, 70000, step=1000)
        credit_score = st.slider("Credit score", 300, 849, 600)
        months_employed = st.slider("Months employed", 0, 119, 48)
    with b:
        interest_rate = st.slider("Interest rate (%)", 2.0, 25.0, 12.0, step=0.1)
        loan_amount = st.slider("Loan amount", 5000, 249999, 120000, step=1000)
        dti = st.slider("Debt-to-income ratio", 0.10, 0.90, 0.40, step=0.01)
        credit_lines = st.slider("Number of credit lines", 1, 4, 2)

row = pd.DataFrame([{
    "Age": age, "Income": income, "LoanAmount": loan_amount,
    "CreditScore": credit_score, "MonthsEmployed": months_employed,
    "NumCreditLines": credit_lines, "InterestRate": interest_rate,
    "DTIRatio": dti, "LoanTerm": 36,
}])
X = row.reindex(columns=columns, fill_value=0).astype(float)
prob = float(model.predict_proba(X)[0, 1])

if prob >= cut_high:
    tier, color, msg = "High risk", "#D64545", "Flag for manual review."
elif prob >= cut_medium:
    tier, color, msg = "Medium risk", "#E8A33D", "Moderate chance of default."
else:
    tier, color, msg = "Low risk", "#1F5FBF", "Lower chance of default."

mid_pct = cut_medium * 100
high_pct = cut_high * 100
marker = min(max(prob * 100, 0), 100)
gradient = (
    f"linear-gradient(90deg, #1F5FBF 0%, #1F5FBF {mid_pct:.1f}%, "
    f"#E8A33D {mid_pct:.1f}%, #E8A33D {high_pct:.1f}%, "
    f"#D64545 {high_pct:.1f}%, #D64545 100%)"
)

result = (
    '<div class="card">'
    '<div class="label">Probability of default</div>'
    f'<div class="big" style="color:{color};">{prob:.1%}</div>'
    f'<div class="badge" style="background:{color};">{tier}</div>'
    f'<div style="margin-top:10px;">{msg}</div>'
    f'<div class="bar" style="background:{gradient};">'
    f'<div class="marker" style="left:calc({marker:.1f}% - 2px);"></div></div>'
    '<div class="scale"><span>Low</span><span>Medium</span><span>High</span></div>'
    '<div class="note">Based on the eight strongest drivers in the model. '
    "Risk tiers match the Power BI dashboard.</div>"
    "</div>"
)

with right:
    st.subheader("Prediction")
    st.markdown(result, unsafe_allow_html=True)
