# Credit Risk Intelligence: Loan Default Dashboard and Predictor

An end-to-end credit risk project: a two-page Power BI dashboard, a machine learning model that predicts loan default, and a live web app where anyone can try the model.

**Live predictor:** [credit-risk-predictor-asdf.streamlit.app](https://credit-risk-predictor-asdf.streamlit.app)

> **Data note:** the data is a public Kaggle loan dataset (nikhil1e9, "Loan Default Prediction Dataset"), not data from any real bank. The dashboard uses an SBI-inspired blue theme as a design choice.

## Business problem
Banks lose money when borrowers default. This project answers two questions:
1. **Who defaults?** Which borrower and loan characteristics go with higher default rates?
2. **Can we predict it?** Can a model rank loans by risk so that the riskiest applications get a manual review?

## Dataset
- 255,347 loans, 18 columns (age, income, loan amount, credit score, months employed, interest rate, DTI ratio, loan purpose, and more)
- Target: `Default` (1 = defaulted, 0 = repaid)
- Overall default rate: **11.61%**
- No missing values and no duplicate loan IDs

## Data preparation (Power Query)
- Checked column quality and duplicates on the full dataset
- Created band columns for segmentation: Age Group, Credit Band, Income Band, DTI Band, Interest Rate Band, Loan Amount Band, plus a readable Default Status
- Built DAX measures for total loans, defaults, default rate, average credit score, and the model metrics

## Dashboard (Power BI)

### Page 1: Risk Dashboard
![Risk Dashboard](.Image/Risk%20Dashboard.png)

Key findings:
- **Age is the biggest driver.** Borrowers aged 18-25 default at 20.8%, against 5.5% for those 56+.
- **Pricing tracks risk.** Very high interest loans default at 17.8%, against 6.4% for very low rates.
- **Credit score helps, but less.** Poor credit defaults at 13.0%, Excellent at 10.2%.
- **Riskiest segments:** High DTI (12.2%) and Business loans (12.3%). Home loans are the safest at 10.2%.

### Page 2: ML Predictions
![ML Predictions](.Image/ML%20Dashboard.png)

This page shows the model's metrics, the top default drivers, a chart proving the risk tiers work on unseen loans, and a table of the 10 highest-risk borrowers.

## Machine learning model
- **Model:** Random Forest classifier (200 trees), compared against a Logistic Regression baseline. The data was split 80/20 with stratification, and class weights were balanced because defaults are rare.
- **Top drivers:** age, interest rate, income, months employed, loan amount.

| Model | ROC-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.753 | 0.220 | 0.699 | 0.334 | 0.677 |
| Random Forest | 0.756 | 0.248 | 0.633 | 0.356 | 0.734 |

### Risk tiers
Loans are ranked by predicted default probability: the top 20% is **High**, the next 30% is **Medium**, and the bottom 50% is **Low**. On loans the model never saw, the actual default rates were:

| Tier | Actual default rate |
|---|---|
| High | 29.0% |
| Medium | 11.9% |
| Low | 4.4% |

The High tier defaults about 6.6 times as often as the Low tier.

### Why recall over precision
A missed defaulter costs the bank more than reviewing a good application, so the model is tuned to catch defaulters (recall 63%). Precision is 25%, about twice the 11.6% base rate. The High tier is meant as a manual-review list, not an automatic decline.

### Threshold analysis
| Threshold | Precision | Recall | F1 | Share of loans flagged |
|---|---|---|---|---|
| 0.40 | 18.7% | 80.9% | 0.304 | 50.2% |
| 0.45 | 21.4% | 72.7% | 0.331 | 39.4% |
| **0.50 (used)** | **24.7%** | **63.3%** | **0.356** | **29.7%** |
| 0.55 | 28.5% | 52.3% | 0.369 | 21.3% |
| 0.60 | 32.7% | 40.3% | 0.361 | 14.3% |
| 0.65 | 38.9% | 28.2% | 0.327 | 8.4% |
| 0.70 | 47.5% | 17.2% | 0.253 | 4.2% |

A threshold of 0.50 balances catching defaulters with a manageable review workload. A higher threshold raises precision but misses more defaulters.

## Live predictor (Streamlit)
The web app takes the eight strongest drivers as inputs (age, income, loan amount, credit score, months employed, number of credit lines, interest rate, and debt-to-income ratio) and returns a default probability and a risk tier. Other inputs are held at baseline values. The tier cutoffs are the same as in the dashboard.

## How to open the dashboard
Download `Credit_Risk_Intelligence.pbix` and open it with the free [Power BI Desktop](https://powerbi.microsoft.com/desktop/). The screenshots above show both pages.

## Tools
Python (pandas, scikit-learn), Power BI (Power Query, DAX), Streamlit, GitHub.

## Limitations
- The data is a public dataset with no date column, so the project can't show trends over time.
- Interest rate partly reflects how a lender prices risk, so its link with default is a pattern, not proof of cause.
- This is a portfolio project, not a real credit decision tool.

## Repository structure
- `.Notebook/ml.ipynb`: model training, evaluation, and threshold analysis
- `.Image/`: dashboard screenshots
- `Credit_Risk_Intelligence.pbix`: Power BI dashboard
- `app.py`, `loan_model.joblib`, `requirements.txt`, `.streamlit/`: the live predictor
