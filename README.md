# LoanLens AI — Loan Approval Prediction System

LoanLens AI is an end-to-end Machine Learning and Analytics solution designed to assess credit risk and predict whether a loan application is likely to be **Approved** or **Rejected**. By processing key applicant and financial attributes, LoanLens AI delivers real-time predictions, model diagnostics, feature reliance insights, and automated PDF assessment reports through an interactive Streamlit web application.

---

## 1. Project Overview

In credit evaluation, financial institutions require fast, accurate, and consistent risk assessment tools to process loan applications efficiently. LoanLens AI automates this workflow by evaluating applicant profile data through a trained machine learning pipeline. The project encompasses complete data ingestion, exploratory data analysis, feature selection, machine learning model comparison, pipeline serialization, and web application deployment.

---

## 2. Problem Statement

Manual credit underwriting can be slow, subjective, and prone to operational bottlenecks. Financial institutions face the challenge of evaluating large volumes of loan applications while maintaining rigorous risk management standards. An automated machine learning prediction system provides standard risk scoring, speeds up application turnaround times, and provides objective decision-support metrics for underwriting teams.

---

## 3. Objectives

- **Automated Data Processing:** Establish a scalable data processing pipeline handling key financial variables.
- **Exploratory Data Insights:** Analyze key financial indicators and their relationships with loan approval outcomes.
- **Feature Selection:** Simplify the model from 25 initial dataset attributes down to 7 core predictive features based on feature-importance analysis and retraining.
- **Model Benchmarking:** Train and benchmark classification algorithms (**Logistic Regression** vs. **Random Forest Classifier**) using comprehensive performance metrics.
- **Pipeline Serialization:** Package preprocessing transformers (`StandardScaler`) and the trained classifier into a single deployable scikit-learn Pipeline via `joblib`.
- **Interactive UI Deployment:** Deliver a Streamlit application (`app.py`) for real-time predictions, interactive input management, performance visualization, and downloadable PDF risk reports.

---

## 4. Dataset

The project utilizes a comprehensive dataset structured specifically for credit decision analytics:

- **Total Records:** 52,000 applicant records
- **Total Attributes:** 27 columns in raw dataset
- **Model Features:** 7 selected input features
- **Identifier Column:** `Applicant_ID` (unique applicant reference, excluded from model training)
- **Target Variable:** `Loan_Approval_Status`
  - `1` = Approved
  - `0` = Rejected

---

## 5. Selected Model Features

After feature importance analysis and retraining, the deployed model was simplified to focus on seven core input features:

1. `Credit_Score`: Credit score rating (300 to 849)
2. `Loan_Amount_Requested`: Requested loan principal amount (₹)
3. `Annual_Income`: Gross annual income (₹)
4. `Age`: Applicant age in years
5. `Interest_Rate`: Interest rate applied to loan (%)
6. `Outstanding_Debt`: Total remaining debt balance (₹)
7. `Monthly_Expenses`: Estimated monthly living expenses (₹)

---

## 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed in `python/02_eda_visualization.ipynb` and focuses on core visualizations:

1. **Loan Approval Distribution:** Evaluates target balance across approved and rejected loan applications.
2. **Credit Score vs Loan Approval:** Analyzes credit score distributions and threshold behaviors for approved versus rejected applicants.
3. **Annual Income vs Loan Approval:** Examines income levels and capacity to service debt against approval outcomes.
4. **Loan Amount Requested vs Loan Approval:** Inspects requested loan amounts to assess capital risk exposure.
5. **Employment Status vs Loan Approval:** Compares approval proportions across employment categories.

---

## 7. Machine Learning Approach

The machine learning workflow evaluates two classification algorithms:

- **Logistic Regression:** Serves as a baseline linear classification model.
- **Random Forest Classifier:** Ensembles multiple decision trees to capture non-linear interactions between financial features.

Model evaluation uses a stratified 80/20 train-test split (`test_size=0.20`, `random_state=42`, `stratify=y`). The complete preprocessing and modeling workflow is combined into a scikit-learn `Pipeline` to ensure data leakage prevention and simplified production deployment.

---

## 8. Data Preprocessing

Data preprocessing is structured using scikit-learn's `ColumnTransformer`:

- **Numerical Features (7):** Transformed using `StandardScaler` to ensure zero mean and unit variance.

### Pipeline Architecture:
$$\text{7 Selected Features} \longrightarrow \text{ColumnTransformer (StandardScaler)} \longrightarrow \text{Classifier}$$

The trained pipeline is serialized using `joblib` into `models/loan_approval_random_forest_7features.pkl`.

---

## 9. Model Training & Project Workflow

Model training and evaluation are conducted in `python/03_ML.ipynb`.

### Project Workflow:
$$\text{Loan Dataset} \longrightarrow \text{Data Preprocessing} \longrightarrow \text{EDA} \longrightarrow \text{Feature Selection} \longrightarrow \text{Train/Test Split} \longrightarrow \text{Logistic Regression \& Random Forest} \longrightarrow \text{Model Evaluation} \longrightarrow \text{Random Forest Selection} \longrightarrow \text{New Applicant Prediction}$$

1. Data loading and validation (`dataset/Loan_Dataset.csv`).
2. Feature selection (retaining 7 core numerical input features).
3. Stratified train-test split (41,600 training samples, 10,400 testing samples).
4. Pipeline fitting and metric evaluation across both algorithms.
5. Serialization of the final production pipeline (`models/loan_approval_random_forest_7features.pkl`).

---

## 10. Model Evaluation

Model evaluation on the independent test set (10,400 records) yielded the following 7-feature results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 85.04% | 85.08% |
| Precision | 85.00% | 85.07% |
| Recall | 93.11% | 93.08% |
| F1 Score | 88.87% | 88.89% |
| ROC-AUC | 81.48% | 81.88% |

**Model Selection Analysis:**
Random Forest was selected as the final deployed model because it achieved slightly higher Accuracy (85.08%), Precision (85.07%), F1-Score (88.89%), and ROC-AUC (81.88%) on the test set, while Logistic Regression achieved slightly higher Recall (93.11% vs. 93.08%).

---

## 11. Confusion Matrix

The confusion matrix for the final Random Forest model on the test set (10,400 samples) is:

```
                Predicted
              0        1
Actual 0     2637     1090
Actual 1      462     6211
```

### Breakdown:
- **True Negatives (TN = 2,637):** Correctly predicted Rejections.
- **False Positives (FP = 1,090):** Incorrectly predicted Approvals (Actual Rejections).
- **False Negatives (FN = 462):** Incorrectly predicted Rejections (Actual Approvals).
- **True Positives (TP = 6,211):** Correctly predicted Approvals.

*(Rows represent actual ground-truth classes and columns represent predicted model classes).*

---

## 12. Feature Importance

Feature importance extracted from the trained 7-feature Random Forest model:

1. **Credit Score** — 28.92%
2. **Loan Amount Requested** — 22.42%
3. **Annual Income** — 14.97%
4. **Outstanding Debt** — 8.83%
5. **Monthly Expenses** — 8.74%
6. **Interest Rate** — 8.55%
7. **Age** — 7.58%

Credit Score was the most influential feature in the Random Forest model, followed by Loan Amount Requested and Annual Income. Feature importance represents the model's reliance on these variables and does not imply that a feature directly causes loan approval.

---

## 13. Streamlit Application

The interactive web application (`app.py`) provides an intuitive dashboard for credit analysts and underwriters:

- **Model Execution:** Loads the serialized Random Forest pipeline directly from `models/loan_approval_random_forest_7features.pkl`.
- **Interactive Inputs:** Enables user input for the 7 selected applicant and loan attributes.
- **Real-Time Prediction:** Displays instant **Approved** or **Rejected** decisions alongside calculated approval probability.
- **Model Diagnostics:** Presents model accuracy metrics, 7-feature importances, and confusion matrix tables.
- **Automated PDF Export:** Generates downloadable assessment reports using ReportLab for official recordkeeping.

---

## 14. Project Structure

```
Loan-Credit-Analytics/
├── python/
│   ├── 01_loadData.ipynb
│   ├── 02_eda_visualization.ipynb
│   └── 03_ML.ipynb
├── dataset/
│   └── Loan_Dataset.csv
├── models/
│   └── loan_approval_random_forest_7features.pkl
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

> **Note:** The `dataset/` directory and trained model files in `models/` are excluded from Git repository tracking via `.gitignore`.

---

## 15. Installation

Set up a virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 16. Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Access the web interface at `http://localhost:8501`.

---

## 17. Example Prediction

Below is a sample applicant profile and the resulting prediction from the 7-feature model:

### Sample Input:
- **Credit Score:** 750
- **Loan Amount Requested:** ₹25,000
- **Annual Income:** ₹90,000
- **Age:** 35
- **Interest Rate:** 7.5%
- **Outstanding Debt:** ₹10,000
- **Monthly Expenses:** ₹3,000

### Model Output:
- **Prediction:** Approved
- **Approval Probability:** 81.5%

*(Note: This sample prediction is provided strictly as a documented illustration in the README and is not hardcoded into the application's prediction logic.)*

---

## 18. Limitations

- **Decision-Support Prototype:** This is a decision-support prototype and does not replace regulatory checks, bank policies, or human review.
- **Training Data Scope:** Model predictions are constrained by the statistical distribution of the training dataset.
- **Correlation vs. Causation:** Feature importance represents the model's reliance on these variables and does not imply that a feature directly causes loan approval.

---

## 19. Future Enhancements

- **Model Interpretability:** Incorporate feature explanation views into the UI.
- **API Microservice:** Deploy REST API endpoints (e.g., FastAPI) for integration with banking systems.
- **Monitoring & Retraining:** Establish continuous model monitoring for data drift and automated retraining pipelines.
- **Expanded Risk Metrics:** Integrate macroeconomic indicators to refine credit risk assessments.