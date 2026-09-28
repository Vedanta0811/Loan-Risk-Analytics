# LoanLens AI — Loan Approval Prediction System

LoanLens AI is an end-to-end Machine Learning and Analytics solution designed to assess credit risk and predict whether a loan application is likely to be **Approved** or **Rejected**. By processing demographic, financial, and historical credit risk attributes, LoanLens AI delivers real-time predictions, model diagnostics, feature reliance insights, and automated PDF assessment reports through an interactive Streamlit web application.

---

## 1. Project Overview

In credit evaluation, financial institutions require fast, accurate, and consistent risk assessment tools to process loan applications efficiently. LoanLens AI automates this workflow by evaluating applicant profile data through a trained machine learning pipeline. The project encompasses complete data ingestion, exploratory data analysis, machine learning model comparison, pipeline serialization, and web application deployment.

---

## 2. Problem Statement

Manual credit underwriting can be slow, subjective, and prone to operational bottlenecks. Financial institutions face the challenge of evaluating large volumes of loan applications while maintaining rigorous risk management standards. An automated machine learning prediction system provides standard risk scoring, speeds up application turnaround times, and provides objective decision-support metrics for underwriting teams.

---

## 3. Objectives

- **Automated Data Processing:** Establish a scalable data processing pipeline handling both numerical and categorical financial variables.
- **Exploratory Data Insights:** Analyze key financial indicators and their relationships with loan approval outcomes.
- **Model Benchmarking:** Train and benchmark classification algorithms (**Logistic Regression** vs. **Random Forest Classifier**) using comprehensive performance metrics.
- **Pipeline Serialization:** Package preprocessing transformers (`StandardScaler`, `OneHotEncoder`) and the trained classifier into a single deployable scikit-learn Pipeline via `joblib`.
- **Interactive UI Deployment:** Deliver a Streamlit application (`app.py`) for real-time predictions, interactive input management, performance visualization, and downloadable PDF risk reports.

---

## 4. Dataset

The project utilizes a comprehensive dataset structured specifically for credit decision analytics:

- **Total Records:** 52,000 applicant records
- **Total Columns:** 27 attributes
- **Model Features:** 25 input variables
- **Identifier Column:** `Applicant_ID` (unique applicant reference, excluded from model training)
- **Target Variable:** `Loan_Approval_Status`
  - `1` = Approved
  - `0` = Rejected

---

## 5. Dataset Features

The model processes 25 input features divided into 14 numerical attributes and 11 categorical attributes:

### Numerical Features (14)
1. `Age`: Applicant age in years
2. `Dependents`: Number of financial dependents
3. `Annual_Income`: Gross annual income ($)
4. `Monthly_Expenses`: Estimated monthly living expenses ($)
5. `Credit_Score`: Credit score rating
6. `Existing_Loans`: Count of active existing loans
7. `Total_Existing_Loan_Amount`: Total principal amount of active loans ($)
8. `Outstanding_Debt`: Total remaining debt balance ($)
9. `Loan_Amount_Requested`: Requested loan principal amount ($)
10. `Loan_Term`: Requested loan term duration (months)
11. `Interest_Rate`: Interest rate applied to loan (%)
12. `Bank_Account_History`: Bank account relationship duration (years)
13. `Transaction_Frequency`: Monthly bank transaction frequency
14. `Default_Risk`: Estimated default probability risk score (0.0 to 1.0)

### Categorical Features (11)
1. `Gender`: Male / Female
2. `Marital_Status`: Married / Single / Divorced
3. `Education`: Graduate / High School / Master / PhD
4. `Employment_Status`: Employed / Self-Employed / Unemployed
5. `Occupation_Type`: Salaried / Business / Professional
6. `Residential_Status`: Own / Rent / Mortgaged
7. `City/Town`: Urban / Semi-Urban / Rural
8. `Loan_History`: Previous loan repayment history status (1 = Clean / 0 = Defaulted)
9. `Loan_Purpose`: Home / Education / Personal / Auto / Business
10. `Loan_Type`: Secured / Unsecured
11. `Co-Applicant`: Co-applicant presence (Yes / No)

---

## 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed in `python/02_eda_visualization.ipynb` and focuses on five core visualizations:

1. **Loan Approval Distribution:** Evaluates target balance across approved and rejected loan applications.
2. **Credit Score vs Loan Approval:** Analyzes credit score distributions and threshold behaviors for approved versus rejected applicants.
3. **Annual Income vs Loan Approval:** Examines income levels and capacity to service debt against approval outcomes.
4. **Loan Amount Requested vs Loan Approval:** Inspects requested loan amounts to assess capital risk exposure.
5. **Employment Status vs Loan Approval:** Compares approval proportions across employment categories (Employed, Self-Employed, Unemployed).

---

## 7. Machine Learning Approach

The machine learning workflow evaluates two classification algorithms:

- **Logistic Regression:** Serves as a baseline linear classification model.
- **Random Forest Classifier:** Ensembles multiple decision trees to capture non-linear interactions between financial features.

Model evaluation uses a stratified 80/20 train-test split (`test_size=0.20`, `random_state=42`, `stratify=y`). The complete preprocessing and modeling workflow is combined into a scikit-learn `Pipeline` to ensure data leakage prevention and simplified production deployment.

---

## 8. Data Preprocessing

Data preprocessing is structured using scikit-learn's `ColumnTransformer`:

- **Numerical Features (14):** Transformed using `StandardScaler` to ensure zero mean and unit variance.
- **Categorical Features (11):** Encoded using `OneHotEncoder(handle_unknown="ignore")` to handle unseen categorical levels safely during inference.

### Pipeline Architecture:
$$\text{Raw Features} \longrightarrow \text{ColumnTransformer (StandardScaler + OneHotEncoder)} \longrightarrow \text{Classifier}$$

The trained pipeline is serialized using `joblib` into `models/loan_approval_random_forest.pkl`.

---

## 9. Model Training

Model training is conducted in `python/03_ML.ipynb`:

1. Data loading and validation (`dataset/Loan_Dataset.csv`).
2. Feature matrix separation (`X` containing 25 features, `y` containing target `Loan_Approval_Status`).
3. Stratified train-test split (41,600 training samples, 10,400 testing samples).
4. Pipeline fitting and metric evaluation across both algorithms.
5. Serialization of the final production pipeline (`models/loan_approval_random_forest.pkl`).

---

## 10. Model Evaluation

Model evaluation on the independent test set (10,400 records) yielded the following metrics:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 85.07% | 85.11% |
| Precision | 85.03% | 85.10% |
| Recall | 93.12% | 93.09% |
| F1 Score | 88.89% | 88.91% |
| ROC-AUC | 81.52% | 82.03% |

**Model Selection Analysis:**
Random Forest was selected as the final deployed model because it achieved slightly better Accuracy (85.11%), Precision (85.10%), F1 Score (88.91%), and ROC-AUC (82.03%) on the test set, while Logistic Regression maintained marginally higher Recall (93.12% vs. 93.09%).

---

## 11. Confusion Matrix

The confusion matrix for the final Random Forest model on the test set (10,400 samples) is:

```
                Predicted
              0        1
Actual 0     2639     1088
Actual 1      461     6212
```

### Breakdown:
- **True Negatives (TN = 2,639):** Correctly predicted Rejections.
- **False Positives (FP = 1,088):** Incorrectly predicted Approvals (Actual Rejections).
- **False Negatives (FN = 461):** Incorrectly predicted Rejections (Actual Approvals).
- **True Positives (TP = 6,212):** Correctly predicted Approvals.

*(Rows represent actual ground-truth classes and columns represent predicted model classes).*

---

## 12. Feature Importance

Feature importance extracted from the trained Random Forest model highlights the top 10 contributing factors:

1. **Credit Score** — 21.18%
2. **Loan Amount Requested** — 18.31%
3. **Annual Income** — 11.03%
4. **Age** — 6.95%
5. **Interest Rate** — 3.82%
6. **Outstanding Debt** — 3.80%
7. **Monthly Expenses** — 3.77%
8. **Total Existing Loan Amount** — 3.75%
9. **Loan Term** — 3.54%
10. **Default Risk** — 3.36%

Credit Score was the most influential feature in the Random Forest model, followed by Loan Amount Requested and Annual Income. Feature importance represents the model's reliance on these variables and does not imply causation.

---

## 13. Streamlit Application

The interactive web application (`app.py`) provides an intuitive dashboard for credit analysts and underwriters:

- **Model Execution:** Loads the serialized Random Forest pipeline directly from `models/loan_approval_random_forest.pkl`.
- **Interactive Inputs:** Enables user input for all 25 applicant and loan attributes.
- **Real-Time Prediction:** Displays instant **Approved** or **Rejected** decisions alongside calculated approval probability.
- **Model Diagnostics:** Presents model accuracy metrics, top feature importances, and confusion matrix tables.
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
│   └── loan_approval_random_forest.pkl
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

Below is a sample applicant profile and the resulting prediction:

### Sample Input:
- **Gender:** Male
- **Age:** 35
- **Marital_Status:** Married
- **Dependents:** 2
- **Education:** Graduate
- **Employment_Status:** Employed
- **Occupation_Type:** Salaried
- **Residential_Status:** Own
- **City/Town:** Urban
- **Annual_Income:** 90,000
- **Monthly_Expenses:** 3,000
- **Credit_Score:** 750
- **Existing_Loans:** 1
- **Total_Existing_Loan_Amount:** 20,000
- **Outstanding_Debt:** 10,000
- **Loan_History:** 1
- **Loan_Amount_Requested:** 25,000
- **Loan_Term:** 120
- **Loan_Purpose:** Home
- **Interest_Rate:** 7.5
- **Loan_Type:** Secured
- **Co-Applicant:** Yes
- **Bank_Account_History:** 7
- **Transaction_Frequency:** 20
- **Default_Risk:** 0.15

### Model Output:
- **Prediction:** Approved
- **Approval Probability:** Approximately 93%

*(Note: This sample prediction is provided strictly as a documented illustration in the README and is not hardcoded into the application's prediction logic.)*

---

## 18. Limitations

- **Decision-Support Prototype:** LoanLens AI is intended as a machine-learning decision-support tool and does not replace formal bank policy, regulatory compliance checks, legal verification, or manual underwriting review.
- **Training Data Scope:** Model predictions are constrained by the statistical distribution of the training dataset.
- **Correlation vs. Causation:** Feature importance reflects model statistical reliance during training and does not establish direct real-world causation.

---

## 19. Future Enhancements

- **Model Interpretability:** Incorporate feature explanation views into the UI.
- **API Microservice:** Deploy REST API endpoints (e.g., FastAPI) for integration with banking systems.
- **Monitoring & Retraining:** Establish continuous model monitoring for data drift and automated retraining pipelines.
- **Expanded Risk Metrics:** Integrate macroeconomic indicators to refine credit risk assessments.