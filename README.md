# Medical Insurance Cost Prediction

## Project Overview
This project builds an end-to-end data science pipeline to analyze and predict **medical insurance costs** using demographic and health-related factors. The workflow covers data cleaning, exploratory analysis, SQL validation, feature engineering, and machine learning model comparison.

The objective is twofold:
1. Understand the key drivers of insurance charges  
2. Compare interpretable linear models with non-linear machine learning models

---

## Dataset
- Source: Medical Insurance dataset (Kaggle)
- Records: ~1,337 individuals
- Key attributes:
  - Age
  - Sex
  - BMI
  - Number of children
  - Smoking status
  - Region
  - Insurance charges (target)

---

## Project Structure
medical-insurance-project/
│── data/
│ ├── insurance.csv
│ ├── insurance_sql_ready.csv
│ └── insurance.db
│── notebooks/
│ ├── 04_feature_engineering.ipynb
│ ├── 05_pandas_business_questions.ipynb
│ ├── 06_sql_analysis.ipynb
│ ├── 07_modeling_linear_regression.ipynb
│ └── 08_modeling_random_forest.ipynb
│── sql/
│ └── queries.sql
│── utils/
│ └── insurance_pipeline.py
│── README.md


---

## Methodology

### 1. Data Cleaning & Feature Engineering
- Standardized categorical values
- Removed duplicate records
- Log-transformed insurance charges to reduce skew
- Engineered interaction feature: **BMI × Smoking**
- One-hot encoded regional categories
- Centralized logic in a reusable pipeline (`insurance_pipeline.py`)

---

### 2. Exploratory Analysis (Pandas)
- Compared insurance costs by smoking status, age group, region, and BMI
- Identified smoking as the dominant cost driver
- Confirmed heavy right-skew in raw insurance charges

---

### 3. SQL Validation
- Exported a SQL-ready dataset
- Loaded data into SQLite
- Reproduced Pandas insights using SQL queries
- Ensured consistency across analytical layers

---

### 4. Modeling Approach

#### Baseline Model: Linear Regression
- Target: `log_charges`
- Strengths: Interpretability, coefficient-based reasoning
- Limitations: Assumes linear relationships

#### Non-Linear Model: Random Forest Regressor
- Captures non-linear effects and interactions automatically
- Provides feature importance rankings
- Improves predictive accuracy

---

## Model Performance

| Model              | R²    | RMSE  | MAE   |
|-------------------|-------|-------|-------|
| Linear Regression | 0.837 | 0.389 | 0.248 |
| Random Forest     | 0.841 | 0.385 | 0.204 |

**Observation:**  
Random Forest provides a modest but consistent improvement, particularly in reducing average prediction error.

---

## Key Insights
- Smoking-related features are the strongest predictors of insurance costs.
- BMI significantly amplifies cost risk for smokers.
- Age contributes steadily to higher charges.
- Regional differences exist but are relatively minor.
- Non-linear models better capture complex interactions.

---

## Tools & Technologies
- Python (Pandas, NumPy)
- scikit-learn
- SQLite
- Matplotlib
- Jupyter Notebook

---

## Conclusion
This project demonstrates a full data science workflow—from raw data ingestion to validated SQL analysis and model comparison. Linear regression offers interpretability, while Random Forest improves predictive performance, highlighting the trade-off between explainability and accuracy.

---
