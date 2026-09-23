# 📊 HR Employee Attrition Analysis with Explainable AI (XAI)

An end-to-end **Python data science project** for analyzing and predicting **employee attrition** using the IBM HR Analytics dataset. This project goes beyond traditional machine learning by integrating **Explainable AI (XAI)** techniques — particularly **SHAP (SHapley Additive exPlanations)** — to interpret model predictions and uncover the key drivers behind employee turnover.

---

## 🎯 Project Overview

Employee attrition is one of the most costly challenges for organizations. This project aims to:

- 🔍 **Explore** the IBM HR dataset through comprehensive Exploratory Data Analysis (EDA)
- 🧹 **Clean & preprocess** the data (missing values, duplicates, outliers, low-variance features)
- 🎯 **Select** the most relevant features correlated with attrition
- 🤖 **Train** machine learning models to predict employee attrition
- 🧠 **Explain** model decisions using **SHAP** and other XAI techniques
- 📈 **Deliver actionable insights** for HR decision-makers

---

## ✨ Key Features

- 📊 **Exploratory Data Analysis (EDA)** — Distribution plots, boxplots, correlation heatmaps, count plots
- 🧹 **Robust Preprocessing Pipeline** — Null handling, duplicate removal, IQR-based outlier detection
- 🎯 **Feature Selection** — Correlation analysis + low-variance feature removal
- 🤖 **Machine Learning Modeling** — Classification models to predict attrition
- 🧠 **Explainable AI (XAI)** — SHAP values for both global and local interpretability
- 🔍 **Feature Importance Analysis** — Understand which factors drive attrition the most
- 📈 **Business-Friendly Visualizations** — Clear charts for HR stakeholders

---

## 🧠 Why Explainable AI (XAI)?

Traditional ML models act as **"black boxes"** — they give predictions but don't explain *why*. In HR analytics, interpretability is critical because:

- 🏢 **HR teams need to justify decisions** based on model outputs
- ⚖️ **Fairness & bias detection** — Ensure models don't discriminate
- 🔎 **Actionable insights** — Know *which* factors (e.g., OverTime, MonthlyIncome) drive attrition
- 📋 **Regulatory compliance** — Many industries require explainable models

This project uses **SHAP (SHapley Additive exPlanations)**, a game-theory-based approach that fairly distributes the prediction among features — providing both **global** and **local** explanations.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **Data Manipulation** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Explainable AI** | **SHAP**, XAI techniques |
| **Environment** | Jupyter Notebook |
| **Other** | pydotplus, IPython.display |

---

## 📂 Project Structure

```
HR-Attrition-Analysis/
├── Paksima-Project Notebook.ipynb          # Main analysis notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Original dataset
├── cleaned_hr_dataset.csv                  # Cleaned dataset (post-preprocessing)
└── README.md
```

---

## 📊 Dataset Overview

**Source:** IBM HR Analytics Employee Attrition & Performance Dataset

- **Rows:** 1,470 employees
- **Columns:** 35 features
- **Target:** `Attrition` (Yes / No)

**Feature Categories:**
- 👤 **Demographics** — Age, Gender, MaritalStatus
- 💼 **Job Info** — Department, JobRole, JobLevel, BusinessTravel
- 💰 **Compensation** — MonthlyIncome, DailyRate, HourlyRate, StockOptionLevel
- 📈 **Satisfaction & Performance** — JobSatisfaction, EnvironmentSatisfaction, PerformanceRating
- ⏳ **Tenure** — YearsAtCompany, YearsInCurrentRole, TotalWorkingYears
- 🎓 **Education** — Education, EducationField

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/FatemehPaksima/HR-Attrition-Analysis.git
cd HR-Attrition-Analysis
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap jupyter pydotplus
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```

### 4. Open and run
Open `Paksima-Project Notebook.ipynb` and run all cells sequentially.

> ⚠️ Make sure `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in the same directory.

---

## 🔬 Methodology

### 📊 1. Data Exploration
- Distribution of the target variable (`Attrition`)
- Age, income, and tenure distributions
- Categorical feature analysis (Gender, OverTime, Education, Department)
- Pyramid chart: Education level by Gender

### 🧹 2. Data Preprocessing
- Null value check
- Duplicate removal
- **IQR-based outlier detection** across all numerical features
- Outlier distribution analysis per feature, split by Attrition label
- Removal of records with ≥ 3 outlier features

### 🎯 3. Feature Selection
- Correlation analysis with the target variable
- Removal of low-variance features (`EmployeeCount`, `Over18`, `StandardHours`)
- Dropping irrelevant ID columns (`EmployeeNumber`)
- High-correlation pair detection (e.g., `MonthlyIncome` ↔ `JobLevel`)

### 🤖 4. Machine Learning Modeling
- Train/test split
- Classification models to predict attrition

### 🧠 5. Explainable AI with SHAP
- **Global explanations** — Which features matter most overall
- **Local explanations** — Why a specific employee is predicted to leave
- **SHAP summary plots** — Visualize feature impact distribution
- **SHAP dependence plots** — Understand feature interactions

---

## 📈 Key Insights

> 🔍 *(This section will be updated with the actual insights from the analysis.)*

Some early findings from EDA:

- 📉 Employees with **lower MonthlyIncome** show a higher attrition rate
- ⏰ **OverTime** is strongly associated with attrition
- 🧑‍💼 **Younger employees** tend to leave more frequently
- 🎓 **Education level** has a moderate impact on attrition

SHAP analysis will reveal **the exact contribution of each feature** to individual predictions.

---

## 📜 License

This project was developed as a university **BSc final project**.  
Free to use for learning and research purposes.

---

## 👨‍💻 Author

**Fatemeh Paksima**
GitHub: [@FatemehPaksima](https://github.com/FatemehPaksima)

---

## ⭐ Acknowledgments

- IBM HR Analytics Dataset (publicly available on Kaggle)
- SHAP library by Scott Lundberg
- Scikit-learn community
