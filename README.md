
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

#  Real Estate Investment Advisor  
Predict 5-year future price & classify whether a property is a **good investment** using Machine Learning, statistical analysis, and explainability with SHAP — deployed as an interactive **Streamlit Web App**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Business Use Case](#business-use-case)
- [Project Overview](#project-overview)
- [Demo Streamlit Preview](#demo-streamlit-preview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Hypothesis Testing](#hypothesis-testing)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Results](#model-results)
- [Why These Final Models?](#why-these-final-models)
- [SHAP Explainability](#shap-explainability)
- [Conclusion](#conclusion)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)
- [Connect With Me](#connect-with-me)
- [Acknowledgements](#acknowledgements)

---

##  Problem Statement

Real estate buyers struggle to estimate whether a property is worth investing in. This project builds a system to:

1. **Predict the estimated price of a property after 5 years**, and  
2. **Classify whether the property is a good investment**, using multiple economic, location-based, and structural property features.

---

##  Business Use Case

This system helps:

- 🏦 **Investors** – understand long-term appreciation  
- 🏢 **Real Estate Platforms** – recommend properties intelligently  
- 🏘️ **Home Buyers** – identify underpriced or high-growth opportunities  
- 📊 **Analysts** – leverage historical data + ML to predict appreciation  

---

##  Project Overview

The project includes:

- End-to-end **data cleaning**, **processing**, and **EDA**
- **Hypothesis testing** for statistical validation
- **Feature engineering** (amenity scoring, BHK per 1000 sqft, city-tier encoding, growth-rate modeling)
- **Regression models** to predict 5-year future price  
- **Classification models** to label “Good Investment”  
- **MLflow experiment tracking**
- **Streamlit App** with:
  - Form-based predictions  
  - Investment classification  
  - Price forecast  
  - Real-time SHAP explanations  
  - Visual analytics (boxplots, heatmaps, maps)

---

##  Demo Streamlit Preview

<p align='center'>
<img width="1917" height="873" alt="front_view" src="https://github.com/user-attachments/assets/8cd1a050-3be9-4922-a947-4a3aebc5868d" />
<img width="953" height="837" alt="image" src="https://github.com/user-attachments/assets/9a5f876c-82e4-42ff-a04e-2f5e0a8011ba" />
<img width="1902" height="857" alt="image" src="https://github.com/user-attachments/assets/3c961ca7-0f2c-4bad-a3d7-cc5edabab709" />
</p>

###  Live App / Demo Video (Upload your link)
➡️ [Streamlit Video Demo](https://drive.google.com/file/d/1gT7fNhCxcfRlV4HWteDSui9cKeAR0pKE/view?usp=sharing)

---

##  Objectives

- Understand data relationships via **EDA & statistical tests**
- Create **future price estimation model (Regression)**
- Build **Good Investment classifier (Classification)**
- Use **MLflow** to version models
- Deploy a **user-friendly Streamlit app**
- Add **model explainability** using SHAP  
- Provide **insights for investors**

---

##  Dataset

| Source | Synthetic Data |
|--------|--------------------------------|
| Total Rows | 2,50,000 |
| Features | 22 |
| Targets | `Future_Price_5Y`, `Good_Investment` |

Key Columns:  
`City`, `State`, `Locality`, `BHK`, `Size_in_SqFt`, `Price_in_Lakhs`, `Parking_Space`, `Amenities`, `Transport_Score`, etc.

---

##  Data Preprocessing

Steps applied:

1. Missing value treatment (median/mode imputation)
2. Categorical encoding (OneHotEncoder)
3. Outlier handling for price, sqft
4. StandardScaler for numeric features
5. 623 final engineered model-ready features

---

##  Exploratory Data Analysis (EDA)

Key findings:

- **City-level variation** huge → strong indicator for price  
- **Locality** influences price more than **State**  
- **BHK vs Price** shows high variance → nonlinear relationship  
- **Amenities count** correlates with higher price  
- **PPSF distribution** varies significantly across cities  
- **Price heavily right-skewed** → tree-based models preferred  
- **No strong linearity between size & price**  
- **Transport score, availability, parking** show patterns but mild impact  

---

##  Hypothesis Testing

| Test | Summary | Outcome |
|------|---------|---------|
| ANOVA (Property Type vs PPSF) | No PPSF difference across property types | ❌ Fail to Reject H₀ |
| T-Test (Parking vs Price) | Parking does not influence price | ❌ Fail to Reject H₀ |
| Chi-Square (Owner Type vs Availability) | No dependency | ❌ Fail to Reject H₀ |
| Pearson (Size vs Price) | No linear correlation | ❌ Fail to Reject H₀ |
| Spearman (Age vs Price) | No monotonic relation | ❌ Fail to Reject H₀ |
| Kruskal-Wallis (PPSF Across States) | No state-wise difference | ❌ Fail to Reject H₀ |

**Insight:**  
> Statistically, many features that *looked* different in EDA were not significantly different.  
> So modeling emphasizes **City**, **Locality**, **Amenities**, **Transport**, **Property Type (encoded)**, not age/size.

---

##  Feature Engineering

Key engineered features:

- `Amenity_Count`  
- `BHK_per_1000SqFt`  
- `Price_per_100SqFt`  
- `City_Tier`  
- `Owner_Score`  
- `Furnish_Score`  
- `Facing_Score`  
- `Availability_Flag`  
- Growth-rate features using:  
```

Future_Price_5Y = Current_Price * (1 + Hybrid_Growth_Rate)^5

````

---

##  Machine Learning Models

### Regression  
- **Linear Regression**  
- **Random Forest Regressor**  
- **XGBoost Regressor** (Winner 🏆)

### Classification  
- **Logistic Regression**  
- **Random Forest Classifier**   
- **XGBoost Classifier** (Winner 🏆)

### MLflow  
- Track parameters, metrics  
- Store versions of models  
- Compare experiments wrt MAE, RMSE, R², ROC-AUC

---

##  Model Results

### Regression Results

| Model | MAE | RMSE | R² |
|-------|------|--------|------|
| Linear Regression | 84.28 | 106.50 | 0.76 |
| Random Forest | 114.13 | 135.78 | 0.61 |
| **XGBoost** | **10.47** | **15.30** | **0.995** |

### Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|------------|---------|---------|-----------|
| Logistic Regression | 0.906 | 0.93 | 0.956 | 0.943 | 0.957 |
| Random Forest | 0.951 | 0.999 | 0.94 | 0.969 | 0.987 |
| **XGBoost** | **0.9976** | **0.9990** | **0.9980** | **0.9985** | **0.99994** |

---

##  Why These Final Models?

### ✔ **XGBoost Regressor = Best Price Predictor**  
- Captures non-linear interactions  
- Handles categorical encodings well  
- 99.5% R² → extremely accurate

### ✔ **Random Forest Classifier = Best Investment Classifier**  
- Handles imbalanced target better  
- Great precision (0.99+)  
- Stable, interpretable via SHAP  

---

##  SHAP Explainability

<p align='center'>
<img src="https://github.com/user-attachments/assets/shap_reg.png" width="48%"/>
<img src="https://github.com/user-attachments/assets/shap_clf.png" width="48%"/>
</p>

**Top Features Driving Predictions:**

### Regression  
- City  
- Locality OHE vectors  
- Transport Score  
- BHK  
- Size Bucket  
- Amenity Count  

### Classification  
- BHK  
- Transport Score  
- Parking  
- Amenity Count  
- City Tier  

---

##  Conclusion

This project builds an end-to-end **real estate intelligence system** combining:

- **EDA insights**  
- **Statistical validation**  
- **ML modeling**  
- **Explainability**  
- **Deployment**

**Final Deliverables Include:**

✔ ML models for price prediction & investment classification  
✔ SHAP-based transparency  
✔ MLflow experiment logs  
✔ A fully interactive Streamlit web application  
✔ High-quality EDA visual summaries  
✔ Statistical hypothesis validation  

---

##  Installation & Usage

### Create Environment

```bash
git clone https://github.com/ut-si-ch/Real-Estate-Investment-Advisor.git
cd Real-Estate-Investment-Advisor

conda create -n reia_env python=3.9
conda activate reia_env

pip install -r requirements.txt
````

### Run Streamlit App

```bash
streamlit run app.py
```

---

##  Project Structure

```
├── dataset/
│   ├── india_housing_prices.csv
│   ├── processed_housing_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_streamlit_app_dev.ipynb
│
├── models/
│   ├── RandomForest_reg_model.pkl
│   ├── RandomForest_clf_model.pkl
│   ├── preprocessor.pkl
│   └── feature_names.csv
│
├── app.py
├── requirements.txt
└── README.md
```

---

##  Key Learnings

* Importance of validating EDA insights using **hypothesis tests**
* Handling large datasets with efficient preprocessing pipelines
* Impact of feature engineering (amenity scores, growth rates)
* Benefits of XGBoost for tabular problems
* Using SHAP for interpretability
* Building professional Streamlit dashboards
* MLflow for experiment tracking & reproducibility

---

##  Connect With Me

* **LinkedIn:** [https://www.linkedin.com/in/uttam-singh-chaudhary-98408214b](https://www.linkedin.com/in/uttam-singh-chaudhary-98408214b)
* **Portfolio:** [https://datascienceportfol.io/uttamsinghchaudhary](https://datascienceportfol.io/uttamsinghchaudhary)
* **Email:** [uttamsinghchaudhary@gmail.com](mailto:uttamsinghchaudhary@gmail.com)

---

##  Acknowledgements
* Scikit-Learn
* XGBoost
* Streamlit
* SHAP
* MLflow
* Folium

---
