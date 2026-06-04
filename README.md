# 🏥 Hospital Readmission Risk Prediction

## Project Description
A complete end-to-end Machine Learning pipeline that predicts whether 
a diabetic patient will be readmitted to hospital within 30 days of discharge.

The project covers data cleaning, exploratory data analysis, feature 
engineering, model training and comparison, hyperparameter tuning, 
evaluation, and deployment as a live web application.

**Target variable:** Readmitted within 30 days → Binary classification (0 or 1)  
**Best model:** XGBoost (Tuned) — F1: 0.8429 | AUC: 0.9944 | Recall: 0.8888

---

## 📊 Dataset

| Detail | Info |
|---|---|
| Name | UCI Diabetes 130-US Hospitals (1999–2008) |
| Original Source | [Kaggle - Diabetes 130 Hospitals](https://www.kaggle.com/datasets/brandao/diabetes) |
| Records | 101,766 patient encounters |
| Features | 50 original columns |
| Cleaned Dataset | [Download from Google Drive](https://drive.google.com/file/d/1PXXaD54hLj_stQIx5nD0S35-BVnmJRZ0/view?usp=sharing) |

---

## 🎬 Demo Video

📺 [Watch the full project presentation and app demo](https://canva.link/vv7s23bor2x3snz)

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/ShroukElsheikh/Analyzing-of-130-US-Hospitals-Diabetes-1999-2008-using-Machine-Learning.git
cd hospital-readmission-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Open the notebook
Open `ML_project.ipynb` in Google Colab or Jupyter Notebook

---

## 📦 Required Libraries
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
imbalanced-learn
matplotlib
seaborn
---

## 📈 Results

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.7393 | 0.9754 | 0.8411 | 0.9902 |
| Random Forest | 0.7683 | 0.9404 | 0.8457 | 0.9938 |
| XGBoost (default) | 0.7794 | 0.9118 | 0.8404 | 0.9946 |
| **XGBoost (tuned)** | **0.8016** | **0.8888** | **0.8429** | **0.9944** |

---

## 🌐 Live Web Application

🔗 [Open the live Streamlit app](https://analyzing130hospitaldiabetes.streamlit.app/)

---

## 👤 Team Members

- Shrouk Taher Elsheikh

---

## 📝 Additional Notes

- SMOTE was applied on training data only to handle class imbalance
- Duplicate patient encounters were deduplicated before modeling
- Patient history features were engineered before deduplication
- All dates reference the UCI dataset collection period (1999–2008)
