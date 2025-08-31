# 📊 Customer Churn Prediction

This project focuses on building and evaluating machine learning models to predict customer churn based on various demographic and service-related features. It involves end-to-end steps from data preprocessing, exploratory data analysis (EDA), model building, hyperparameter tuning, and insights generation.

---

## 📂 Project Structure
- `notebook.ipynb`: The main Jupyter Notebook containing EDA, preprocessing, modeling, and evaluation.  
- `label_encoders.pkl`: Pickled encoders used for transforming categorical features.  
- `README.md`: Project documentation.  

---

## 📑 Dataset Overview
The dataset contains information about telecom customers, including:  
- **Demographics** (e.g., gender, senior citizen status)  
- **Account information** (e.g., contract type, payment method)  
- **Usage patterns** (e.g., tenure, monthly charges, total charges)  
- **Churn status** (target variable)  

---

## 🧹 Data Preprocessing
- Converted `TotalCharges` to numeric and handled missing values.  
- Encoded categorical variables using `LabelEncoder`.  
- Scaled numerical features using `StandardScaler`.  
- Balanced the training dataset using **ADASYN** to address class imbalance.  

---

## 🔎 Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Tenure and TotalCharges show skewed or segmented patterns.  
- **Correlation Matrix**: Strong positive correlation between tenure and total charges.  
- **Churn Patterns**:  
  - High churn among newer customers.  
  - Customers with high monthly charges are more likely to churn.  

---

## 💡 Actionable Insights
- Focus retention efforts on new users and high-paying customers.  
- Provide loyalty rewards for long-term customers.  
- Consider pricing experiments for sensitive segments.  

---

## 🤖 Modeling
The following models were trained and tuned using **GridSearchCV**:  
- Logistic Regression  
- Random Forest  
- XGBoost  
- CatBoost  

Each model was evaluated using **accuracy** and other classification metrics.  

**Best Results**  
- Random Forest: **82.21%**  
- CatBoost: 81.86%  
- XGBoost: 81.55%  
- Logistic Regression: 75.88%  

---

## ⚙️ Dependencies
Install all required libraries using:  

```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost catboost imbalanced-learn
