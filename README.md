# 🚗 Used Car Price Prediction System

An end-to-end Machine Learning project that predicts the resale price of used cars based on vehicle specifications and historical market data. The project covers data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, hyperparameter tuning, and deployment using Streamlit.

---

## 📌 Project Overview

The objective of this project is to build an accurate machine learning model that estimates the selling price of a used car using features such as company, model, fuel type, transmission, ownership history, mileage, warranty, and quality score.

The project compares multiple regression algorithms and selects the best-performing model based on evaluation metrics and cross-validation.

---
## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

**Live Demo:** *(https://used-car-price-prediction-new.streamlit.app/)*

---

## 🚀 Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- Multiple Regression Models
- Hyperparameter Tuning
- Cross Validation
- Model Comparison
- Model Persistence using Joblib
- Streamlit Web Application

---

## 📂 Dataset

The dataset contains information about used cars, including:

- Company
- Model
- Variant
- Fuel Type
- Kilometer Driven
- Body Style
- Transmission Type
- Model Year
- Number of Owners
- Warranty Status
- Quality Score
- Selling Price (Target Variable)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Git & GitHub

---

## 📊 Exploratory Data Analysis

The project includes comprehensive EDA to understand the data and identify important patterns.

Performed analysis includes:

- Price Distribution
- Log Distribution of Price
- Correlation Heatmap
- Price vs Car Age
- Top Car Brands
- Average Price by Brand
- Transmission Type vs Price
- Fuel Type vs Price
- Owner vs Price
- Quality Score Distribution

---

## ⚙️ Feature Engineering

The following features were created to improve model performance:

- CarAge
- MileagePerYear
- IsAutomatic
- LuxuryBrand

Additional preprocessing steps include:

- Price conversion to numeric format
- One-Hot Encoding
- Removal of constant features
- Feature scaling for Ridge Regression

---

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Hyperparameter tuning was performed for Random Forest and XGBoost.

---

## 📈 Model Performance

| Model | MAE | RMSE | R² | CV Mean |
|------|------:|------:|------:|------:|
| Ridge Regression | 98,345 | 147,368 | **0.8053** | 0.7567 |
| XGBoost | 113,539 | 170,882 | 0.7383 | **0.7612** |
| Random Forest | 134,844 | 196,085 | 0.6554 | 0.7378 |
| Decision Tree | 170,908 | 259,201 | 0.3978 | 0.5308 |

**Final Model:** Ridge Regression

---

## 📉 Evaluation Metrics

The following metrics were used for model evaluation:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-Fold Cross Validation

---

## 💡 Key Insights

- Vehicle age significantly impacts resale price.
- Premium brands generally retain higher market value.
- Cars with lower mileage tend to have higher selling prices.
- Automatic transmission vehicles typically command better resale prices.
- Ridge Regression provided the best balance between accuracy and generalization.

---

## 🔮 Future Improvements

- Add real-time price estimation through external APIs.
- Integrate image-based car condition assessment.
- Include location-based market price adjustments.
- Deploy using Docker and cloud platforms.
- Expand the dataset with additional vehicle attributes.

---

## 👨‍💻 Author

**Aman Shaikh**

