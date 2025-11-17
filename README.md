# Diabetes Detection System

A machine learning project that detects diabetes using multiple classification algorithms. This system analyzes medical data and compares the performance of different models to predict whether a patient has diabetes.

## 📋 Project Overview

This project implements a comprehensive diabetes detection system that uses exploratory data analysis (EDA) and multiple machine learning classifiers to predict diabetes outcomes based on medical measurements. The system evaluates four different algorithms and provides detailed performance metrics and visualizations.

## 🎯 Objectives

- Load and explore diabetes medical dataset
- Clean and preprocess data by handling missing/invalid values
- Train multiple classification models
- Compare model performance using various metrics
- Visualize results with ROC curves and confusion matrices
- Identify feature importance

## 📊 Dataset

The project uses a diabetes dataset containing the following features:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mmHg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin level
- **BMI**: Body mass index (weight in kg / height in m²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Binary target variable (0 = No diabetes, 1 = Diabetes)

## 🔧 Data Preprocessing

The project includes several preprocessing steps:

1. **Missing Value Handling**: Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are replaced with the median values (as zero is not realistic for these medical measurements)
2. **Feature Scaling**: StandardScaler is applied to normalize features for better model performance
3. **Train-Test Split**: Data is split into 80% training and 20% testing sets

## 🤖 Machine Learning Models

The system evaluates four different classification algorithms:

1. **Logistic Regression**
   - Linear classification model
   - Good baseline for binary classification

2. **K-Nearest Neighbors (KNN)**
   - Instance-based learning with k=5 neighbors
   - Simple yet effective algorithm

3. **Decision Tree**
   - Tree-based model with interpretable results
   - Provides feature importance insights

4. **Random Forest**
   - Ensemble method with 100 decision trees
   - Reduces overfitting through averaging

## 📈 Evaluation Metrics

Each model is evaluated using:

- **Confusion Matrix**: Shows True Positives, True Negatives, False Positives, and False Negatives
- **Classification Report**: Precision, Recall, and F1-Score
- **ROC Curve**: Receiver Operating Characteristic curve
- **AUC Score**: Area Under the ROC Curve (0 to 1 scale)
- **Feature Importance**: For tree-based models (Decision Tree and Random Forest)

## 📦 Requirements

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 🚀 How to Use

1. Ensure all required libraries are installed
2. Place the diabetes dataset in the project directory
3. Open `Module5_Assigment_5.ipynb` in Jupyter Notebook
4. Run all cells sequentially to:
   - Load and explore the data
   - Preprocess and prepare features
   - Train all four models
   - Generate evaluation metrics and visualizations

## 📊 Output & Visualizations

The notebook generates:

- **ROC Curves**: Comparison of all models' ROC curves with AUC scores
- **Confusion Matrices**: 2x2 grid showing confusion matrices for all models
- **Feature Importance Charts**: Bar plots showing feature importance for Decision Tree and Random Forest models
- **Classification Reports**: Detailed metrics including precision, recall, and F1-scores

## 💡 Key Findings

The project helps identify:

- Which classification algorithm performs best on diabetes prediction
- Most important features for diabetes detection
- Trade-offs between precision and recall
- Overall model reliability through AUC scores

## 📝 Notes

- The random_state parameter is set to 42 for reproducible results
- StandardScaler is fit on training data and then applied to test data to prevent data leakage
- ROC curves are plotted only for models with `predict_proba` method

## 👤 Author

Course Assignment 5 (Module 5)

## 📄 License

This project is part of an educational assignment.

---

