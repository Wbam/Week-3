# Task 4: Model Evaluation and Feature Importance Analysis

## Overview

In this task, we perform a comprehensive evaluation of regression models to predict the `TotalPremium` of a dataset. We use three different models: Linear Regression, Random Forest, and XGBoost. Additionally, we analyze feature importance using SHAP (SHapley Additive exPlanations) to understand how individual features influence the model's predictions.

## Project Structure

- `data/` - Contains the dataset.
- `scripts/` - Includes Python scripts for data processing and model evaluation.
- `notebooks/` - Contains Jupyter notebooks for exploratory data analysis and visualizations.

## Data Preparation

1. **Loading Data:**

   - Load a subset of the data (10,000 samples) for model training and evaluation.

2. **Preprocessing:**

   - Handle missing values using the mean imputation strategy.
   - Convert data to numeric format and prepare it for modeling.

3. **Splitting Data:**
   - Split the dataset into training and test sets for model evaluation.

## Models

1. **Linear Regression:**

   - Fit a Linear Regression model to the training data.
   - Evaluate the model using RMSE (Root Mean Squared Error) and R² score on the test set.

2. **Random Forest:**

   - Fit a Random Forest Regressor with 100 trees.
   - Evaluate the model using RMSE and R² score.

3. **XGBoost:**
   - Fit an XGBoost Regressor with 100 trees and a squared error objective.
   - Evaluate the model using RMSE and R² score.

## Feature Importance Analysis

### SHAP Analysis

SHAP (SHapley Additive exPlanations) is used to analyze the impact of each feature on model predictions. We generate SHAP values for Random Forest and XGBoost models to identify the most influential features.

1. **Random Forest:**

   - Initialize SHAP TreeExplainer for the Random Forest model.
   - Compute SHAP values and generate a summary plot to visualize feature importance.

2. **XGBoost:**
   - Initialize SHAP TreeExplainer for the XGBoost model.
   - Compute SHAP values and generate a summary plot to visualize feature importance.

### LIME Analysis

LIME (Local Interpretable Model-agnostic Explanations) is used to interpret the predictions of the Linear Regression model.

1. **Linear Regression:**
   - Initialize LIME TabularExplainer.
   - Explain individual predictions and visualize the feature contributions.

## Instructions

1. **Install Dependencies:**
   Ensure that the required libraries are installed. You can install them using pip:

   ```bash
   pip install numpy pandas scikit-learn xgboost shap lime
   ```
