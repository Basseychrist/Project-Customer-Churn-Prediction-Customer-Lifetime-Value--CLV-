"""
Prediction Module

Handles single-instance predictions with interpretability.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def load_models_and_encoders(models_dir='models'):
    """Load trained models and encoders."""
    models = {
        'Logistic Regression': joblib.load(f'{models_dir}/logistic_regression.pkl'),
        'Random Forest': joblib.load(f'{models_dir}/random_forest.pkl'),
        'XGBoost': joblib.load(f'{models_dir}/xgboost.pkl')
    }
    
    return models


def fit_encoders_from_training_data(train_df):
    """
    Fit categorical encoders from training data.
    
    Args:
        train_df: Training DataFrame with all features
        
    Returns:
        dict: Dictionary of fitted LabelEncoders
    """
    categorical_cols = [col for col in train_df.select_dtypes(include=['object']).columns 
                       if col not in ['Churn', 'customerID']]
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        encoders[col] = le
    
    return encoders, categorical_cols


def predict_churn(input_dict, models_dict, train_df):
    """
    Predict churn probability for a customer.
    
    Args:
        input_dict: Dictionary of feature values (raw, before encoding)
        models_dict: Dictionary of trained models
        train_df: Training DataFrame (used to get feature order and fit encoders)
        
    Returns:
        dict: Predictions and probabilities from all models
    """
    # Get encoders from training data
    encoders, categorical_cols = fit_encoders_from_training_data(train_df)
    
    # Create DataFrame from input
    X = pd.DataFrame([input_dict])
    
    # Identify all columns that need encoding (anything not in numeric training data)
    numeric_training_cols = set(train_df.select_dtypes(include=[np.number]).columns)
    
    # Encode ALL categorical variables (not just those we identified)
    for col in X.columns:
        # Skip non-feature columns
        if col in ['customerID', 'Churn', 'CLV', 'CLV_quartile', 'Churn_encoded']:
            continue
            
        # If column is in training data and is categorical, encode it
        if col in train_df.columns:
            if col not in numeric_training_cols:  # It's categorical in training data
                if col in encoders:
                    try:
                        X.loc[0, col] = encoders[col].transform([str(X.loc[0, col])])[0]
                    except ValueError:
                        # Use first class if value not found
                        print(f"Warning: {col}='{X.loc[0, col]}' not in training classes. Using '{encoders[col].classes_[0]}'.")
                        X.loc[0, col] = encoders[col].transform([encoders[col].classes_[0]])[0]
        # If column is not in training data, it might be an engineered feature we need to skip
        elif col in ['tenure_bucket', 'services_count', 'monthly_to_total_ratio', 'internet_no_techsupport']:
            # These are engineered features - keep as is, they should already be numeric or will be encoded
            if col == 'tenure_bucket':
                # Encode tenure_bucket using same logic as in data_prep
                buckets = ['0-6m', '6-12m', '12-24m', '24m+']
                if col in encoders:
                    try:
                        X.loc[0, col] = encoders[col].transform([str(X.loc[0, col])])[0]
                    except ValueError:
                        print(f"Warning: tenure_bucket value not found. Using first class.")
                        X.loc[0, col] = 0
    
    # Get feature order from training data (excluding non-feature columns)
    drop_cols = ['customerID', 'Churn', 'CLV', 'CLV_quartile', 'Churn_encoded']
    expected_features = [col for col in train_df.columns if col not in drop_cols]
    
    # Make sure all expected features are in X
    for feat in expected_features:
        if feat not in X.columns:
            # This shouldn't happen if input_dict has all features
            print(f"Warning: Expected feature '{feat}' not in input. Setting to 0.")
            X[feat] = 0
    
    # Reorder and select only expected features
    X = X[expected_features]
    
    # Convert all remaining string values to numeric (in case any slipped through)
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert string columns to numeric
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Fill any NaN from failed conversions
                X[col] = X[col].fillna(0)
            except:
                X[col] = 0
    
    # Final conversion to float
    X = X.astype(float)
    
    # Check for NaN or Inf values and replace with 0
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    predictions = {}
    probabilities = {}
    
    for model_name, model in models_dict.items():
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        
        predictions[model_name] = pred
        probabilities[model_name] = proba
    
    # Ensemble: average probability across models
    ensemble_proba = np.mean(list(probabilities.values()))
    
    # Ensure ensemble_proba is valid (0-1)
    if np.isnan(ensemble_proba) or np.isinf(ensemble_proba):
        ensemble_proba = 0.5
    else:
        ensemble_proba = max(0.0, min(1.0, float(ensemble_proba)))
    
    return {
        'individual_probabilities': probabilities,
        'ensemble_probability': ensemble_proba,
        'individual_predictions': predictions
    }


def get_churn_risk_label(probability):
    """
    Get human-readable churn risk label based on probability.
    
    Args:
        probability: Churn probability (0-1)
        
    Returns:
        str: Risk label with emoji
    """
    if probability < 0.3:
        return "🟢 Low Risk"
    elif probability < 0.6:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"


def calculate_clv(monthly_charges, tenure_months=None, expected_tenure=24):
    """
    Calculate estimated CLV for a customer.
    
    CLV = MonthlyCharges × ExpectedTenure
    
    Args:
        monthly_charges: Monthly charges ($)
        tenure_months: Actual tenure (optional, for adjustment)
        expected_tenure: Default expected tenure in months
        
    Returns:
        float: Estimated CLV
    """
    if tenure_months is None:
        adjusted_tenure = expected_tenure
    else:
        # Use actual tenure if longer than expected
        adjusted_tenure = max(tenure_months, expected_tenure)
    
    clv = monthly_charges * adjusted_tenure
    return clv


if __name__ == '__main__':
    # Example usage
    from data_prep import prepare_data
    
    df_train, df_val, df_test, encoders, clv_info = prepare_data()
    models = load_models_and_encoders()
    
    # Sample customer (with all required features)
    sample_input = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.5,
        'TotalCharges': 1026.0,
        'tenure_bucket': '6-12m',
        'services_count': 0,
        'monthly_to_total_ratio': 0.083,
        'internet_no_techsupport': 1
    }
    
    result = predict_churn(sample_input, models, df_train)
    print("Churn Probabilities:", result['individual_probabilities'])
    print("Ensemble Probability:", result['ensemble_probability'])
    print("Risk Label:", get_churn_risk_label(result['ensemble_probability']))
