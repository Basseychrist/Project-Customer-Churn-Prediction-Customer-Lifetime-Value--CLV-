"""
Data Preparation Module for Customer Churn Prediction

Handles:
- Loading IBM Telco Customer Churn dataset
- Handling missing values in TotalCharges
- Feature engineering (tenure_bucket, services_count, monthly_to_total_ratio)
- CLV calculation (MonthlyCharges × ExpectedTenure)
- Train/Val/Test stratified split (60/20/20)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath=None):
    """
    Load IBM Telco Customer Churn dataset and handle missing values.
    
    Args:
        filepath: Path to CSV file. If None, attempts to load from sklearn or download.
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    if filepath:
        df = pd.read_csv(filepath)
    else:
        # Define raw data path
        raw_data_path = 'data/raw/WA_Fn-UseC_-_Telco_Customer_Churn.csv'
        
        # Try to load from local raw directory
        try:
            df = pd.read_csv(raw_data_path)
        except FileNotFoundError:
            print("Raw data not found. Downloading IBM Telco dataset from GitHub...")
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            df = pd.read_csv(url)
            
            # Save to raw directory for future use
            from pathlib import Path
            Path('data/raw').mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_data_path, index=False)
            print(f"✅ Dataset saved to {raw_data_path}")
    
    # Handle TotalCharges: convert to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Approach: Fill missing TotalCharges with MonthlyCharges × tenure
    # Reasoning: TotalCharges = MonthlyCharges × tenure, so missing values
    # are likely data entry errors or new customers
    df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df


def engineer_features(df):
    """
    Engineer explainable, business-driven features.
    
    New features:
    - tenure_bucket: 0-6m, 6-12m, 12-24m, 24m+
    - services_count: total number of services (Internet, Security, Backup, etc.)
    - monthly_to_total_ratio: monthly relative to total charges
    """
    df = df.copy()
    
    # tenure_bucket
    def tenure_to_bucket(months):
        if months < 6:
            return '0-6m'
        elif months < 12:
            return '6-12m'
        elif months < 24:
            return '12-24m'
        else:
            return '24m+'
    
    df['tenure_bucket'] = df['tenure'].apply(tenure_to_bucket)
    
    # services_count: count of service columns (binary: Yes/No)
    service_cols = [
        'PhoneService', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    
    def count_services(row):
        count = 0
        for col in service_cols:
            if col in df.columns and row[col] == 'Yes':
                count += 1
        return count
    
    df['services_count'] = df.apply(count_services, axis=1)
    
    # monthly_to_total_ratio
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / np.maximum(1, df['TotalCharges'])
    
    # Additional flag: Internet but no tech support (high risk)
    df['internet_no_techsupport'] = (
        (df['InternetService'] != 'No') & 
        (df['TechSupport'] == 'No')
    ).astype(int)
    
    print(f"Engineered features created: tenure_bucket, services_count, monthly_to_total_ratio, internet_no_techsupport")
    
    return df


def calculate_clv(df, monthly_charge_col='MonthlyCharges', expected_tenure_months=24):
    """
    Calculate Customer Lifetime Value (CLV) for each customer.
    
    Formula: CLV = MonthlyCharges × ExpectedTenure
    
    Assumption: 
    - Expected tenure of 24 months (2 years) as a baseline
    - For longer-tenure customers (24+ months), use their observed tenure
    - This reflects that established customers are higher-value but helps avoid
      overestimating CLV for very new customers
    
    Args:
        df: DataFrame
        monthly_charge_col: Column name for monthly charges
        expected_tenure_months: Default expected tenure in months
        
    Returns:
        pd.Series: CLV for each customer
    """
    df_clv = df.copy()
    
    # Adjust expected tenure: use actual tenure if > expected, else use expected
    adjusted_tenure = df_clv['tenure'].apply(
        lambda x: max(x, expected_tenure_months)
    )
    
    clv = df_clv[monthly_charge_col] * adjusted_tenure
    
    print(f"\nCLV Summary Statistics:")
    print(f"Mean CLV: ${clv.mean():.2f}")
    print(f"Median CLV: ${clv.median():.2f}")
    print(f"Std Dev CLV: ${clv.std():.2f}")
    
    return clv


def create_clv_quartiles(df, clv_series):
    """
    Split customers into CLV quartiles: Low, Medium, High, Premium
    
    Args:
        df: DataFrame
        clv_series: CLV series
        
    Returns:
        pd.Series: CLV quartile labels
    """
    quartiles = pd.qcut(clv_series, q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    return quartiles


def encode_categorical_features(df, categorical_cols, fit_encoders=True, encoders=None):
    """
    Encode categorical variables using LabelEncoder (alphabetical order).
    
    Important: LabelEncoder sorts alphabetically, so:
    - Gender: Female=0, Male=1
    - MultipleLines: No=0, No Phone Service=1, Yes=2
    
    Args:
        df: DataFrame
        categorical_cols: List of categorical column names
        fit_encoders: If True, fit new encoders; if False, use provided encoders
        encoders: Dict of fitted encoders (used when fit_encoders=False)
        
    Returns:
        pd.DataFrame: Encoded DataFrame
        dict: Dictionary of fitted encoders
    """
    df_encoded = df.copy()
    encoder_dict = {}
    
    if encoders is None:
        encoders = {}
    
    for col in categorical_cols:
        if col not in df_encoded.columns:
            continue
            
        if fit_encoders:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoder_dict[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        else:
            if col in encoders:
                df_encoded[col] = encoders[col].transform(df_encoded[col].astype(str))
                encoder_dict[col] = encoders[col]
    
    return df_encoded, encoder_dict


def prepare_data(raw_data_path=None, test_size=0.2, val_size=0.2, random_state=42):
    """
    Complete data preparation pipeline.
    
    Stratified split: 60% train, 20% val, 20% test
    Stratification on Churn to maintain class balance.
    
    Args:
        raw_data_path: Path to raw CSV file
        test_size: Proportion for test set (from remaining after val split)
        val_size: Proportion for validation set (from original)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (df_train, df_val, df_test, encoder_dict, clv_dict)
    """
    # Load and clean
    df = load_and_clean_data(raw_data_path)
    
    # Engineer features
    df = engineer_features(df)
    
    # Calculate CLV
    clv = calculate_clv(df, expected_tenure_months=24)
    df['CLV'] = clv
    
    # Create CLV quartiles
    df['CLV_quartile'] = create_clv_quartiles(df, clv)
    
    # Define categorical and numeric columns
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns 
                       if col not in ['Churn', 'customerID']]
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns]
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    
    # Encode categorical variables (fit on full data, then split)
    df_encoded, encoder_dict = encode_categorical_features(
        df, categorical_cols, fit_encoders=True
    )
    
    # Prepare target variable
    df_encoded['Churn_encoded'] = (df_encoded['Churn'] == 'Yes').astype(int)
    
    # Stratified split: first split into 80/20 (train+val vs test)
    df_temp, df_test = train_test_split(
        df_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=df_encoded['Churn_encoded']
    )
    
    # Then split train+val into 75/25 (60% train, 20% val of total)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_size / (1 - test_size),  # Adjust for already-split data
        random_state=random_state,
        stratify=df_temp['Churn_encoded']
    )
    
    print(f"\nData Split Completed:")
    print(f"Train: {df_train.shape[0]} ({100*df_train.shape[0]/df_encoded.shape[0]:.1f}%)")
    print(f"Val: {df_val.shape[0]} ({100*df_val.shape[0]/df_encoded.shape[0]:.1f}%)")
    print(f"Test: {df_test.shape[0]} ({100*df_test.shape[0]/df_encoded.shape[0]:.1f}%)")
    
    print(f"\nChurn Rate by Split:")
    print(f"Train: {df_train['Churn_encoded'].mean():.2%}")
    print(f"Val: {df_val['Churn_encoded'].mean():.2%}")
    print(f"Test: {df_test['Churn_encoded'].mean():.2%}")
    
    # Prepare CLV dict for reference
    clv_dict = {
        'expected_tenure_months': 24,
        'monthly_charge_col': 'MonthlyCharges',
        'train_clv_mean': df_train['CLV'].mean(),
        'train_clv_median': df_train['CLV'].median()
    }
    
    return df_train, df_val, df_test, encoder_dict, clv_dict


if __name__ == '__main__':
    # Example usage
    df_train, df_val, df_test, encoders, clv_info = prepare_data()
    
    # Save processed data
    df_train.to_csv('data/processed/train.csv', index=False)
    df_val.to_csv('data/processed/val.csv', index=False)
    df_test.to_csv('data/processed/test.csv', index=False)
    
    print("\nProcessed data saved to data/processed/")
