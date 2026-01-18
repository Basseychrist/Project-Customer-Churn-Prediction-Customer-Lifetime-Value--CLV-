"""
Interpretability Module

Provides model explanations:
- SHAP TreeExplainer for Random Forest and XGBoost
- Standardized coefficient analysis for Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Fallback to feature importance will be used.")


def load_models(output_dir='models'):
    """Load trained models from disk."""
    return {
        'Logistic Regression': joblib.load(f'{output_dir}/logistic_regression.pkl'),
        'Random Forest': joblib.load(f'{output_dir}/random_forest.pkl'),
        'XGBoost': joblib.load(f'{output_dir}/xgboost.pkl')
    }


def get_feature_names(X_sample):
    """Get feature names from data."""
    if isinstance(X_sample, pd.DataFrame):
        return X_sample.columns.tolist()
    return [f'Feature_{i}' for i in range(X_sample.shape[1])]


def get_logistic_regression_importance(model, X_train):
    """
    Extract standardized feature importance for Logistic Regression.
    
    Formula: importance = |coefficient * std_dev_of_feature|
    
    This is more appropriate for linear models than SHAP because:
    1. Coefficients directly represent feature contribution
    2. Standardization accounts for feature scale
    3. Faster and more interpretable than KernelExplainer
    
    Args:
        model: Fitted LogisticRegression
        X_train: Training features (for computing std)
        
    Returns:
        pd.DataFrame: Feature importance sorted by absolute value
    """
    feature_names = get_feature_names(X_train)
    
    # Get coefficients
    coefs = model.coef_[0]
    
    # Standardize features
    X_std = (X_train - X_train.mean()) / X_train.std()
    feature_std = X_std.std()
    
    # Compute importance
    importance = np.abs(coefs * feature_std)
    
    result = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance,
        'Coefficient': coefs
    }).sort_values('Importance', ascending=False)
    
    return result


def get_tree_explainer_importance(model, X_sample):
    """
    Extract global feature importance using SHAP TreeExplainer.
    
    Args:
        model: Fitted Random Forest or XGBoost model
        X_sample: Sample of data (can be subset for speed)
        
    Returns:
        pd.DataFrame: Feature importance sorted by mean absolute SHAP value
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Using model's built-in feature_importances_")
        feature_names = get_feature_names(X_sample)
        importance = model.feature_importances_
        
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values (sample if large dataset for speed)
    if len(X_sample) > 200:
        X_sample_small = X_sample.sample(200, random_state=42)
    else:
        X_sample_small = X_sample
    
    shap_values = explainer.shap_values(X_sample_small)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # For binary classification, SHAP returns [neg_class, pos_class]
        shap_values = shap_values[1]
    
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Ensure mean_abs_shap is 1D (take first row if 2D, handle edge case)
    mean_abs_shap = np.asarray(mean_abs_shap).ravel()
    
    feature_names = get_feature_names(X_sample)
    
    # Ensure lengths match
    if len(mean_abs_shap) != len(feature_names):
        mean_abs_shap = mean_abs_shap[:len(feature_names)]
    
    result = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)
    
    return result


def create_global_importance_plots(importance_dicts, output_dir='figures'):
    """
    Create and save global feature importance plots.
    
    Args:
        importance_dicts: Dictionary of {model_name: importance_df}
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for ax, (model_name, importance_df), color in zip(axes, importance_dicts.items(), colors):
        top_features = importance_df.head(15)
        
        ax.barh(range(len(top_features)), top_features['Importance'].values, color=color, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values, fontsize=9)
        ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\nTop 15 Features', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/global_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/global_feature_importance.png")
    plt.close()


def get_local_explanation(model, X_sample, instance_idx=0):
    """
    Get local explanation for a single instance.
    
    Args:
        model: Fitted model
        X_sample: Feature data
        instance_idx: Index of instance to explain
        
    Returns:
        dict: Explanation with prediction and top contributing features
    """
    feature_names = get_feature_names(X_sample)
    X_instance = X_sample.iloc[[instance_idx]]
    
    pred_proba = model.predict_proba(X_instance)[0][1]
    pred_class = model.predict(X_instance)[0]
    
    # Get feature importance using SHAP if available
    if SHAP_AVAILABLE and hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_values = shap_values[0]
        
        explanation = pd.DataFrame({
            'Feature': feature_names,
            'Value': X_instance.iloc[0].values,
            'SHAP': shap_values
        })
        explanation['Abs_SHAP'] = np.abs(explanation['SHAP'])
        explanation = explanation.sort_values('Abs_SHAP', ascending=False)
    else:
        # Fallback: use feature values (simple approach)
        explanation = pd.DataFrame({
            'Feature': feature_names,
            'Value': X_instance.iloc[0].values
        })
    
    return {
        'pred_proba': pred_proba,
        'pred_class': pred_class,
        'explanation': explanation
    }


def create_model_interpretability(models_dict, X_train, output_dir='models'):
    """
    Create and save model interpretability artifacts.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X_train: Training features (for computing importance)
        output_dir: Directory to save artifacts
    """
    print("\n" + "="*80)
    print("GENERATING MODEL INTERPRETABILITY")
    print("="*80)
    
    importance_dicts = {}
    
    # Logistic Regression
    print("\nExtracting Logistic Regression importance (standardized coefficients)...")
    lr_importance = get_logistic_regression_importance(
        models_dict['Logistic Regression'], 
        X_train
    )
    importance_dicts['Logistic Regression'] = lr_importance
    print(lr_importance.head(10).to_string())
    
    # Random Forest
    print("\nExtracting Random Forest importance...")
    rf_importance = get_tree_explainer_importance(
        models_dict['Random Forest'],
        X_train
    )
    importance_dicts['Random Forest'] = rf_importance
    print(rf_importance.head(10).to_string())
    
    # XGBoost
    print("\nExtracting XGBoost importance...")
    xgb_importance = get_tree_explainer_importance(
        models_dict['XGBoost'],
        X_train
    )
    importance_dicts['XGBoost'] = xgb_importance
    print(xgb_importance.head(10).to_string())
    
    # Create and save plots
    create_global_importance_plots(importance_dicts, 'figures')
    
    # Save importance tables
    for model_name, imp_df in importance_dicts.items():
        filepath = f"{output_dir}/{model_name.lower().replace(' ', '_')}_importance.csv"
        imp_df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")
    
    return importance_dicts


if __name__ == '__main__':
    # Load models and training data
    models = load_models()
    X_train = pd.read_csv('data/processed/train.csv')
    
    # Prepare features
    drop_cols = ['customerID', 'Churn', 'CLV', 'CLV_quartile', 'Churn_encoded']
    X_train = X_train.drop(columns=[col for col in drop_cols if col in X_train.columns])
    
    # Generate interpretability
    importance_dicts = create_model_interpretability(models, X_train)
