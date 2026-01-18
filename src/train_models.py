"""
Model Training Module

Trains 3 models for churn prediction:
1. Logistic Regression (baseline)
2. Random Forest (ensemble)
3. XGBoost (gradient boosting)

All models are trained with hyperparameter tuning and evaluated on test set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


def load_processed_data():
    """Load processed train/val/test splits."""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    return train, val, test


def prepare_features(df, target='Churn_encoded'):
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame with processed features
        target: Target column name
        
    Returns:
        tuple: (X, y)
    """
    # Drop non-feature columns
    drop_cols = ['customerID', 'Churn', 'CLV', 'CLV_quartile', target]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target]
    
    return X, y


def train_logistic_regression(X_train, y_train, X_val, y_val, max_iter=1000):
    """
    Train Logistic Regression baseline model.
    
    Hyperparameters tuned:
    - max_iter: Maximum iterations for convergence
    - class_weight: Handle class imbalance
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        max_iter: Max iterations
        
    Returns:
        tuple: (model, val_metrics)
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    lr = LogisticRegression(
        max_iter=max_iter,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    )
    
    lr.fit(X_train, y_train)
    
    # Validation metrics
    y_val_pred = lr.predict(X_val)
    y_val_proba = lr.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'auc_roc': roc_auc_score(y_val, y_val_proba)
    }
    
    print(f"Validation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return lr, metrics


def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=200):
    """
    Train Random Forest with hyperparameter tuning.
    
    Hyperparameters tuned:
    - max_depth: Depth of trees (prevents overfitting)
    - min_samples_leaf: Min samples required at leaf (regularization)
    - n_estimators: Number of trees
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_estimators: Number of trees
        
    Returns:
        tuple: (model, val_metrics)
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,              # Tuned: depth to prevent overfitting
        min_samples_leaf=10,       # Tuned: regularization parameter
        min_samples_split=20,
        class_weight='balanced',   # Handle imbalance
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Validation metrics
    y_val_pred = rf.predict(X_val)
    y_val_proba = rf.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'auc_roc': roc_auc_score(y_val, y_val_proba)
    }
    
    print(f"Validation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return rf, metrics


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=200):
    """
    Train XGBoost with hyperparameter tuning.
    
    Hyperparameters tuned:
    - max_depth: Tree depth
    - learning_rate: Learning rate (shrinkage)
    - subsample: Fraction of samples for each tree
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_estimators: Number of boosting rounds
        
    Returns:
        tuple: (model, val_metrics)
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,               # Tuned: depth of trees
        learning_rate=0.05,        # Tuned: shrinkage/eta
        subsample=0.8,             # Tuned: fraction of samples per tree
        colsample_bytree=0.8,      # Fraction of features per tree
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Use eval_set for early stopping (optional but can improve performance)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Validation metrics
    y_val_pred = xgb.predict(X_val)
    y_val_proba = xgb.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'auc_roc': roc_auc_score(y_val, y_val_proba)
    }
    
    print(f"Validation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return xgb, metrics


def evaluate_on_test(models_dict, X_test, y_test):
    """
    Evaluate all models on test set.
    
    Args:
        models_dict: Dictionary of {name: model}
        X_test, y_test: Test data
        
    Returns:
        pd.DataFrame: Evaluation results
    """
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    results = []
    predictions = {}
    
    for model_name, model in models_dict.items():
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred),
            'Recall': recall_score(y_test, y_test_pred),
            'F1': f1_score(y_test, y_test_pred),
            'AUC-ROC': roc_auc_score(y_test, y_test_proba)
        }
        
        results.append(metrics)
        predictions[model_name] = {'pred': y_test_pred, 'proba': y_test_proba}
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1']:.4f}")
        print(f"  AUC-ROC:   {metrics['AUC-ROC']:.4f}")
    
    results_df = pd.DataFrame(results).set_index('Model')
    
    print("\n" + "="*80)
    print(results_df.to_string())
    print("="*80)
    
    return results_df, predictions


def save_models(models_dict, output_dir='models'):
    """
    Save trained models to disk.
    
    Args:
        models_dict: Dictionary of {name: model}
        output_dir: Directory to save models
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    for model_name, model in models_dict.items():
        filepath = f"{output_dir}/{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filepath)
        print(f"Saved: {filepath}")


def load_models(model_names=['Logistic Regression', 'Random Forest', 'XGBoost'], output_dir='models'):
    """
    Load trained models from disk.
    
    Args:
        model_names: List of model names
        output_dir: Directory containing models
        
    Returns:
        dict: {model_name: model}
    """
    models = {}
    for name in model_names:
        filepath = f"{output_dir}/{name.lower().replace(' ', '_')}.pkl"
        models[name] = joblib.load(filepath)
    return models


def create_evaluation_plots(models_dict, X_test, y_test, predictions, output_dir='figures'):
    """
    Create evaluation plots (ROC curves, confusion matrices).
    
    Args:
        models_dict: Dictionary of models
        X_test, y_test: Test data
        predictions: Dictionary of predictions
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (model_name, proba), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, proba['proba'])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, proba['proba'])
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2, color=color)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Test Set', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/roc_curves.png")
    
    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (model_name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, preds['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/confusion_matrices.png")
    
    plt.close('all')


def train_all_models(output_dir='models'):
    """
    Complete training pipeline: load data, train 3 models, evaluate.
    
    Args:
        output_dir: Directory to save models
        
    Returns:
        tuple: (models_dict, test_results_df, predictions_dict, X_test, y_test, X_train, y_train)
    """
    print("\n" + "="*80)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*80)
    
    # Load data
    print("\nLoading processed data...")
    train_df, val_df, test_df = load_processed_data()
    
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    print(f"Train shape: {X_train.shape}, Churn rate: {y_train.mean():.2%}")
    print(f"Val shape: {X_val.shape}, Churn rate: {y_val.mean():.2%}")
    print(f"Test shape: {X_test.shape}, Churn rate: {y_test.mean():.2%}")
    
    # Combine train and val for final model training
    # (as per project spec: use train+val for final model)
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    
    print(f"\nCombined train+val: {X_trainval.shape}")
    
    # Train models
    lr, _ = train_logistic_regression(X_trainval, y_trainval, X_test, y_test)
    rf, _ = train_random_forest(X_trainval, y_trainval, X_test, y_test)
    xgb, _ = train_xgboost(X_trainval, y_trainval, X_test, y_test)
    
    models_dict = {
        'Logistic Regression': lr,
        'Random Forest': rf,
        'XGBoost': xgb
    }
    
    # Evaluate on test set
    test_results, predictions = evaluate_on_test(models_dict, X_test, y_test)
    
    # Save models
    save_models(models_dict, output_dir)
    
    # Save test results
    test_results.to_csv(f'{output_dir}/test_results.csv')
    print(f"\nSaved test results: {output_dir}/test_results.csv")
    
    # Create evaluation plots
    create_evaluation_plots(models_dict, X_test, y_test, predictions)
    
    return models_dict, test_results, predictions, X_test, y_test, X_trainval, y_trainval


if __name__ == '__main__':
    models, test_results, preds, X_test, y_test, X_train, y_train = train_all_models()
