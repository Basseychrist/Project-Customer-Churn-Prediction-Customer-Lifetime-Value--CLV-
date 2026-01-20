# AI_USAGE.md: AI Assistance Summary

## Overview
This document describes what AI assistance was used in building this Customer Churn Prediction & CLV Analysis project.

---

## What AI Helped With

### Model Training (`src/train_models.py`)
**Task**: Train 3 models with hyperparameter tuning and evaluation.

**AI Input**:
- **Logistic Regression**: `class_weight='balanced'` to handle 26% churn imbalance
- **Random Forest**: `max_depth=15`, `min_samples_leaf=10` to prevent overfitting; 200 estimators for stability
- **XGBoost**: `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`; auto-calculate `scale_pos_weight` for imbalance
- Suggested ensemble averaging final predictions
- Recommended combining train+val for final model training (per spec)

**Implementation**:
- Each model trained with documented hyperparameter rationale
- Test set evaluation produces Precision, Recall, F1, AUC-ROC
- Models serialized to `models/` directory with `.pkl` format
- Test results saved to CSV for transparency

---

**Debugging**:
- **GitHub Copilot** was used to diagnose and resolve data pipeline issues:
  - Identified CSV encoding issues when processing raw telco data
  - Debugged feature scaling mismatches between train/test/val splits
  - Fixed dimensionality errors in model prediction pipelines
  - Traced and resolved categorical variable encoding inconsistencies
  - Provided inline code fixes for data transformation pipeline errors
- **Claude AI** provided detailed explanations for:
  - Resolving imbalanced dataset handling errors
  - Optimizing hyperparameter selections when models underperformed
  - Debugging pickle serialization and model loading issues
  - Identifying and fixing data leakage in feature engineering