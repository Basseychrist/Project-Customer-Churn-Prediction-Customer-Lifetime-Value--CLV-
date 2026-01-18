# AI_USAGE.md: AI Assistance Summary

## Overview
This document describes what AI assistance was used in building this Customer Churn Prediction & CLV Analysis project.

---

## What AI Helped With

### 1. Project Architecture & Structure
**Task**: Design the overall project structure, module organization, and data pipeline.

**AI Input**: 
- Suggested modular design with separate files for data prep, modeling, interpretability, and app
- Recommended caching strategies for Streamlit to optimize performance
- Proposed clear separation of concerns (training vs. prediction)

**Implementation**: 
- Created `src/` directory with focused modules
- Each module handles one responsibility (data prep, CLV analysis, model training, interpretability, prediction)
- Implemented Streamlit caching with `@st.cache_data` and `@st.cache_resource`

---

### 2. Data Preparation (`src/data_prep.py`)
**Task**: Load IBM Telco dataset, handle missing values, engineer features, calculate CLV, and split data.

**AI Input**:
- Suggested handling TotalCharges missing values using formula: `MonthlyCharges × tenure`
- Recommended feature engineering approach (tenure buckets, services count, ratios)
- Proposed stratified split logic (train/val/test 60/20/20) to maintain class balance
- Advised on LabelEncoder alphabetical sorting for reproducibility

**Implementation**:
- Missing value strategy documented and implemented
- Feature engineering produces explainable features
- `prepare_data()` function returns organized train/val/test with encoders

---

### 3. CLV Analysis (`src/clv_analysis.py`)
**Task**: Compute CLV, segment customers, analyze churn patterns, generate insights.

**AI Input**:
- CLV formula: `MonthlyCharges × ExpectedTenure` with 24-month baseline
- Adjustment for long-tenure customers (use actual tenure if >24m)
- Quartile segmentation (Low, Medium, High, Premium)
- Visualization recommendations (histogram, bar charts, box plots)

**Implementation**:
- CLV calculated with transparent assumptions
- Quartile-based segmentation enables business insights
- Plots saved to `figures/` directory for app display
- Business insights generated programmatically from data patterns

---

### 4. Model Training (`src/train_models.py`)
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

### 5. Interpretability (`src/interpretability.py`)
**Task**: Explain model predictions using SHAP and feature importance.

**AI Input**:
- **Tree Models (RF/XGBoost)**: Use SHAP TreeExplainer (fast, accurate)
- **Logistic Regression**: Standardized coefficients formula: `|coef × feature_std|`
  - Reasoning: More interpretable than KernelExplainer; avoids slow approximation
- Global importance: Top 15 features per model
- Local explanation: SHAP values for single predictions
- Sampling optimization: Use 200-row sample for global plots if dataset >200

**Implementation**:
- `get_logistic_regression_importance()` computes standardized coefficients
- `get_tree_explainer_importance()` uses SHAP for tree models
- Importance tables saved to CSV
- Plots generated for visualization

---

### 6. Streamlit App (`app.py`)
**Task**: Build interactive web app with 3 tabs for prediction, performance, and CLV analysis.

**AI Input**:
- Tab-based structure for organization
- Tab 1 (Predict):
  - Input form with validation
  - Risk labels based on probability thresholds
  - Feature importance visualization per model
  - CLV calculation displayed with formula
- Tab 2 (Model Performance):
  - Metrics table
  - ROC curves overlay
  - Confusion matrices
  - Global feature importance
- Tab 3 (CLV Overview):
  - CLV distribution histogram
  - Churn by quartile bar charts
  - Business takeaway summary

**Implementation**:
- Caching optimizes load time (models cached as resource, data as persistent)
- Input validation handles edge cases
- Ensemble probability shown alongside individual model predictions
- Risk color-coding (green=low, orange=medium, red=high)
- Model agreement display shows confidence

---

### 7. Documentation & Deployment
**Task**: Create comprehensive README.md and deployment guide.

**AI Input**:
- Structure: Overview → Quick Start → Project Structure → Technical Details → Troubleshooting
- Include encoding reference (LabelEncoder alphabetical order)
- Deployment steps for Streamlit Community Cloud
- Troubleshooting section for common issues

**Implementation**:
- README includes all required information
- Quick start with 5-step pipeline
- Technical details section for reproducibility
- Deployment guide with `.gitignore` sample
- Troubleshooting table for common issues

---

## What Was NOT AI-Generated

### Manual Verification & Testing
1. **Hyperparameter Tuning Logic**: AI suggested parameters; you should verify on your dataset
   - Test that recall ≥ 60% (catch most churners)
   - Verify AUC-ROC ≥ 0.80
   - Check that tenure, contract, services have high feature importance

2. **Feature Importance Validation**: 
   - Verify features ranked by importance make business sense
   - Confirm high-risk customer profile (tenure <6m, month-to-month, no support) → >60% churn

3. **Business Insights**: 
   - AI generated insights structure; YOU should validate against your data
   - Read the churn analysis output and refine insights

4. **Dataset**:
   - Download IBM Telco dataset independently
   - Place in `data/raw/` directory
   - Run `python src/data_prep.py` to verify it works with your data

---

## Key Decisions Made by AI

| Decision | Rationale |
|----------|-----------|
| CLV = MonthlyCharges × 24 months | Standard lifetime value approach; 24 months balances uncertainty vs. long tenure |
| Use actual tenure if > 24 months | Reflects that established customers have proven, longer relationships |
| LabelEncoder (alphabetical) | Ensures reproducibility across runs; important for app predictions |
| Stratified train/val/test split | Maintains churn class balance; prevents biased evaluation |
| class_weight='balanced' | Handles 26% churn imbalance automatically |
| Ensemble (average 3 models) | Reduces individual model variance; improves robustness |
| SHAP TreeExplainer for trees | Fast, accurate; avoids slow KernelExplainer |
| Standardized coefficients for LR | More interpretable than SHAP approximation for linear models |
| Streamlit caching strategy | Models cached as resource (persistent); data cached 60 min |

---

## Key Prompts That Shaped This Project

1. **"How should I calculate CLV given customer tenure uncertainty?"**
   - → Use expected 24-month baseline; adjust upward for proven long-tenure customers

2. **"How do I handle class imbalance (26% churn)?"**
   - → Use `class_weight='balanced'`, stratified splits, and focus on Recall metric

3. **"What's the fastest way to explain tree model predictions?"**
   - → SHAP TreeExplainer is O(depth) per sample; sample rows for global plots

4. **"How do I optimize Streamlit app performance?"**
   - → Cache models as resource (persistent), data as data (60-min TTL)

5. **"Should I use SHAP for Logistic Regression?"**
   - → No; standardized coefficients are faster, more interpretable for linear models

---

## What You Should Do Before Deployment

1. **Download the data**:
   - Get IBM Telco Customer Churn dataset from Kaggle or GitHub
   - Place in `data/raw/WA_Fn-UseC_-_Telco_Customer_Churn.csv`

2. **Run the full pipeline**:
   ```bash
   python src/data_prep.py
   python src/clv_analysis.py
   python src/train_models.py
   python src/interpretability.py
   ```

3. **Verify model quality**:
   - Check `models/test_results.csv` for AUC-ROC ≥ 0.80
   - Confirm Recall ≥ 0.60
   - Review feature importances (tenure, contract should rank high)

4. **Test the app**:
   ```bash
   streamlit run app.py
   ```
   - Try the high-risk customer test case (should show >60% churn)
   - Verify all 3 tabs load without errors
   - Check that plots display correctly

5. **Deploy**:
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Share public URL

---

## Summary

This project demonstrates:
- ✅ **Data-driven decision making**: CLV segments guide retention prioritization
- ✅ **Multiple models**: 3 models + ensemble for robustness
- ✅ **Interpretability**: SHAP + coefficients explain predictions
- ✅ **Production-ready**: Caching, error handling, documentation
- ✅ **Deployment-friendly**: Single Streamlit app, minimal dependencies

AI was used to design architecture, suggest algorithms, and structure code. **YOU are responsible for verifying model quality, testing assumptions, and validating business insights.**

---

**Last Updated**: January 2026
