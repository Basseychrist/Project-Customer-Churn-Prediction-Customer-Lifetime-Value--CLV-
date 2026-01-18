# PROJECT SETUP COMPLETE ✅

## What's Been Created

Your **Customer Churn Prediction & Customer Lifetime Value (CLV) Analysis** project is now fully scaffolded and ready to run!

### 📁 Directory Structure

```
project2-churn-prediction/
├── README.md                    # Complete project documentation
├── AI_USAGE.md                  # AI assistance summary
├── requirements.txt             # All dependencies (pinned versions)
├── .gitignore                   # Git ignore rules
├── app.py                       # Streamlit interactive web app (3 tabs)
├── run_pipeline.py              # One-command pipeline runner
├── data/
│   ├── raw/                     # For raw IBM Telco dataset
│   └── processed/               # Processed train/val/test splits
├── src/
│   ├── data_prep.py             # Data loading, cleaning, feature engineering
│   ├── clv_analysis.py          # CLV calculation and business insights
│   ├── train_models.py          # Train 3 models (LR, RF, XGB)
│   ├── interpretability.py      # SHAP & feature importance
│   └── predict.py               # Prediction utilities
├── models/                      # Trained models & results
├── figures/                     # Generated visualizations
└── notebooks/                   # Optional: exploratory analysis
```

---

## 🚀 Quick Start (5 Steps)

### 1. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
 source venv/Scripts/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
- Get IBM Telco Customer Churn dataset from:
  - Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn
  - GitHub: https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data
- Save to: `data/raw/WA_Fn-UseC_-_Telco_Customer_Churn.csv`

### 3. Run Complete Pipeline
```bash
python run_pipeline.py
```
This runs all 5 steps automatically:
- Data prep (clean, engineer features)
- CLV analysis (segment customers)
- Model training (3 models + evaluation)
- Interpretability (SHAP + feature importance)
- Output (models, plots, results)

### 4. Review Results
```bash
# Check model performance
cat models/test_results.csv

# Check business insights
open figures/churn_by_clv.png
```

### 5. Launch Web App
```bash
streamlit run app.py
```
Open browser to `http://localhost:8501`

---

## 📊 What Each Component Does

### Data Preparation (`src/data_prep.py`)
✅ Loads IBM Telco dataset  
✅ Handles missing TotalCharges (fills with `MonthlyCharges × tenure`)  
✅ Engineers 4 explainable features  
✅ Calculates CLV using formula: `MonthlyCharges × ExpectedTenure`  
✅ Splits into train/val/test (60/20/20) with stratification  
✅ Encodes categorical variables (LabelEncoder, alphabetical order)  

### CLV Analysis (`src/clv_analysis.py`)
✅ Computes CLV per customer  
✅ Segments into quartiles (Low, Medium, High, Premium)  
✅ Analyzes churn rate by segment  
✅ Generates business insights  
✅ Creates visualizations (saved to `figures/`)  

### Model Training (`src/train_models.py`)
✅ Trains 3 models:
  - Logistic Regression (baseline, interpretable)
  - Random Forest (ensemble, robust)
  - XGBoost (state-of-the-art, tuned)  
✅ Light hyperparameter tuning  
✅ Evaluates on test set (Precision, Recall, F1, AUC)  
✅ Creates ROC curves and confusion matrices  
✅ Saves models to `models/` directory  

### Interpretability (`src/interpretability.py`)
✅ SHAP TreeExplainer for tree models (RF, XGBoost)  
✅ Standardized coefficients for Logistic Regression  
✅ Global feature importance (top 15 per model)  
✅ Saves importance tables to CSV  
✅ Creates visualization plots  

### Streamlit App (`app.py`)
✅ **Tab 1 - Predict**:
  - Input customer features
  - View churn probability + risk label
  - See estimated CLV
  - Model agreement & feature importance  

✅ **Tab 2 - Model Performance**:
  - Metrics table (all 3 models)
  - ROC curves overlay
  - Confusion matrices
  - Global feature importance  

✅ **Tab 3 - CLV Overview**:
  - CLV distribution histogram
  - Churn rate by segment
  - Business takeaway  

---

## 🎯 Expected Results

### Model Performance (Test Set)
- **Accuracy**: ~80%
- **Precision**: ~65%
- **Recall**: 60–70% ← Important! Catch most churners
- **AUC-ROC**: ~84%

### Feature Importance (Top Features)
- Tenure (strong negative, more tenure = less churn)
- Contract (month-to-month = high risk)
- Services (bundles reduce churn)
- Monthly charges (sometimes proxy for tenure)

### CLV Insights
- Low-CLV customers: 50%+ churn rate
- High/Premium: <5% churn rate
- **Implication**: Focus retention on high-value segments

---

## 🔧 Customization Options

### Change CLV Expected Tenure
Edit `src/data_prep.py`, line with:
```python
def calculate_clv(df, monthly_charge_col='MonthlyCharges', expected_tenure_months=24):
    # Change 24 to your assumed tenure (e.g., 36 for 3 years)
```

### Adjust Hyperparameters
Edit `src/train_models.py` for:
- Logistic Regression: `max_iter`, `class_weight`
- Random Forest: `max_depth`, `min_samples_leaf`
- XGBoost: `max_depth`, `learning_rate`, `subsample`

### Change Risk Thresholds
Edit `src/predict.py` in `get_churn_risk_label()`:
```python
if churn_probability < 0.3:      # Change thresholds
    return 'Low Risk'
```

---

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `README.md` | Full project documentation (start here) |
| `AI_USAGE.md` | What AI helped with, key decisions |
| `requirements.txt` | Python dependencies (pinned versions) |
| `app.py` | Streamlit web app (launch with `streamlit run app.py`) |
| `run_pipeline.py` | Runs all pipeline steps in sequence |
| `src/data_prep.py` | Data loading, cleaning, feature engineering |
| `src/clv_analysis.py` | CLV computation, segmentation, insights |
| `src/train_models.py` | Model training, evaluation, ROC curves |
| `src/interpretability.py` | SHAP explainers, feature importance |
| `src/predict.py` | Single-customer prediction utilities |

---

## 🧪 Verification Checklist

Before declaring success, verify:

- [ ] Downloaded IBM Telco dataset to `data/raw/`
- [ ] `python run_pipeline.py` completes without errors
- [ ] `models/test_results.csv` shows AUC-ROC ≥ 0.80 and Recall ≥ 0.60
- [ ] `figures/` directory contains all plots
- [ ] `models/` contains 3 pkl files + CSV importance tables
- [ ] `streamlit run app.py` launches without errors
- [ ] App displays all 3 tabs correctly
- [ ] Predict tab accepts inputs and shows churn probability
- [ ] Model Performance tab shows ROC curves and metrics
- [ ] CLV Overview tab displays business insights

---

## 🚢 Next Steps (Deployment)

### For Local Development
1. Keep iterating, tuning hyperparameters
2. Test with different customer profiles
3. Validate business insights against domain experts

### For Production/Sharing
1. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial: Churn prediction & CLV analysis"
   git push origin main
   ```

2. Deploy to Streamlit Cloud:
   - Go to https://streamlit.io/cloud
   - Click "New app" → Select repo, branch, file (`app.py`)
   - Deploy (automatic)
   - Share public URL

3. Ensure requirements.txt is complete:
   ```bash
   pip freeze > requirements.txt  # (Optional: update with frozen versions)
   ```

---

## 💡 Pro Tips

1. **Model Training**: The script trains on combined train+val data (as per spec) for final models. This maximizes training data.

2. **Feature Engineering**: All features are explainable and business-relevant (no black-box embeddings). This makes the model trustworthy.

3. **SHAP**: Tree models use TreeExplainer (fast). Logistic Regression uses standardized coefficients (faster + more interpretable than KernelExplainer).

4. **Caching**: Streamlit app caches models persistently and data for 60 minutes. This makes the app snappy.

5. **Ensemble**: Final prediction is the average of all 3 models. This reduces variance and improves robustness.

---

## 📞 Support & Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
→ Run `pip install -r requirements.txt`

**"Models not found"**
→ Run `python src/train_models.py` to train

**"Port 8501 in use"**
→ Use `streamlit run app.py --server.port=8502`

**"SHAP import error"**
→ Install with: `pip install shap==0.43.0`

For more, see **Troubleshooting** section in `README.md`.

---

## 📖 Learning Resources

- **Streamlit**: https://docs.streamlit.io/
- **SHAP**: https://shap.readthedocs.io/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Scikit-Learn**: https://scikit-learn.org/
- **Churn Prediction**: https://www.kaggle.com/blastchar/telco-customer-churn

---

## ✨ You're All Set!

Everything is scaffolded and ready. The next steps are:

1. **Download the data** (IBM Telco dataset)
2. **Run the pipeline** (`python run_pipeline.py`)
3. **Launch the app** (`streamlit run app.py`)
4. **Deploy when ready** (Streamlit Cloud)

Questions? Check `README.md` for detailed instructions!

**Happy analyzing! 🎉**

---

**Last Updated**: January 2026
