"""
Streamlit App: Customer Churn Prediction & CLV Analysis

Single-page app with 3 tabs:
1. Predict: Input customer features and get churn prediction + CLV
2. Model Performance: View metrics, ROC curves, confusion matrices
3. CLV Overview: CLV distribution and churn analysis by segment
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
import gc

warnings.filterwarnings('ignore')

# Configure matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# Force garbage collection to prevent memory leaks during rapid reloads
gc.collect()

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_prep import encode_categorical_features
from interpretability import (
    get_logistic_regression_importance,
    get_tree_explainer_importance,
    get_local_explanation
)
from predict import predict_churn, calculate_clv, get_churn_risk_label


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_processed_data():
    """Load processed splits."""
    try:
        train = pd.read_csv('data/processed/train.csv')
        val = pd.read_csv('data/processed/val.csv')
        test = pd.read_csv('data/processed/test.csv')
        return train, val, test
    except Exception as e:
        st.error(f"Error loading processed data: {e}")
        return None, None, None


@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        return {
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'XGBoost': joblib.load('models/xgboost.pkl')
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


@st.cache_resource
def load_importance():
    """Load feature importance tables."""
    try:
        return {
            'Logistic Regression': pd.read_csv('models/logistic_regression_importance.csv'),
            'Random Forest': pd.read_csv('models/random_forest_importance.csv'),
            'XGBoost': pd.read_csv('models/xgboost_importance.csv')
        }
    except Exception as e:
        st.error(f"Error loading importance: {e}")
        return {}


@st.cache_data(ttl=3600)
def load_test_results():
    """Load test evaluation results."""
    try:
        return pd.read_csv('models/test_results.csv', index_col='Model')
    except Exception as e:
        st.error(f"Error loading test results: {e}")
        return None


@st.cache_resource
def get_shap_explainer(model):
    """Get SHAP TreeExplainer for a model (cached)."""
    if not SHAP_AVAILABLE:
        return None
    try:
        return shap.TreeExplainer(model)
    except:
        return None


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Churn Prediction & CLV Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for stability
if 'last_error' not in st.session_state:
    st.session_state.last_error = None

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk { color: #d62728; font-weight: bold; }
    .medium-risk { color: #ff7f0e; font-weight: bold; }
    .low-risk { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Customer Churn Prediction & CLV Analysis")
st.markdown("""
This app predicts customer churn and estimates Customer Lifetime Value (CLV) 
using machine learning. Optimize retention efforts by identifying high-value 
customers at risk of leaving.
""")

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

# ========================================================================
# TABS
# ========================================================================

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Performance", "💰 CLV Overview"])

# ============================================================================
# TAB 1: PREDICT
# ============================================================================

with tab1:
    st.header("Make a Prediction")
    
    # Load data for categorical encoding info
    train_df, _, _ = load_processed_data()
    if train_df is None:
        st.error("Could not load training data. Please check if data files exist.")
        st.stop()
    
    models = load_models()
    if not models:
        st.error("Could not load models. Please check if model files exist.")
        st.stop()
    
    importance_dict = load_importance()
    if not importance_dict:
        st.error("Could not load feature importance. Please check if importance files exist.")
        st.stop()
    
    st.markdown("""
    Enter customer details below to predict churn probability and estimated CLV.
    **Note:** Numeric values should match the expected ranges in the dataset.
    """)
    
    # ========== PERSONAL INFO ==========
    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Customer gender")
        senior_citizen = st.checkbox("Senior Citizen?", value=False)
    with col2:
        partner = st.checkbox("Has Partner?", value=False)
        dependents = st.checkbox("Has Dependents?", value=False)
    
    # ========== SERVICE INFO ==========
    st.subheader("📞 Service Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], help="Has phone service")
        multiple_lines = st.selectbox(
            "Multiple Lines", 
            ["No", "No phone service", "Yes"],
            help="Multiple phone lines"
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["No", "DSL", "Fiber optic"],
            help="Type of internet service"
        )
    
    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No"], help="Online security addon")
        online_backup = st.selectbox("Online Backup", ["Yes", "No"], help="Online backup addon")
        device_protection = st.selectbox("Device Protection", ["Yes", "No"], help="Device protection addon")
    
    with col3:
        tech_support = st.selectbox("Tech Support", ["Yes", "No"], help="Tech support addon")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], help="Streaming TV addon")
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"], help="Streaming movies addon")
    
    # ========== ACCOUNT INFO ==========
    st.subheader("💳 Account Information")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
            help="Number of months as a customer"
        )
        
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=65.0,
            step=5.0,
            help="Monthly service charges"
        )
        
        # Auto-calculate total_charges based on tenure and monthly_charges
        # Defensive checks and type conversion
        try:
            tenure = int(tenure) if tenure is not None else 0
            tenure = max(0, min(72, tenure))
            
            monthly_charges = float(monthly_charges) if monthly_charges is not None else 0.0
            monthly_charges = max(0.0, min(200.0, monthly_charges))
            
            total_charges = float(monthly_charges * max(1, tenure))
        except (ValueError, TypeError):
            tenure = 0
            monthly_charges = 0.0
            total_charges = 0.0
        
        st.write(f"💡 **Total Charges (auto-calculated):** ${total_charges:.2f}")
    
    with col2:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help="Contract length"
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], help="Paperless billing enabled")
        payment_method = st.selectbox(
            "Payment Method",
            ["Bank transfer", "Credit card", "Electronic check", "Mailed check"],
            help="Preferred payment method"
        )
    
    # ========== FEATURE ENGINEERING DISPLAY ==========
    st.subheader("📊 Calculated Features")
    col1, col2, col3 = st.columns(3)
    
    # Tenure bucket
    if tenure < 6:
        tenure_bucket = '0-6m'
    elif tenure < 12:
        tenure_bucket = '6-12m'
    elif tenure < 24:
        tenure_bucket = '12-24m'
    else:
        tenure_bucket = '24m+'
    
    with col1:
        st.info(f"📍 **Tenure Bucket:** {tenure_bucket}")
    
    # Services count (manually calculate)
    services = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    services_count = sum(1 for s in services if s == 'Yes')
    
    with col2:
        st.info(f"🔧 **Services Count:** {services_count}/6")
    
    # Monthly to total ratio (protected from edge cases)
    ratio = monthly_charges / max(1.0, total_charges) if total_charges > 0 else 0.0
    if np.isnan(ratio) or np.isinf(ratio):
        ratio = 0.0
    with col3:
        st.info(f"📊 **Monthly/Total Ratio:** {ratio:.4f}")
    
    # CLV calculation
    expected_tenure = 24
    adjusted_tenure = max(tenure, expected_tenure)
    estimated_clv = monthly_charges * adjusted_tenure
    
    st.metric("Estimated CLV", f"${estimated_clv:.2f}", 
              delta=f"${monthly_charges:.2f}/month × {adjusted_tenure} months")
    
    # Make prediction
    if st.button("🚀 Predict Churn & CLV", width='stretch'):
        
        # Build input dict with all features in alphabetical order (matching encoder)
        input_dict = {
            'gender': gender,
            'SeniorCitizen': int(senior_citizen),
            'Partner': 'Yes' if partner else 'No',
            'Dependents': 'Yes' if dependents else 'No',
            'tenure': int(tenure),
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'tenure_bucket': tenure_bucket,
            'services_count': services_count,
            'monthly_to_total_ratio': ratio,
            'internet_no_techsupport': int(internet_service != 'No' and tech_support == 'No')
        }
        
        # Get predictions with error handling
        try:
            result = predict_churn(input_dict, models, train_df)
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
            st.info("Please check your inputs and try again.")
            st.stop()
        
        # Validate prediction results
        if result is None or 'ensemble_probability' not in result:
            st.error("Invalid prediction result. Please try again.")
            st.stop()
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Churn probability
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ensemble_proba = result['ensemble_probability']
            st.metric(
                "Ensemble Probability",
                f"{ensemble_proba*100:.1f}%",
                delta=f"{get_churn_risk_label(ensemble_proba)}"
            )
        
        with col2:
            lr_proba = result['individual_probabilities']['Logistic Regression']
            st.metric(
                "Logistic Regression",
                f"{lr_proba*100:.1f}%"
            )
        
        with col3:
            # Show the higher risk model prediction
            rf_proba = result['individual_probabilities']['Random Forest']
            xgb_proba = result['individual_probabilities']['XGBoost']
            
            st.metric(
                "RF & XGBoost (Avg)",
                f"{(rf_proba + xgb_proba)/2*100:.1f}%"
            )
        
        # Risk label with color
        risk_label = get_churn_risk_label(ensemble_proba)
        if ensemble_proba < 0.3:
            st.success(f"✅ **Risk Level: {risk_label}** - This customer is stable.")
        elif ensemble_proba < 0.6:
            st.warning(f"⚠️ **Risk Level: {risk_label}** - Monitor this customer.")
        else:
            st.error(f"🚨 **Risk Level: {risk_label}** - Urgent action recommended!")
        
        # CLV and value retention
        st.markdown("---")
        st.subheader("💰 Customer Lifetime Value")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Estimated CLV", f"${estimated_clv:.2f}")
            st.caption(f"Expected value over {adjusted_tenure} months at ${monthly_charges:.2f}/month")
        
        with col2:
            retention_value = estimated_clv * (1 - ensemble_proba)
            st.metric("Retained Value at Current Risk", f"${retention_value:.2f}")
            st.caption(f"Expected CLV if retention efforts succeed")
        
        # Model agreement
        st.markdown("---")
        st.subheader("🔍 Model Agreement")
        
        proba_values = list(result['individual_probabilities'].values())
        proba_std = np.std(proba_values)
        proba_range = max(proba_values) - min(proba_values)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction Range", f"{proba_range*100:.1f}%")
        with col2:
            if proba_range < 0.1:
                st.success("✅ Models agree (high confidence)")
            elif proba_range < 0.2:
                st.info("ℹ️ Models mostly agree")
            else:
                st.warning("⚠️ Models disagree (lower confidence)")
        
        # Feature importance explanation
        st.markdown("---")
        st.subheader("📊 Feature Importance")
        
        st.info("""
        The charts below show which features are most influential in predicting churn.
        **For this customer:** The features entered above (tenure, monthly charges, services, etc.)
        are evaluated by each model to produce the churn prediction.
        """)
        
        # Global importance for reference
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Logistic Regression")
            lr_imp = importance_dict['Logistic Regression'].head(8)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(range(len(lr_imp)), lr_imp['Importance'].values, color='#1f77b4')
            ax.set_yticks(range(len(lr_imp)))
            ax.set_yticklabels(lr_imp['Feature'].values, fontsize=9)
            ax.set_xlabel('Importance', fontsize=9)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Random Forest")
            rf_imp = importance_dict['Random Forest'].head(8)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(range(len(rf_imp)), rf_imp['Importance'].values, color='#ff7f0e')
            ax.set_yticks(range(len(rf_imp)))
            ax.set_yticklabels(rf_imp['Feature'].values, fontsize=9)
            ax.set_xlabel('Importance', fontsize=9)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        with col3:
            st.subheader("XGBoost")
            xgb_imp = importance_dict['XGBoost'].head(8)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(range(len(xgb_imp)), xgb_imp['Importance'].values, color='#2ca02c')
            ax.set_yticks(range(len(xgb_imp)))
            ax.set_yticklabels(xgb_imp['Feature'].values, fontsize=9)
            ax.set_xlabel('Importance', fontsize=9)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ============================================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================================

with tab2:
    st.header("Model Performance on Test Set")
    
    test_results = load_test_results()
    
    st.markdown("""
    This section shows how all 3 models performed on the held-out test set.
    **Test set:** 20% of data, unseen during training.
    """)
    
    # Metrics table
    st.subheader("📊 Performance Metrics")
    st.dataframe(
        test_results.round(4),
        width='stretch',
        height=200
    )
    
    st.markdown("""
    **Metric Definitions:**
    - **Accuracy**: % of correct predictions
    - **Precision**: Of predicted churners, % that actually churned
    - **Recall**: Of actual churners, % we correctly identified
    - **F1-Score**: Harmonic mean of Precision and Recall
    - **AUC-ROC**: Area Under the Receiver Operating Characteristic curve (0.5–1.0)
    """)
    
    # ROC curves
    st.subheader("📈 ROC Curves")
    
    try:
        if Path('figures/roc_curves.png').exists():
            img = plt.imread('figures/roc_curves.png')
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("ROC curves plot not found. Please run training first.")
    except Exception as e:
        st.warning(f"Could not load ROC curves: {e}")
    
    st.markdown("""
    The ROC curve shows the trade-off between True Positive Rate and False Positive Rate.
    A perfect model touches the top-left corner (AUC = 1.0). Random guessing is the diagonal line (AUC = 0.5).
    """)
    
    # Confusion matrices
    st.subheader("🎯 Confusion Matrices")
    
    try:
        if Path('figures/confusion_matrices.png').exists():
            img = plt.imread('figures/confusion_matrices.png')
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.imshow(img)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Confusion matrices plot not found. Please run training first.")
    except Exception as e:
        st.warning(f"Could not load confusion matrices: {e}")
    
    st.markdown("""
    Each matrix shows:
    - **TN (Top-left)**: Correctly predicted no churn
    - **FP (Top-right)**: Incorrectly predicted churn
    - **FN (Bottom-left)**: Missed churn
    - **TP (Bottom-right)**: Correctly predicted churn
    """)
    
    # Feature importance
    st.subheader("🔍 Global Feature Importance")
    
    try:
        if Path('figures/global_feature_importance.png').exists():
            img = plt.imread('figures/global_feature_importance.png')
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.imshow(img)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Feature importance plot not found. Please run training first.")
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
    
    st.markdown("""
    Feature importance shows which inputs drive churn predictions:
    - **Logistic Regression**: Standardized coefficients (|coef × std|)
    - **Random Forest & XGBoost**: Mean absolute SHAP values (impact on predictions)
    """)


# ============================================================================
# TAB 3: CLV OVERVIEW
# ============================================================================

with tab3:
    st.header("💰 Customer Lifetime Value Overview")
    
    train_df, _, _ = load_processed_data()
    
    st.markdown("""
    This section analyzes CLV distribution and churn patterns by segment.
    Insights guide retention prioritization.
    """)
    
    # Ensure CLV_quartile exists and handle edge cases
    if 'CLV_quartile' not in train_df.columns:
        try:
            # Handle NaN values in CLV before creating quartiles
            clv_values = train_df['CLV'].fillna(train_df['CLV'].mean())
            train_df['CLV_quartile'] = pd.qcut(clv_values, q=4, labels=['Low', 'Medium', 'High', 'Premium'], duplicates='drop')
        except Exception as e:
            st.error(f"Could not create CLV quartiles: {e}")
            train_df['CLV_quartile'] = 'Unknown'
    
    # CLV distribution
    st.subheader("📊 CLV Distribution")
    
    try:
        if Path('figures/clv_distribution.png').exists():
            img = plt.imread('figures/clv_distribution.png')
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(img)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("CLV distribution plot not found.")
    except Exception as e:
        st.warning(f"Could not load CLV distribution: {e}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean CLV", f"${train_df['CLV'].mean():.2f}")
    with col2:
        st.metric("Median CLV", f"${train_df['CLV'].median():.2f}")
    with col3:
        st.metric("Total Customer Value", f"${train_df['CLV'].sum():.0f}")
    
    # Churn by CLV quartile
    st.subheader("📈 Churn Rate by CLV Segment")
    
    try:
        if Path('figures/churn_by_clv.png').exists():
            img = plt.imread('figures/churn_by_clv.png')
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(img)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Churn by CLV plot not found.")
    except Exception as e:
        st.warning(f"Could not load churn by CLV: {e}")
    
    # Detailed analysis by quartile
    st.subheader("📋 Segment Analysis")
    
    try:
        churn_by_quartile = train_df.groupby('CLV_quartile', observed=True).agg({
            'Churn_encoded': ['count', 'sum', 'mean'],
            'CLV': ['mean', 'median']
        }).round(2)
        
        churn_by_quartile.columns = ['Total_Customers', 'Churned', 'Churn_Rate', 'Avg_CLV', 'Median_CLV']
        churn_by_quartile['Churn_Rate'] = (churn_by_quartile['Churn_Rate'] * 100).round(1)
        
        st.dataframe(churn_by_quartile, width='stretch')
    except Exception as e:
        st.error(f"Could not generate segment analysis: {e}")
    
    # Business takeaway
    st.subheader("💡 Business Takeaway")
    
    try:
        low_clv_churn = train_df[train_df['CLV_quartile'] == 'Low']['Churn_encoded'].mean()
        premium_churn = train_df[train_df['CLV_quartile'] == 'Premium']['Churn_encoded'].mean()
        
        # Handle potential NaN values
        low_clv_churn = low_clv_churn if not pd.isna(low_clv_churn) else 0.0
        premium_churn = premium_churn if not pd.isna(premium_churn) else 0.0
        
        takeaway = f"""
        **Prioritize High/Premium segments for retention.** Low-CLV customers churn at {low_clv_churn*100:.1f}% 
        compared to just {premium_churn*100:.1f}% for Premium customers. However, given the volume of Low/Medium 
        customers, retention efforts should balance:
        
        1. **High-ROI Retention**: Focus on preventing Premium customer churn (highest value at risk)
        2. **Upgrade Programs**: Convert Medium to High via service bundles and upsells
        3. **Efficient Churn Recovery**: For Low-segment, use automated re-engagement or win-back campaigns
        4. **Proactive Monitoring**: Use this app's prediction model to identify at-risk customers early
        """
        
        st.info(takeaway)
    except Exception as e:
        st.warning(f"Could not generate business takeaway: {e}")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><small>
    Customer Churn Prediction & CLV Analysis | 
    Powered by ML (Logistic Regression, Random Forest, XGBoost) | 
    SHAP + Feature Importance Interpretability
    </small></p>
</div>
""", unsafe_allow_html=True)
