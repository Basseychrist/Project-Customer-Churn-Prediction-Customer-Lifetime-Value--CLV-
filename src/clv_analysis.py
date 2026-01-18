"""
CLV Analysis Module

Analyzes Customer Lifetime Value by:
- Computing CLV per customer
- Creating quartile segments
- Analyzing churn rates by segment
- Generating business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_processed_data():
    """Load processed train/val/test splits."""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    return train, val, test


def analyze_clv_distribution(df, title="CLV Distribution"):
    """
    Analyze and visualize CLV distribution.
    
    Args:
        df: DataFrame with CLV column
        title: Title for visualization
        
    Returns:
        dict: Summary statistics
    """
    stats = {
        'mean': df['CLV'].mean(),
        'median': df['CLV'].median(),
        'std': df['CLV'].std(),
        'min': df['CLV'].min(),
        'max': df['CLV'].max(),
        'q25': df['CLV'].quantile(0.25),
        'q75': df['CLV'].quantile(0.75)
    }
    
    print(f"\n{title} Statistics:")
    print(f"Mean: ${stats['mean']:.2f}")
    print(f"Median: ${stats['median']:.2f}")
    print(f"Std Dev: ${stats['std']:.2f}")
    print(f"Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
    
    return stats


def churn_by_clv_quartile(df, output_dir='data/processed'):
    """
    Analyze churn rate by CLV quartile.
    
    Args:
        df: DataFrame with CLV and CLV_quartile columns
        output_dir: Directory to save analysis
        
    Returns:
        pd.DataFrame: Churn analysis by quartile
    """
    # Ensure CLV_quartile exists
    if 'CLV_quartile' not in df.columns:
        df['CLV_quartile'] = pd.qcut(df['CLV'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    
    churn_by_quartile = df.groupby('CLV_quartile', observed=True).agg({
        'Churn_encoded': ['count', 'sum', 'mean'],
        'CLV': ['mean', 'median']
    }).round(3)
    
    churn_by_quartile.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate', 
                                  'Avg_CLV', 'Median_CLV']
    churn_by_quartile['Churn_Rate'] = (churn_by_quartile['Churn_Rate'] * 100).round(2)
    churn_by_quartile['Churn_Rate_Pct'] = churn_by_quartile['Churn_Rate'].astype(str) + '%'
    
    print("\n" + "="*80)
    print("CHURN ANALYSIS BY CLV QUARTILE")
    print("="*80)
    print(churn_by_quartile.to_string())
    
    return churn_by_quartile


def business_insights(df):
    """
    Generate key business insights from CLV and churn data.
    
    Args:
        df: DataFrame with CLV and churn data
        
    Returns:
        list: List of insight strings
    """
    insights = []
    
    # Insight 1: Low CLV customers have high churn
    low_clv_churn = df[df['CLV_quartile'] == 'Low']['Churn_encoded'].mean()
    premium_churn = df[df['CLV_quartile'] == 'Premium']['Churn_encoded'].mean()
    
    insight1 = (
        f"💔 **Churn Concentration**: Low-CLV customers churn at {low_clv_churn*100:.1f}% vs "
        f"Premium at {premium_churn*100:.1f}%. "
        f"Focusing retention efforts on High/Premium segments offers better ROI."
    )
    insights.append(insight1)
    
    # Insight 2: CLV by tenure and contract
    if 'Contract' in df.columns:
        contract_clv = df.groupby('Contract')['CLV'].agg(['mean', 'count'])
        best_contract = contract_clv['mean'].idxmax()
        worst_contract = contract_clv['mean'].idxmin()
        
        insight2 = (
            f"📋 **Contract Impact**: Customers on '{best_contract}' contracts have "
            f"${contract_clv.loc[best_contract, 'mean']:.0f} average CLV vs "
            f"${contract_clv.loc[worst_contract, 'mean']:.0f} for '{worst_contract}' contracts. "
            f"Upgrade efforts on short-contract customers could significantly increase retention."
        )
        insights.append(insight2)
    
    # Insight 3: Services and CLV
    if 'services_count' in df.columns:
        high_service_churn = df[df['services_count'] >= 4]['Churn_encoded'].mean()
        low_service_churn = df[df['services_count'] < 2]['Churn_encoded'].mean()
        
        insight3 = (
            f"🔧 **Service Bundle Effect**: Customers with 4+ services churn at {high_service_churn*100:.1f}% vs "
            f"{low_service_churn*100:.1f}% for those with <2 services. "
            f"Cross-selling relevant service bundles is a key retention lever."
        )
        insights.append(insight3)
    
    return insights


def create_visualizations(df, output_dir='figures'):
    """
    Create and save CLV analysis visualizations.
    
    Args:
        df: DataFrame with CLV and churn data
        output_dir: Directory to save figures
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Figure 1: CLV Distribution by Quartile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['CLV'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['CLV'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ${df['CLV'].mean():.0f}")
    axes[0].axvline(df['CLV'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: ${df['CLV'].median():.0f}")
    axes[0].set_xlabel('CLV ($)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
    axes[0].set_title('CLV Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Quartile box plot
    df['CLV_quartile'] = pd.qcut(df['CLV'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    df.boxplot(column='CLV', by='CLV_quartile', ax=axes[1])
    axes[1].set_xlabel('CLV Quartile', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('CLV ($)', fontsize=11, fontweight='bold')
    axes[1].set_title('CLV by Quartile', fontsize=12, fontweight='bold')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/clv_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/clv_distribution.png")
    
    # Figure 2: Churn Rate by CLV Quartile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    churn_data = df.groupby('CLV_quartile', observed=True)['Churn_encoded'].mean() * 100
    
    # Bar chart
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    axes[0].bar(churn_data.index, churn_data.values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('CLV Quartile', fontsize=11, fontweight='bold')
    axes[0].set_title('Churn Rate by CLV Quartile', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(churn_data.items()):
        axes[0].text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Customer count by quartile
    customer_counts = df['CLV_quartile'].value_counts().sort_index()
    axes[1].bar(customer_counts.index, customer_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('CLV Quartile', fontsize=11, fontweight='bold')
    axes[1].set_title('Customer Count by CLV Quartile', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/churn_by_clv.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/churn_by_clv.png")
    
    plt.close('all')


def run_clv_analysis(output_dir='data/processed'):
    """
    Run complete CLV analysis pipeline.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        tuple: (churn_analysis_df, insights_list)
    """
    print("\n" + "="*80)
    print("CUSTOMER LIFETIME VALUE (CLV) ANALYSIS")
    print("="*80)
    
    # Load processed data (use training set for analysis)
    train, _, _ = load_processed_data()
    
    # Analyze CLV distribution
    analyze_clv_distribution(train, "Training Set CLV")
    
    # Analyze churn by CLV quartile
    churn_by_quartile = churn_by_clv_quartile(train, output_dir)
    
    # Generate insights
    insights = business_insights(train)
    
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS")
    print("="*80)
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight}")
    
    # Create visualizations
    create_visualizations(train, 'figures')
    
    return churn_by_quartile, insights


if __name__ == '__main__':
    churn_analysis, business_insights = run_clv_analysis()
