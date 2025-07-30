# Credit Card Fraud Detection - EDA Summary

## Overview
This EDA analyzes the credit card fraud dataset with 284,807 transactions, focusing on key patterns and insights for fraud detection.

## Key Statistics
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.17%)
- **Dataset Duration**: 48 hours
- **Average Transaction**: $80.19

## Generated Visualizations

### 1. Fraud Distribution (`fraud_distribution.png`)
**What it shows**: Pie chart and bar plot of legitimate vs fraudulent transactions
**Key Insight**: Extreme class imbalance - only 0.17% of transactions are fraudulent, which is typical for fraud detection datasets and requires special handling in model training.

### 2. Amount Distribution (`amount_distribution.png`)
**What it shows**: Histograms comparing transaction amounts for legitimate vs fraudulent transactions
**Key Insight**: Fraudulent transactions tend to be smaller amounts, with most transactions under $100. Legitimate transactions show higher variance in amounts.

### 3. Temporal Patterns (`temporal_patterns.png`)
**What it shows**: Fraud rate and transaction volume by hour of day
**Key Insight**: Fraud rate varies throughout the day, and peak transaction volume doesn't correlate with peak fraud rate. Time-based features could be valuable for fraud detection.

### 4. Correlation Heatmap (`correlation_heatmap.png`)
**What it shows**: Correlation matrix between key features (amount, time, V1-V14, fraud label)
**Key Insight**: Most V-features show low correlation with each other, suggesting they are well-engineered PCA components. Amount has minimal correlation with fraud.

### 5. Feature Importance (`feature_importance.png`)
**What it shows**: Top features ranked by absolute correlation with fraud
**Key Insight**: V-features dominate the top correlations with fraud, while amount has relatively low correlation. Time shows moderate correlation with fraud.

## Key Findings

1. **Extreme Class Imbalance**: Only 0.17% of transactions are fraudulent, requiring special handling in model training (class weights, sampling techniques, etc.)

2. **Amount Patterns**: Fraudulent transactions tend to be smaller amounts, with most transactions under $100. This suggests fraudsters may test with small amounts first.

3. **Temporal Patterns**: Fraud rate varies throughout the day, indicating that time-based features could be valuable for fraud detection.

4. **Feature Engineering**: V-features (PCA components) are the most predictive of fraud, while raw amount has minimal correlation.

5. **Low Feature Correlation**: Most features show low correlation with each other, suggesting good feature engineering and no multicollinearity issues.

## Implications for Model Development

- **Class Imbalance**: Need to use techniques like class weights, SMOTE, or focal loss
- **Feature Selection**: Focus on V-features and temporal features
- **Amount Handling**: Consider log transformation or binning for amount features
- **Temporal Features**: Include hour-of-day and time-based features
- **Validation Strategy**: Use stratified sampling and appropriate metrics (AUC-ROC, F1-score)

## Files Generated
- `credit_card_fraud_eda.py`: Main EDA script
- `results/fraud_distribution.png`: Fraud vs legitimate distribution
- `results/amount_distribution.png`: Transaction amount analysis
- `results/temporal_patterns.png`: Time-based patterns
- `results/correlation_heatmap.png`: Feature correlations
- `results/feature_importance.png`: Feature importance ranking 