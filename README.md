# E-commerce Fraud Detection with Autoencoders

This project demonstrates a production-ready fraud detection system using autoencoders for anomaly detection. The system is designed to identify fraudulent e-commerce transactions using unsupervised learning techniques.

## Dataset Overview

We use the **E-commerce Fraud Detection Dataset** containing 23,634 transactions with a 5.17% fraud rate (1,222 fraudulent transactions). This represents a realistic scenario where fraud is rare but costly.

### Dataset Characteristics
- **Size**: 23,634 transactions × 16 features
- **Fraud Rate**: 5.17% (1,222 fraudulent, 22,412 legitimate)
- **Class Imbalance**: 18.3:1 ratio (legitimate:fraudulent)
- **Memory Usage**: 19.2 MB
- **Data Quality**: No missing values (production-ready)

![Fraud Distribution](docs/fraud_distribution.png)

## Data Cleaning Decisions & Analysis

### 1. Transaction Amount Analysis

```python
# Key Statistics
Mean: $229.37
Median: $151.41
Std: $282.05
99th percentile: $1,162.04

# Fraud vs Legitimate Patterns
Fraudulent transactions: Mean $562.08, Median $251.03
Legitimate transactions: Mean $211.23, Median $148.20
```

**Decision**: Use robust scaling and normalization instead of capping
- **Rationale**: Capping at percentiles breaks model generalizability and inference
- **Impact**: Model can handle new high-value products without retraining
- **Business Value**: Maintains model performance when business introduces new products

**Logical Approach**: Instead of capping, we implement:
1. **Log transformation**: `log(amount + 1)` to handle skewed distributions
2. **Robust scaling**: Use median and IQR instead of mean and std
3. **Feature engineering**: Create amount-based ratios and interactions
4. **Domain-specific thresholds**: Set business rules for suspicious amounts
5. **Model interpretability**: Ensure high-value transactions can be explained

**Why not capping?**: If the business introduces a new $50,000 product, capping would:
- Flag all legitimate high-value sales as fraud
- Require immediate model retraining
- Break inference on new transaction patterns
- Reduce model generalizability

**Skewness Analysis**: The original transaction amount data is highly skewed (skewness: 6.697), with a long right tail. Log transformation reduces skewness to -0.228, making the data more suitable for modeling.

![Transaction Amount Skewness Analysis](docs/transaction_amount_skewness_analysis.png)

![Transaction Amount Analysis](docs/transaction_amount_analysis.png)

### 2. Customer Age Analysis

```python
# Age Distribution
Mean age: 34.6 years
Median age: 35.0 years
Range: -2.0 to 73.0 years

# Data Quality Issues
Found 331 transactions with unrealistic ages (negative or >100)
```

**Decision**: Filter to customers 18+ years old and clip to 18-100 range
- **Rationale**: Customers under 18 can be issued debit cards but present different fraud patterns
- **Impact**: Ensures data quality for production
- **Business Value**: Focuses on adult customers with more predictable fraud patterns

![Customer Age Analysis](docs/customer_age_analysis.png)


**Note**: Customers aged 13-17 can be issued debit cards, but these cards are more likely to be stolen or lost at school, presenting a different fraud detection challenge. For this portfolio piece, we focus on adult customers (18+) to simplify the fraud detection model.

### 3. Payment Method Analysis

```python
# Distribution
debit card: 25.2% (4.79% fraud rate)
credit card: 25.1% (5.08% fraud rate)
PayPal: 25.0% (5.26% fraud rate)
bank transfer: 24.8% (5.56% fraud rate)
```

**Decision**: Keep all payment methods in the model
- **Rationale**: Different payment methods have different fraud patterns
- **Impact**: Model can learn payment-specific fraud signals
- **Business Value**: Captures nuanced fraud patterns across payment types

![Payment Method Analysis](docs/payment_method_analysis.png)

### 4. Temporal Patterns

```python
# High Fraud Hours (>8% fraud rate)
Hours 0, 1, 2, 3, 4, 5 (early morning)
Most common transaction hour: 0 (midnight)
```

**Decision**: Create time-based features
- **Rationale**: Fraud patterns vary by time of day
- **Impact**: Model can learn temporal fraud patterns
- **Business Value**: Enables real-time fraud detection based on time patterns

![Temporal Patterns](docs/temporal_patterns.png)

### 5. Location Analysis

```python
# Geographic Distribution
Unique locations: 14,868
Most common: North Michael (30 transactions)
High fraud locations: Multiple locations with 100% fraud rate
```

**Decision**: Use frequency encoding for customer locations
- **Rationale**: Too many unique locations for one-hot encoding
- **Impact**: Captures location popularity as a fraud signal
- **Business Value**: Handles high cardinality while preserving geographic patterns

## Dataset Summary

![Dataset Summary](docs/dataset_summary.png)

## Feature Engineering Strategy

Based on our EDA, we implement the following feature engineering approach:

### 1. Transaction Features
- Log and sqrt transformations of transaction amount
- Amount per item (amount/quantity)
- Quantity squared

### 2. Customer Features
- Age bins (0-25, 26-35, 36-50, 50+)
- Account age in years
- Age to account age ratio

### 3. Time Features
- Hour bins (0-6, 7-12, 13-18, 19-24)
- Night transaction flag (10 PM - 6 AM)

### 4. Interaction Features
- Amount × Age interaction
- Amount × Quantity interaction
- Amount × Payment method interactions

### 5. Encoding Strategy
- Label encoding for low-cardinality categoricals
- Frequency encoding for high-cardinality categoricals

## Data Cleaning Pipeline

### Columns Removed
- **Transaction ID**: Unique identifier, not useful for modeling
- **Customer ID**: Unique identifier, not useful for modeling
- **IP Address**: Privacy concern, too many unique values
- **Shipping/Billing Address**: Privacy concern, too many unique values
- **Transaction Date**: Will be replaced by derived time features

### Data Quality Improvements
-  No missing values found
-  Robust scaling and log transformation for transaction amounts
-  Filtered to customers 18+ years old
-  Capped quantities at 99th percentile
-  Clipped account ages to max 10 years
-  Ensured transaction hours in 0-23 range

### Final Dataset
- **Shape**: 22,580 rows × 11 columns
- **Memory**: 2.0 MB (optimized)
- **Data Types**: 2 float64, 8 int64, 1 object
- **Encoding**: Label encoding for categoricals, frequency encoding for locations

## Production Considerations

### Model Selection
- **Autoencoder**: For anomaly detection (good for imbalanced data)
- **Random Forest**: For interpretability
- **Ensemble Approach**: Consider combining multiple models

### Evaluation Metrics
- **Primary**: Precision, Recall, F1-score (not just accuracy)
- **Secondary**: ROC AUC for model comparison
- **Business**: False positive rate monitoring

### Deployment Strategy
- Real-time scoring capability
- Model interpretability for business stakeholders
- Regular model retraining schedule
- Comprehensive logging and monitoring

## Project Architecture

```
data/
 raw/                    # Original dataset
 ingested/              # Processed raw data
 cleaned/               # Cleaned dataset
 processed/             # Feature-engineered data

src/
 ingest_data.py         # Data ingestion
 data_cleaning.py       # Data cleaning pipeline
 feature_engineering.py # Feature creation
 feature_factory.py     # Feature orchestration
 autoencoder.py         # Autoencoder model
 train.py              # Training pipeline
 evaluate.py           # Model evaluation
 config.py             # Configuration management
```

## Usage

### Running the Pipeline
```bash
# Run complete pipeline
python run_pipeline.py

# Run individual stages
python -c "from src.data_cleaning import DataCleaner; from src.config import PipelineConfig; cleaner = DataCleaner(PipelineConfig()); cleaner.clean_data()"
```

### Testing
```bash
# Test data cleaning
python test_cleaning.py

# Test feature engineering
python test_feature_engineering.py
```

### Creating Visualizations
```bash
# Generate all visualizations for the README
python create_visualizations.py
```

## Key Insights

1. **Fraud Patterns**: Early morning hours (0-5 AM) show highest fraud rates
2. **Payment Risk**: Bank transfers have slightly higher fraud rates
3. **Amount Patterns**: Fraudulent transactions tend to be higher value
4. **Age Issues**: 331 transactions had unrealistic ages (data quality concern)
5. **Geographic Spread**: Fraud is distributed across many locations

## Next Steps

1.  Implement data cleaning pipeline
2.  Create feature engineering pipeline
3.  Build and evaluate multiple models
4.  Create model interpretability reports
5.  Design production deployment strategy

---

> This project demonstrates a production-ready fraud detection system with comprehensive data cleaning, feature engineering, and model development. The decisions made are based on real-world considerations including privacy, business impact, and model performance. 