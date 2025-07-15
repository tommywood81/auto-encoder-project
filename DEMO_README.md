# Autoencoder Fraud Detection Demo Dashboard

A clean, explainable, and interactive dashboard demonstrating how an unsupervised autoencoder detects anomalies (fraud) in transaction data.

## 🎯 Demo Features

### 📅 Date Selector
- Dropdown with all available dates from test set
- "All Dates" option for complete dataset view
- Filters table and metrics by selected date

### 🎚️ Threshold Slider
- Range: 80 to 100 (percentile-based)
- Default: 97 (trained model threshold)
- Label: "Fraud Sensitivity Threshold"
- Dynamically adjusts fraud detection sensitivity

### 📊 Metrics Panel
Real-time metrics based on date selection and current threshold:
- **Total Transactions** - All transactions in selected date
- **Total Ground Truth Frauds** - Actual frauds in data
- **Flagged Transactions** - Model predictions
- **✅ Correctly Flagged Frauds** - True Positives
- **❌ Missed Frauds** - False Negatives  
- **⚠️ False Positives** - Flagged but not fraud
- **AUC ROC** - Model performance (0.73)

### 🧾 Flagged Transactions Table
When a user selects a date, only the transactions flagged as potential fraud are shown:

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique transaction identifier |
| `customer_location` | Customer ID/location |
| `amount` | Transaction amount |
| `payment_method` | How the transaction was paid |
| `product_category` | Type of product/service |
| `customer_age` | Customer's age |
| `device_used` | Device used for transaction |
| `reconstruction_error` | Autoencoder reconstruction error |
| `is_actual_fraud` | Whether it was actually fraud |
| `fraud_indicator` | Clear visual indicator |
| `fraud_status` | True Positive or False Positive |

#### Fraud Indicators:
- **✅ "ACTUAL FRAUD"** - Model correctly flagged an actual fraud (True Positive)
- **⚠️ "FALSE ALARM"** - Model flagged but it wasn't actually fraud (False Positive)

#### Visual Styling:
- **Green row**: Actual fraud correctly caught
- **Yellow row**: False alarm (flagged but not fraud)

### 🔍 Toggle Filters
Filter the table to show only:
- All rows (default)
- Only actual frauds
- Only flagged transactions
- Only missed frauds
- Only false positives

### ℹ️ Information Modal
Explains the autoencoder model, threshold tuning, and AUC-ROC performance.

## 🧠 Why an Autoencoder?

**Unsupervised Learning Benefits:**
- Works without labeled fraud data
- Detects anomalies, not just known patterns
- Ideal for bootstrapping fraud detection systems
- Handles missing, late, or inaccurate fraud labels

## 📈 Understanding AUC ROC (0.73)

- **What it means**: Ability to rank frauds above non-frauds
- **Current performance**: 73% accuracy in ranking
- **Improvement potential**: Better features and retraining

## 🎯 Understanding the Threshold

- **Trained at**: 97th percentile (top 3% reconstruction error)
- **Slider range**: 80-100 for dynamic exploration
- **Purpose**: Trade-off between catching frauds vs. false positives

## 🚀 Improving the Model

This demo shows the foundation. The model can be improved through:

1. **Richer Features**
   - Time-based patterns
   - Device fingerprinting
   - Behavioral bursts
   - Location mismatches

2. **Better Training**
   - More data
   - Semi-supervised learning
   - Ensemble models

3. **Business Impact**
   - More frauds caught
   - Fewer false alarms
   - Significant cost savings

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- Python 3.8+

### Deploy Demo
```bash
# Activate virtual environment
env\Scripts\Activate.ps1

# Deploy demo
python deploy_local.py

# Access dashboard
# http://localhost:5000
```

### Stop Demo
```bash
python deploy_local.py --stop
```

## 📊 API Endpoints

### Main Dashboard Endpoint
```
POST /api/predict
{
  "date": "2023-01-15",  // or "All Dates"
  "threshold": 97.0       // 80-100 range
}
```

### Response Format
```json
{
  "date": "2024-01-01",
  "threshold": 95.0,
  "metrics": {
    "total_transactions": 255,
    "total_ground_truth_frauds": 18,
    "flagged_transactions": 13,
    "correctly_flagged_frauds": 2,
    "missed_frauds": 16,
    "false_positives": 11,
    "auc_roc": 0.73
  },
  "flagged_transactions": [
    {
      "transaction_id": "TXN_000001",
      "customer_location": "US",
      "amount": 150.00,
      "payment_method": "Credit Card",
      "product_category": "Electronics",
      "customer_age": 35,
      "device_used": "Mobile",
      "reconstruction_error": 0.85,
      "is_actual_fraud": true,
      "fraud_indicator": "✅ ACTUAL FRAUD",
      "fraud_status": "True Positive"
    },
    {
      "transaction_id": "TXN_000002",
      "customer_location": "CA",
      "amount": 75.50,
      "payment_method": "Debit Card",
      "product_category": "Clothing",
      "customer_age": 28,
      "device_used": "Desktop",
      "reconstruction_error": 0.78,
      "is_actual_fraud": false,
      "fraud_indicator": "⚠️ FALSE ALARM",
      "fraud_status": "False Positive"
    }
  ]
}
```

## 🔧 Technical Details

### Model Architecture
- **Type**: Autoencoder (unsupervised)
- **Strategy**: Combined feature engineering
- **Features**: 15+ engineered features
- **Threshold**: 97th percentile reconstruction error

### Data Processing
- **Input**: Transaction features
- **Output**: Reconstruction error scores
- **Normalization**: Percentile-based scoring
- **Threshold**: Dynamic percentile-based filtering

### Performance
- **AUC-ROC**: 0.73
- **Training**: Unsupervised on clean data
- **Inference**: Real-time scoring
- **Scalability**: Handles thousands of transactions

## 📈 Business Value

### Cost Savings
- **Fraud Prevention**: Catch fraudulent transactions
- **False Positive Reduction**: Minimize legitimate transaction blocks
- **Operational Efficiency**: Automated review queue prioritization

### Risk Management
- **Real-time Detection**: Immediate fraud flagging
- **Threshold Tuning**: Balance sensitivity vs. specificity
- **Transparency**: Full explainability of decisions

### Continuous Improvement
- **Model Retraining**: Incorporate new patterns
- **Feature Engineering**: Add domain-specific features
- **Performance Monitoring**: Track AUC-ROC improvements

---

**Demo Dashboard**: http://localhost:5000  
**API Documentation**: http://localhost:5000/docs  
**Health Check**: http://localhost:5000/health 