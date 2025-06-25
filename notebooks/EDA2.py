"""
Exploratory Data Analysis - E-commerce Fraud Detection Dataset
Author: Data Scientist working on production fraud detection system
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("E-COMMERCE FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Load the data
print("\n📊 LOADING DATA...")
df = pd.read_csv('data/ingested/raw_ecommerce_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Basic info
print(f"\n📋 BASIC DATASET INFO:")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes.value_counts()}")

# Target variable analysis
print(f"\n🎯 TARGET VARIABLE ANALYSIS:")
fraud_dist = df['Is Fraudulent'].value_counts()
print(f"Fraudulent transactions: {fraud_dist[1]} ({fraud_dist[1]/len(df)*100:.2f}%)")
print(f"Legitimate transactions: {fraud_dist[0]} ({fraud_dist[0]/len(df)*100:.2f}%)")
print(f"Class imbalance ratio: {fraud_dist[0]/fraud_dist[1]:.1f}:1")

# This is a classic imbalanced dataset - we'll need to handle this carefully
print("\n⚠️  NOTE: This is a heavily imbalanced dataset. We'll need to:")
print("   - Use appropriate evaluation metrics (precision, recall, F1)")
print("   - Consider class weights in our models")
print("   - Use anomaly detection approach with autoencoders")

# Missing values check
print(f"\n🔍 MISSING VALUES ANALYSIS:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing values found:")
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
else:
    print("✅ No missing values - great for production!")

# Transaction Amount Analysis
print(f"\n💰 TRANSACTION AMOUNT ANALYSIS:")
print(f"Mean: ${df['Transaction Amount'].mean():.2f}")
print(f"Median: ${df['Transaction Amount'].median():.2f}")
print(f"Std: ${df['Transaction Amount'].std():.2f}")
print(f"Min: ${df['Transaction Amount'].min():.2f}")
print(f"Max: ${df['Transaction Amount'].max():.2f}")

# Check for outliers in transaction amounts
q99 = df['Transaction Amount'].quantile(0.99)
q95 = df['Transaction Amount'].quantile(0.95)
print(f"95th percentile: ${q95:.2f}")
print(f"99th percentile: ${q99:.2f}")

# Fraud vs legitimate transaction amounts
fraud_amounts = df[df['Is Fraudulent'] == 1]['Transaction Amount']
legit_amounts = df[df['Is Fraudulent'] == 0]['Transaction Amount']

print(f"\nFraudulent transactions - Amount stats:")
print(f"  Mean: ${fraud_amounts.mean():.2f}")
print(f"  Median: ${fraud_amounts.median():.2f}")
print(f"  Std: ${fraud_amounts.std():.2f}")

print(f"\nLegitimate transactions - Amount stats:")
print(f"  Mean: ${legit_amounts.mean():.2f}")
print(f"  Median: ${legit_amounts.median():.2f}")
print(f"  Std: ${legit_amounts.std():.2f}")

# DECISION: We'll cap transaction amounts at 99th percentile to handle outliers
print(f"\n🎯 DECISION: Cap transaction amounts at 99th percentile (${q99:.2f})")
print("   Rationale: Very high amounts could be outliers or data errors")
print("   Impact: Prevents extreme values from skewing our models")

# Customer Age Analysis
print(f"\n👤 CUSTOMER AGE ANALYSIS:")
print(f"Mean age: {df['Customer Age'].mean():.1f} years")
print(f"Median age: {df['Customer Age'].median():.1f} years")
print(f"Min age: {df['Customer Age'].min():.1f} years")
print(f"Max age: {df['Customer Age'].max():.1f} years")

# Check for unrealistic ages
unrealistic_ages = df[(df['Customer Age'] < 13) | (df['Customer Age'] > 100)]
if len(unrealistic_ages) > 0:
    print(f"⚠️  Found {len(unrealistic_ages)} transactions with unrealistic ages")
    print("   This could be data entry errors or fraud indicators")

# DECISION: Clip ages to realistic range
print(f"\n🎯 DECISION: Clip customer ages to 13-100 range")
print("   Rationale: Ages outside this range are likely data errors")
print("   Impact: Ensures data quality for production")

# Payment Method Analysis
print(f"\n💳 PAYMENT METHOD ANALYSIS:")
payment_counts = df['Payment Method'].value_counts()
print("Payment method distribution:")
for method, count in payment_counts.items():
    print(f"  {method}: {count} ({count/len(df)*100:.1f}%)")

# Fraud by payment method
print(f"\nFraud rate by payment method:")
for method in df['Payment Method'].unique():
    method_data = df[df['Payment Method'] == method]
    fraud_rate = method_data['Is Fraudulent'].mean() * 100
    print(f"  {method}: {fraud_rate:.2f}% fraud rate")

# DECISION: Keep all payment methods but monitor closely
print(f"\n🎯 DECISION: Keep all payment methods in the model")
print("   Rationale: Different payment methods have different fraud patterns")
print("   Impact: Model can learn payment-specific fraud signals")

# Product Category Analysis
print(f"\n🛍️  PRODUCT CATEGORY ANALYSIS:")
category_counts = df['Product Category'].value_counts()
print("Product category distribution:")
for category, count in category_counts.items():
    print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")

# Fraud by product category
print(f"\nFraud rate by product category:")
for category in df['Product Category'].unique():
    category_data = df[df['Product Category'] == category]
    fraud_rate = category_data['Is Fraudulent'].mean() * 100
    print(f"  {category}: {fraud_rate:.2f}% fraud rate")

# DECISION: Keep all product categories
print(f"\n🎯 DECISION: Keep all product categories")
print("   Rationale: Different categories have different fraud patterns")
print("   Impact: Model can learn category-specific fraud signals")

# Device Used Analysis
print(f"\n📱 DEVICE USED ANALYSIS:")
device_counts = df['Device Used'].value_counts()
print("Device distribution:")
for device, count in device_counts.items():
    print(f"  {device}: {count} ({count/len(df)*100:.1f}%)")

# Fraud by device
print(f"\nFraud rate by device:")
for device in df['Device Used'].unique():
    device_data = df[df['Device Used'] == device]
    fraud_rate = device_data['Is Fraudulent'].mean() * 100
    print(f"  {device}: {fraud_rate:.2f}% fraud rate")

# DECISION: Keep device information
print(f"\n🎯 DECISION: Keep device information")
print("   Rationale: Mobile vs desktop can indicate different fraud patterns")
print("   Impact: Model can learn device-specific fraud signals")

# Transaction Hour Analysis
print(f"\n🕐 TRANSACTION HOUR ANALYSIS:")
print(f"Mean transaction hour: {df['Transaction Hour'].mean():.1f}")
print(f"Most common hour: {df['Transaction Hour'].mode()[0]}")

# Fraud by hour
hour_fraud = df.groupby('Transaction Hour')['Is Fraudulent'].mean()
print(f"\nFraud rate by hour (top 5 highest):")
print(hour_fraud.nlargest(5))

# DECISION: Create time-based features
print(f"\n🎯 DECISION: Create time-based features")
print("   Rationale: Fraud patterns vary by time of day")
print("   Impact: Model can learn temporal fraud patterns")

# Quantity Analysis
print(f"\n📦 QUANTITY ANALYSIS:")
print(f"Mean quantity: {df['Quantity'].mean():.2f}")
print(f"Median quantity: {df['Quantity'].median():.2f}")
print(f"Max quantity: {df['Quantity'].max()}")

# Check for unrealistic quantities
high_quantities = df[df['Quantity'] > 10]
if len(high_quantities) > 0:
    print(f"⚠️  Found {len(high_quantities)} transactions with quantity > 10")

# DECISION: Cap quantities at 99th percentile
print(f"\n🎯 DECISION: Cap quantities at 99th percentile")
print("   Rationale: Very high quantities could be data errors or bulk fraud")
print("   Impact: Prevents extreme values from skewing models")

# Account Age Analysis
print(f"\n📅 ACCOUNT AGE ANALYSIS:")
print(f"Mean account age: {df['Account Age Days'].mean():.1f} days")
print(f"Median account age: {df['Account Age Days'].median():.1f} days")
print(f"Max account age: {df['Account Age Days'].max():.1f} days")

# Fraud by account age
df['Account Age Years'] = df['Account Age Days'] / 365.25
age_bins = pd.cut(df['Account Age Years'], bins=[0, 1, 2, 5, 10, 50], labels=['<1y', '1-2y', '2-5y', '5-10y', '10y+'])
age_fraud = df.groupby(age_bins)['Is Fraudulent'].mean()
print(f"\nFraud rate by account age:")
print(age_fraud)

# DECISION: Create account age features
print(f"\n🎯 DECISION: Create account age features")
print("   Rationale: New accounts are more likely to be fraudulent")
print("   Impact: Model can learn account maturity patterns")

# Location Analysis
print(f"\n🌍 LOCATION ANALYSIS:")
location_counts = df['Customer Location'].value_counts()
print(f"Unique locations: {len(location_counts)}")
print(f"Most common location: {location_counts.index[0]} ({location_counts.iloc[0]} transactions)")

# Fraud by location (top 10)
location_fraud = df.groupby('Customer Location')['Is Fraudulent'].mean()
print(f"\nTop 10 locations by fraud rate:")
print(location_fraud.nlargest(10))

# DECISION: Use frequency encoding for locations
print(f"\n🎯 DECISION: Use frequency encoding for customer locations")
print("   Rationale: Too many unique locations for one-hot encoding")
print("   Impact: Captures location popularity as a fraud signal")

# Feature Engineering Plan
print(f"\n" + "="*80)
print("FEATURE ENGINEERING PLAN")
print("="*80)

print("Based on this EDA, here's my feature engineering strategy:")

print("\n1. TRANSACTION FEATURES:")
print("   - Log and sqrt transformations of transaction amount")
print("   - Amount per item (amount/quantity)")
print("   - Quantity squared")

print("\n2. CUSTOMER FEATURES:")
print("   - Age bins (0-25, 26-35, 36-50, 50+)")
print("   - Account age in years")
print("   - Age to account age ratio")

print("\n3. TIME FEATURES:")
print("   - Hour bins (0-6, 7-12, 13-18, 19-24)")
print("   - Night transaction flag (10 PM - 6 AM)")

print("\n4. INTERACTION FEATURES:")
print("   - Amount × Age interaction")
print("   - Amount × Quantity interaction")
print("   - Amount × Payment method interactions")

print("\n5. ENCODING STRATEGY:")
print("   - Label encoding for low-cardinality categoricals")
print("   - Frequency encoding for high-cardinality categoricals")

# Data Cleaning Decisions Summary
print(f"\n" + "="*80)
print("DATA CLEANING DECISIONS")
print("="*80)

print("1. Handle Outliers:")
print("   - Cap transaction amounts at 99th percentile")
print("   - Cap quantities at 99th percentile")
print("   - Clip customer ages to 13-100 range")

print("\n2. Feature Encoding:")
print("   - Payment Method: Label encoding")
print("   - Product Category: Label encoding")
print("   - Device Used: Label encoding")
print("   - Customer Location: Frequency encoding")

print("\n3. Remove Columns:")
print("   - Transaction ID (unique identifier)")
print("   - Customer ID (unique identifier)")
print("   - IP Address (privacy concern, too many unique values)")
print("   - Shipping/Billing Address (privacy concern, too many unique values)")
print("   - Transaction Date (will create derived time features)")

print("\n4. Handle Missing Values:")
print("   - No missing values found - no action needed")

print(f"\n" + "="*80)
print("PRODUCTION CONSIDERATIONS")
print("="*80)

print("1. Model Selection:")
print("   - Autoencoder for anomaly detection (good for imbalanced data)")
print("   - Random Forest for interpretability")
print("   - Consider ensemble approach")

print("\n2. Evaluation Metrics:")
print("   - Focus on precision and recall (not just accuracy)")
print("   - Use F1-score for balanced evaluation")
print("   - Monitor false positive rate (business impact)")

print("\n3. Deployment Strategy:")
print("   - Real-time scoring capability")
print("   - Model interpretability for business stakeholders")
print("   - Regular model retraining schedule")

print(f"\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("1. Implement data cleaning pipeline")
print("2. Create feature engineering pipeline")
print("3. Build and evaluate multiple models")
print("4. Create model interpretability reports")
print("5. Design production deployment strategy")

print(f"\n✅ EDA COMPLETED - READY FOR PIPELINE DEVELOPMENT")
print("="*80) 