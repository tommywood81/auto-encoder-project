import pandas as pd
import numpy as np

# Load the dataset
print("Loading Credit Card Fraud Detection dataset...")
df = pd.read_csv('creditcard.csv')

print(f"\n{'='*60}")
print("DATASET OVERVIEW")
print(f"{'='*60}")
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {len(df):,}")
print(f"Total Features: {len(df.columns)}")

print(f"\n{'='*60}")
print("FEATURE LIST")
print(f"{'='*60}")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n{'='*60}")
print("FEATURE DETAILS")
print(f"{'='*60}")

# Analyze each feature
for col in df.columns:
    dtype = str(df[col].dtype)
    null_count = df[col].isnull().sum()
    unique_count = df[col].nunique()
    
    if dtype in ['int64', 'float64']:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        print(f"{col:15s} | {dtype:8s} | Nulls: {null_count:3d} | Unique: {unique_count:6d} | Range: [{min_val:8.2f}, {max_val:8.2f}] | Mean: {mean_val:8.2f}")
    else:
        print(f"{col:15s} | {dtype:8s} | Nulls: {null_count:3d} | Unique: {unique_count:6d}")

print(f"\n{'='*60}")
print("TARGET DISTRIBUTION")
print(f"{'='*60}")
class_counts = df['Class'].value_counts()
print("Class Distribution:")
for class_val, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  Class {class_val}: {count:8d} records ({percentage:5.2f}%)")

print(f"\n{'='*60}")
print("SAMPLE DATA")
print(f"{'='*60}")
print(df.head(3).to_string())

print(f"\n{'='*60}")
print("FEATURE CATEGORIES")
print(f"{'='*60}")

# Categorize features
time_features = ['Time']
amount_features = ['Amount']
target_features = ['Class']
v_features = [col for col in df.columns if col.startswith('V')]

print(f"Time Features ({len(time_features)}): {time_features}")
print(f"Amount Features ({len(amount_features)}): {amount_features}")
print(f"Target Features ({len(target_features)}): {target_features}")
print(f"PCA Features ({len(v_features)}): V1-V{len(v_features)}")

print(f"\n{'='*60}")
print("STATISTICAL SUMMARY")
print(f"{'='*60}")
print(df.describe().round(2)) 