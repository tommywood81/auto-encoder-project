#!/usr/bin/env python3
"""
Analyze the value of engineered features in the fraud detection model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def analyze_feature_importance():
    """Analyze feature importance to see if engineered features add value."""
    print("Analyzing feature importance...")
    
    # Load engineered data
    df = pd.read_csv('data/engineered/train_features.csv')
    
    # Define engineered features
    engineered_features = [
        'amount_log', 'amount_sqrt', 'card_count', 'email_count', 'addr_count',
        'v_features_count', 'v_features_mean', 'v_features_std', 'v_features_sum', 
        'id_features_count', 'amount_card_interaction', 'amount_email_interaction',
        'v_count_mean_interaction', 'v_features_q25', 'v_features_q75', 
        'v_features_iqr', 'v_features_range'
    ]
    
    # Check which engineered features exist
    existing_engineered = [f for f in engineered_features if f in df.columns]
    print(f"Found {len(existing_engineered)} engineered features: {existing_engineered}")
    
    # Prepare data
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Analyze engineered features
    engineered_importance = feature_importance[
        feature_importance['feature'].isin(existing_engineered)
    ].copy()
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    print(f"\nEngineered Features Ranked by Importance:")
    print(engineered_importance)
    
    # Calculate metrics
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nRandom Forest Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Check if engineered features are in top features
    top_20_features = feature_importance.head(20)['feature'].tolist()
    engineered_in_top_20 = [f for f in existing_engineered if f in top_20_features]
    
    print(f"\nEngineered features in top 20: {len(engineered_in_top_20)}")
    if engineered_in_top_20:
        print(f"Features: {engineered_in_top_20}")
    
    return feature_importance, engineered_importance

def compare_with_and_without_engineered():
    """Compare model performance with and without engineered features."""
    print("\n" + "="*50)
    print("COMPARING WITH AND WITHOUT ENGINEERED FEATURES")
    print("="*50)
    
    # Load data
    df = pd.read_csv('data/engineered/train_features.csv')
    
    # Define engineered features
    engineered_features = [
        'amount_log', 'amount_sqrt', 'card_count', 'email_count', 'addr_count',
        'v_features_count', 'v_features_mean', 'v_features_std', 'v_features_sum', 
        'id_features_count', 'amount_card_interaction', 'amount_email_interaction',
        'v_count_mean_interaction', 'v_features_q25', 'v_features_q75', 
        'v_features_iqr', 'v_features_range'
    ]
    
    existing_engineered = [f for f in engineered_features if f in df.columns]
    
    # Prepare data with engineered features
    X_with = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # Prepare data without engineered features
    X_without = X_with.drop(columns=existing_engineered)
    
    print(f"Features with engineered: {X_with.shape[1]}")
    print(f"Features without engineered: {X_without.shape[1]}")
    
    # Test both models
    results = {}
    
    for name, X_data in [("With Engineered", X_with), ("Without Engineered", X_without)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y, test_size=0.2, random_state=42, stratify=y
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
    
    # Compare results
    print(f"\nPerformance Comparison:")
    print(f"{'Metric':<15} {'With Engineered':<15} {'Without Engineered':<15} {'Difference':<15}")
    print("-" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        with_val = results["With Engineered"][metric]
        without_val = results["Without Engineered"][metric]
        diff = with_val - without_val
        
        print(f"{metric:<15} {with_val:<15.4f} {without_val:<15.4f} {diff:<15.4f}")
    
    return results

def main():
    """Main analysis function."""
    print("FEATURE ENGINEERING ANALYSIS")
    print("="*50)
    
    # Analyze feature importance
    feature_importance, engineered_importance = analyze_feature_importance()
    
    # Compare performance
    results = compare_with_and_without_engineered()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    # Check if engineered features are valuable
    with_engineered = results["With Engineered"]
    without_engineered = results["Without Engineered"]
    
    improvements = {
        'accuracy': with_engineered['accuracy'] - without_engineered['accuracy'],
        'precision': with_engineered['precision'] - without_engineered['precision'],
        'recall': with_engineered['recall'] - without_engineered['recall'],
        'f1': with_engineered['f1'] - without_engineered['f1']
    }
    
    print("Are engineered features valuable?")
    for metric, improvement in improvements.items():
        if improvement > 0.001:  # Significant improvement
            print(f"✓ {metric}: +{improvement:.4f} improvement")
        elif improvement < -0.001:  # Significant degradation
            print(f"✗ {metric}: {improvement:.4f} degradation")
        else:
            print(f"- {metric}: No significant change ({improvement:.4f})")
    
    # Check if any engineered features are in top 20
    top_20_features = feature_importance.head(20)['feature'].tolist()
    engineered_in_top_20 = [f for f in engineered_importance['feature'].tolist() if f in top_20_features]
    
    if engineered_in_top_20:
        print(f"\n✓ {len(engineered_in_top_20)} engineered features are in top 20 most important")
        print(f"  Features: {engineered_in_top_20}")
    else:
        print(f"\n✗ No engineered features in top 20 most important")

if __name__ == "__main__":
    main() 