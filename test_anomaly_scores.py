#!/usr/bin/env python3
"""
Test Anomaly Scores Calculator

This script calculates the min, max, and mean anomaly scores (reconstruction errors)
for the entire test dataset using the trained autoencoder model.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yaml
from src.config import PipelineConfig
from src.feature_factory import FeatureFactory

def load_model_and_data():
    """Load the trained model, scaler, and test data."""
    # Load configuration
    config = PipelineConfig.get_combined_config()
    
    # File paths
    model_path = "models/final_model.h5"
    scaler_path = "models/final_model_scaler.pkl"
    model_info_path = "models/final_model_info.yaml"
    data_path = os.path.join(config.data.cleaned_dir, "ecommerce_cleaned.csv")
    
    print("Loading model and data...")
    
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✓ Model loaded from {model_path}")
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    print(f"✓ Scaler loaded from {scaler_path}")
    
    # Load model info
    with open(model_info_path, 'r') as file:
        model_info = yaml.safe_load(file)
    print(f"✓ Model info loaded from {model_info_path}")
    
    # Load and process data
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {len(df)} transactions from {data_path}")
    
    # Generate features
    feature_engineer = FeatureFactory.create("combined")
    df_features = feature_engineer.generate_features(df)
    
    # Select numeric features
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    
    # Scale features
    scaled_features = scaler.transform(df_numeric)
    print(f"✓ Features scaled: {scaled_features.shape}")
    
    return model, scaled_features, model_info, df

def calculate_anomaly_scores(model, scaled_features):
    """Calculate reconstruction errors (anomaly scores) for all transactions."""
    print("Calculating anomaly scores...")
    
    # Get reconstructions for all transactions
    reconstructions = model.predict(scaled_features, verbose=0)
    
    # Calculate MSE (reconstruction error) for each transaction
    mse_scores = np.mean(np.square(scaled_features - reconstructions), axis=1)
    
    print(f"✓ Anomaly scores calculated for {len(mse_scores)} transactions")
    
    return mse_scores

def calculate_model_threshold(mse_scores, model_info):
    """Calculate the model threshold using the same method as during training."""
    print("\n" + "="*60)
    print("MODEL THRESHOLD CALCULATION")
    print("="*60)
    
    # Get threshold percentile from model info
    threshold_percentile = model_info.get('threshold_percentile', 95)
    print(f"Threshold percentile from model info: {threshold_percentile}%")
    
    # Calculate threshold using numpy percentile
    threshold = np.percentile(mse_scores, threshold_percentile)
    print(f"Calculated threshold: {threshold:.6f}")
    
    # Show the calculation process
    print(f"\nCalculation Details:")
    print(f"  - Total transactions: {len(mse_scores)}")
    print(f"  - Percentile: {threshold_percentile}%")
    print(f"  - Index for {threshold_percentile}th percentile: {threshold_percentile/100 * len(mse_scores):.0f}")
    
    # Sort scores to show the exact calculation
    sorted_scores = np.sort(mse_scores)
    index = int(threshold_percentile / 100 * len(mse_scores))
    if index >= len(sorted_scores):
        index = len(sorted_scores) - 1
    
    print(f"  - Sorted scores[{index}]: {sorted_scores[index]:.6f}")
    print(f"  - Final threshold: {threshold:.6f}")
    
    # Verify the calculation
    above_threshold = np.sum(mse_scores > threshold)
    below_threshold = np.sum(mse_scores <= threshold)
    
    print(f"\nVerification:")
    print(f"  - Transactions above threshold: {above_threshold} ({above_threshold/len(mse_scores)*100:.1f}%)")
    print(f"  - Transactions below threshold: {below_threshold} ({below_threshold/len(mse_scores)*100:.1f}%)")
    print(f"  - Expected above threshold: {len(mse_scores) * (100-threshold_percentile)/100:.0f} ({100-threshold_percentile:.1f}%)")
    
    return threshold

def analyze_anomaly_scores(mse_scores, model_info):
    """Analyze the distribution of anomaly scores."""
    print("\n" + "="*60)
    print("ANOMALY SCORE ANALYSIS")
    print("="*60)
    
    # Basic statistics
    min_score = np.min(mse_scores)
    max_score = np.max(mse_scores)
    mean_score = np.mean(mse_scores)
    median_score = np.median(mse_scores)
    std_score = np.std(mse_scores)
    
    print(f"Minimum Anomaly Score:  {min_score:.6f}")
    print(f"Maximum Anomaly Score:  {max_score:.6f}")
    print(f"Mean Anomaly Score:     {mean_score:.6f}")
    print(f"Median Anomaly Score:   {median_score:.6f}")
    print(f"Standard Deviation:     {std_score:.6f}")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(mse_scores, p)
        print(f"  {p}th percentile: {value:.6f}")
    
    # Calculate threshold using the same method as the model
    threshold = calculate_model_threshold(mse_scores, model_info)
    
    # Count transactions above threshold
    above_threshold = np.sum(mse_scores > threshold)
    below_threshold = np.sum(mse_scores <= threshold)
    
    print(f"\nModel Threshold ({model_info.get('threshold_percentile', 95)}th percentile): {threshold:.6f}")
    print(f"Transactions above threshold: {above_threshold} ({above_threshold/len(mse_scores)*100:.1f}%)")
    print(f"Transactions below threshold: {below_threshold} ({below_threshold/len(mse_scores)*100:.1f}%)")
    
    return {
        'min': min_score,
        'max': max_score,
        'mean': mean_score,
        'median': median_score,
        'std': std_score,
        'threshold': threshold,
        'above_threshold': above_threshold,
        'below_threshold': below_threshold,
        'percentiles': {p: np.percentile(mse_scores, p) for p in percentiles}
    }

def write_detailed_analysis_to_file(mse_scores, stats, model_info, df):
    """Write comprehensive analysis to file."""
    results_file = "anomaly_score_analysis.txt"
    
    with open(results_file, 'w') as f:
        # Header
        f.write("Autoencoder Anomaly Score Analysis\n")
        f.write("="*60 + "\n\n")
        
        # Model Information
        f.write("MODEL INFORMATION:\n")
        f.write("-"*20 + "\n")
        f.write(f"Model Type: {model_info.get('model_type', 'Unknown')}\n")
        f.write(f"Feature Strategy: {model_info.get('feature_strategy', 'Unknown')}\n")
        f.write(f"Latent Dimensions: {model_info.get('latent_dim', 'Unknown')}\n")
        f.write(f"Training Date: {model_info.get('training_date', 'Unknown')}\n")
        f.write(f"Threshold Percentile: {model_info.get('threshold_percentile', 95)}%\n\n")
        
        # Dataset Information
        f.write("DATASET INFORMATION:\n")
        f.write("-"*20 + "\n")
        f.write(f"Total Transactions: {len(df):,}\n")
        f.write(f"Features Used: {mse_scores.shape[0]} transactions × {df.select_dtypes(include=[np.number]).shape[1]} features\n")
        f.write(f"Data Source: {os.path.join('data/cleaned', 'ecommerce_cleaned.csv')}\n\n")
        
        # Basic Statistics
        f.write("BASIC STATISTICS:\n")
        f.write("-"*20 + "\n")
        f.write(f"Minimum Anomaly Score:  {stats['min']:.6f}\n")
        f.write(f"Maximum Anomaly Score:  {stats['max']:.6f}\n")
        f.write(f"Mean Anomaly Score:     {stats['mean']:.6f}\n")
        f.write(f"Median Anomaly Score:   {stats['median']:.6f}\n")
        f.write(f"Standard Deviation:     {stats['std']:.6f}\n")
        f.write(f"Variance:               {stats['std']**2:.6f}\n")
        f.write(f"Range:                  {stats['max'] - stats['min']:.6f}\n")
        f.write(f"Coefficient of Variation: {stats['std']/stats['mean']:.6f}\n\n")
        
        # Percentile Distribution
        f.write("PERCENTILE DISTRIBUTION:\n")
        f.write("-"*20 + "\n")
        for p, value in stats['percentiles'].items():
            f.write(f"{p:2d}th percentile: {value:.6f}\n")
        f.write("\n")
        
        # Threshold Analysis
        f.write("THRESHOLD ANALYSIS:\n")
        f.write("-"*20 + "\n")
        f.write(f"Model Threshold ({model_info.get('threshold_percentile', 95)}th percentile): {stats['threshold']:.6f}\n")
        f.write(f"Transactions above threshold: {stats['above_threshold']:,} ({stats['above_threshold']/len(mse_scores)*100:.1f}%)\n")
        f.write(f"Transactions below threshold: {stats['below_threshold']:,} ({stats['below_threshold']/len(mse_scores)*100:.1f}%)\n\n")
        
        # Detailed Threshold Calculation
        f.write("THRESHOLD CALCULATION DETAILS:\n")
        f.write("-"*30 + "\n")
        threshold_percentile = model_info.get('threshold_percentile', 95)
        sorted_scores = np.sort(mse_scores)
        index = int(threshold_percentile / 100 * len(mse_scores))
        if index >= len(sorted_scores):
            index = len(sorted_scores) - 1
        
        f.write(f"Total transactions: {len(mse_scores):,}\n")
        f.write(f"Percentile: {threshold_percentile}%\n")
        f.write(f"Index for {threshold_percentile}th percentile: {index:,}\n")
        f.write(f"Sorted scores[{index}]: {sorted_scores[index]:.6f}\n")
        f.write(f"Final threshold: {stats['threshold']:.6f}\n\n")
        
        # Verification
        f.write("VERIFICATION:\n")
        f.write("-"*20 + "\n")
        expected_above = len(mse_scores) * (100-threshold_percentile)/100
        f.write(f"Expected transactions above threshold: {expected_above:.0f} ({100-threshold_percentile:.1f}%)\n")
        f.write(f"Actual transactions above threshold: {stats['above_threshold']:,} ({stats['above_threshold']/len(mse_scores)*100:.1f}%)\n")
        f.write(f"Difference: {abs(stats['above_threshold'] - expected_above):.0f} transactions\n\n")
        
        # Score Distribution Analysis
        f.write("SCORE DISTRIBUTION ANALYSIS:\n")
        f.write("-"*30 + "\n")
        
        # Define score ranges
        ranges = [
            (0, 0.1, "Very Low (0.0 - 0.1)"),
            (0.1, 0.2, "Low (0.1 - 0.2)"),
            (0.2, 0.5, "Medium (0.2 - 0.5)"),
            (0.5, 1.0, "High (0.5 - 1.0)"),
            (1.0, 5.0, "Very High (1.0 - 5.0)"),
            (5.0, float('inf'), "Extreme (> 5.0)")
        ]
        
        for min_val, max_val, label in ranges:
            if max_val == float('inf'):
                count = np.sum((mse_scores >= min_val))
            else:
                count = np.sum((mse_scores >= min_val) & (mse_scores < max_val))
            percentage = count / len(mse_scores) * 100
            f.write(f"{label}: {count:,} transactions ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Example Scores
        f.write("EXAMPLE ANOMALY SCORES (First 50 transactions):\n")
        f.write("-"*45 + "\n")
        f.write("Index | Anomaly Score | Status\n")
        f.write("-"*45 + "\n")
        for i, score in enumerate(mse_scores[:50]):
            status = "FRAUD" if score > stats['threshold'] else "SAFE"
            f.write(f"{i+1:5d} | {score:12.6f} | {status}\n")
        f.write("\n")
        
        # High Anomaly Scores (Top 20)
        f.write("HIGHEST ANOMALY SCORES (Top 20):\n")
        f.write("-"*35 + "\n")
        f.write("Rank | Index | Anomaly Score | Status\n")
        f.write("-"*35 + "\n")
        top_indices = np.argsort(mse_scores)[-20:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            score = mse_scores[idx]
            status = "FRAUD" if score > stats['threshold'] else "SAFE"
            f.write(f"{rank:4d} | {idx:5d} | {score:12.6f} | {status}\n")
        f.write("\n")
        
        # Low Anomaly Scores (Bottom 20)
        f.write("LOWEST ANOMALY SCORES (Bottom 20):\n")
        f.write("-"*35 + "\n")
        f.write("Rank | Index | Anomaly Score | Status\n")
        f.write("-"*35 + "\n")
        bottom_indices = np.argsort(mse_scores)[:20]
        for rank, idx in enumerate(bottom_indices, 1):
            score = mse_scores[idx]
            status = "FRAUD" if score > stats['threshold'] else "SAFE"
            f.write(f"{rank:4d} | {idx:5d} | {score:12.6f} | {status}\n")
        f.write("\n")
        
        # Summary
        f.write("SUMMARY:\n")
        f.write("-"*20 + "\n")
        f.write(f"• Total transactions analyzed: {len(mse_scores):,}\n")
        f.write(f"• Anomaly score range: {stats['min']:.6f} to {stats['max']:.6f}\n")
        f.write(f"• Model threshold: {stats['threshold']:.6f} (95th percentile)\n")
        f.write(f"• Fraud detection rate: {stats['above_threshold']/len(mse_scores)*100:.1f}%\n")
        f.write(f"• Safe transaction rate: {stats['below_threshold']/len(mse_scores)*100:.1f}%\n")
        f.write(f"• Analysis completed successfully!\n")
    
    return results_file

def main():
    """Main function to run the anomaly score analysis."""
    print("Autoencoder Anomaly Score Analysis")
    print("="*60)
    
    try:
        # Load model and data
        model, scaled_features, model_info, df = load_model_and_data()
        
        # Calculate anomaly scores
        mse_scores = calculate_anomaly_scores(model, scaled_features)
        
        # Analyze scores
        stats = analyze_anomaly_scores(mse_scores, model_info)
        
        # Write detailed analysis to file
        results_file = write_detailed_analysis_to_file(mse_scores, stats, model_info, df)
        
        print(f"\n✓ Results saved to {results_file}")
        
        # Display some example scores
        print(f"\nExample Anomaly Scores (first 10 transactions):")
        for i, score in enumerate(mse_scores[:10]):
            status = "🚨 FRAUD" if score > stats['threshold'] else "✅ SAFE"
            print(f"  Transaction {i+1}: {score:.6f} - {status}")
        
        print(f"\nAnalysis complete! 🎯")
        print(f"Check {results_file} for detailed analysis.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 