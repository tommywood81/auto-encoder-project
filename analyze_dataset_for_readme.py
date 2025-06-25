"""
Analyze dataset for README documentation.
"""

import pandas as pd
import numpy as np

def analyze_dataset():
    """Analyze the dataset and print information for README."""
    
    # Load the raw data
    df = pd.read_csv('data/ingested/raw_ecommerce_data.csv')
    
    print("## Dataset Overview")
    print()
    print(f"**Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()
    
    # Print column information
    print("**Columns and Data Types:**")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        print(f"- `{col}`: {dtype} | {null_count:,} nulls ({null_pct:.1f}%) | {unique_count:,} unique values ({unique_pct:.1f}%)")
    
    print()
    print("## Column Analysis")
    print()
    
    # Analyze each column individually
    columns_analysis = {
        'Transaction ID': {
            'description': 'Unique identifier for each transaction',
            'action': 'Remove - not useful for modeling',
            'reason': 'Unique identifier with no predictive value'
        },
        'Customer ID': {
            'description': 'Unique identifier for each customer',
            'action': 'Remove - not useful for modeling',
            'reason': 'Unique identifier with no predictive value'
        },
        'Transaction Amount': {
            'description': 'Dollar amount of the transaction',
            'action': 'Keep - primary feature for fraud detection',
            'reason': 'Key indicator of fraud patterns, highly skewed (6.7)'
        },
        'Transaction Date': {
            'description': 'Date and time of the transaction',
            'action': 'Remove - will create derived time features',
            'reason': 'Raw date not useful, will extract hour, day, etc.'
        },
        'Payment Method': {
            'description': 'Method used for payment (credit card, debit card, PayPal, bank transfer)',
            'action': 'Keep - different fraud rates by payment type',
            'reason': '4 unique values, shows different fraud patterns'
        },
        'Product Category': {
            'description': 'Category of product purchased',
            'action': 'Keep - different fraud rates by category',
            'reason': '5 unique values, shows different fraud patterns'
        },
        'Quantity': {
            'description': 'Number of items purchased',
            'action': 'Keep - useful for fraud detection',
            'reason': 'Low cardinality, can indicate bulk fraud'
        },
        'Customer Age': {
            'description': 'Age of the customer',
            'action': 'Keep - filter to 18+ only',
            'reason': 'Age patterns important for fraud detection'
        },
        'Customer Location': {
            'description': 'Customer ID (misnamed in dataset)',
            'action': 'Keep - use frequency encoding',
            'reason': '14,868 unique values, too many for one-hot encoding'
        },
        'Device Used': {
            'description': 'Device type used for transaction (desktop, mobile, tablet)',
            'action': 'Keep - different fraud rates by device',
            'reason': '3 unique values, shows different fraud patterns'
        },
        'IP Address': {
            'description': 'IP address of the customer',
            'action': 'Remove - privacy concern, too many unique values',
            'reason': 'Privacy issue, too many unique values for modeling'
        },
        'Shipping Address': {
            'description': 'Shipping address of the customer',
            'action': 'Remove - privacy concern, too many unique values',
            'reason': 'Privacy issue, too many unique values for modeling'
        },
        'Billing Address': {
            'description': 'Billing address of the customer',
            'action': 'Remove - privacy concern, too many unique values',
            'reason': 'Privacy issue, too many unique values for modeling'
        },
        'Is Fraudulent': {
            'description': 'Target variable (1 = fraudulent, 0 = legitimate)',
            'action': 'Keep - target variable',
            'reason': 'Binary target for fraud detection'
        },
        'Account Age Days': {
            'description': 'Age of customer account in days',
            'action': 'Keep - useful for fraud detection',
            'reason': 'New accounts more likely to be fraudulent'
        },
        'Transaction Hour': {
            'description': 'Hour of day when transaction occurred',
            'action': 'Keep - temporal patterns important',
            'reason': 'Fraud patterns vary by time of day'
        }
    }
    
    for col in df.columns:
        if col in columns_analysis:
            analysis = columns_analysis[col]
            print(f"### {col}")
            print(f"- **Description**: {analysis['description']}")
            print(f"- **Action**: {analysis['action']}")
            print(f"- **Reason**: {analysis['reason']}")
            print()

if __name__ == "__main__":
    analyze_dataset() 