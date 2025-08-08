# Feature Engineering Documentation

## Overview

The feature engineering pipeline implements domain-driven features based on established fraud detection principles.

## Implementation Details

```python
# src/features/feature_engineer.py
class FraudFeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.amount_transformer = PowerTransformer()
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline."""
        features = df.copy()
        
        if self.config['features']['use_amount_features']:
            features = self._engineer_amount_features(features)
        
        if self.config['features']['use_temporal_features']:
            features = self._engineer_temporal_features(features)
            
        if self.config['features']['use_risk_flags']:
            features = self._add_risk_flags(features)
            
        if self.config['features']['use_interaction_features']:
            features = self._create_interaction_features(features)
            
        return features
    
    def _engineer_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer amount-based features."""
        # Transform amount distribution
        df['amount_scaled'] = self.amount_transformer.fit_transform(
            df[['Amount']].values
        )
        
        # Amount percentile features
        df['amount_percentile'] = df['Amount'].rank(pct=True)
        
        # High value transaction flag
        df['is_high_value'] = df['amount_percentile'] > 0.95
        
        # Amount velocity features
        df['amount_7day_mean'] = df.groupby('CustomerId')['Amount'].transform(
            lambda x: x.rolling('7D').mean()
        )
        
        return df
        
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based features."""
        # Extract datetime components
        df['hour'] = df['TransactionDt'].dt.hour
        df['day_of_week'] = df['TransactionDt'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Business hours flag
        df['is_business_hours'] = (
            (df['hour'] >= 9) & 
            (df['hour'] <= 17) & 
            ~df['is_weekend']
        )
        
        # Late night transaction flag
        df['is_late_night'] = (df['hour'] >= 23) | (df['hour'] <= 4)
        
        # Transaction velocity
        df['tx_count_1h'] = df.groupby('CustomerId')['TransactionDt'].transform(
            lambda x: x.rolling('1H').count()
        )
        
        return df
        
    def _add_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fraud risk indicator flags."""
        # New account flag (less than 30 days)
        df['is_new_account'] = (
            df['AccountAge'].dt.days < 30
        )
        
        # Suspicious patterns
        df['is_suspicious_pattern'] = (
            (df['is_late_night']) &
            (df['is_high_value']) |
            (df['tx_count_1h'] > 10)  # High velocity
        )
        
        # First time merchant transaction
        df['is_first_merchant_tx'] = ~df.groupby(
            ['CustomerId', 'MerchantId']
        )['TransactionDt'].transform('cumcount').astype(bool)
        
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key indicators."""
        risk_columns = [
            'is_high_value', 'is_late_night', 
            'is_new_account', 'is_suspicious_pattern'
        ]
        
        # Create all pairwise interactions
        for col1, col2 in combinations(risk_columns, 2):
            df[f'{col1}_{col2}'] = df[col1] & df[col2]
            
        return df
```

## Feature Categories

1. **Transaction Features**
   - Amount scaling
   - Percentile-based risk flags
   
2. **Temporal Features**
   - Time-based risk patterns
   - Business hours analysis
   
3. **Customer Features**
   - Age groups
   - Account age
   - New account indicators

## Configuration

```yaml
# configs/feature_config.yaml
features:
  use_amount_features: true
  use_temporal_features: true
  use_customer_features: true
  use_risk_flags: true
  use_interaction_features: true
```
