"""
Feature engineering strategies for fraud detection.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering strategies."""
    
    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given dataset."""
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features generated."""
        pass


class BaselineFeatures(FeatureEngineer):
    """Baseline features - basic transaction features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline features."""
        logger.info("Generating baseline features...")
        
        features = df.copy()
        
        # Basic features
        features['transaction_amount_log'] = np.log1p(features['transaction_amount'])
        
        # Categorical encoding
        features['payment_method'] = pd.Categorical(features['payment_method']).codes
        features['product_category'] = pd.Categorical(features['product_category']).codes
        features['device_used'] = pd.Categorical(features['device_used']).codes
        
        logger.info(f"Available baseline features: {list(features.columns)}")
        logger.info(f"Baseline features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'baseline',
            'feature_count': 9,
            'description': 'Basic transaction features only'
        }


class TemporalFeatures(FeatureEngineer):
    """Temporal features - time-based patterns."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features."""
        logger.info("Generating temporal features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        features = baseline.generate_features(df)
        
        # Temporal features
        features['is_between_11pm_and_6am'] = (
            (features['transaction_hour'] >= 23) | (features['transaction_hour'] <= 6)
        ).astype(int)
        
        logger.info(f"Added temporal features: {['is_between_11pm_and_6am']}")
        logger.info(f"Temporal features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'temporal',
            'feature_count': 10,
            'description': 'Basic features + temporal patterns'
        }


class BehaviouralFeatures(FeatureEngineer):
    """Behavioural features - customer behavior patterns."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioural features."""
        logger.info("Generating behavioural features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        features = baseline.generate_features(df)
        
        # Behavioural features
        features['amount_per_item'] = features['transaction_amount'] / features['quantity']
        
        logger.info(f"Added behavioural features: {['amount_per_item']}")
        logger.info(f"Behavioural features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'behavioural',
            'feature_count': 10,
            'description': 'Basic features + customer behavior patterns'
        }


class DemographicRiskFeatures(FeatureEngineer):
    """Demographic risk features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate demographic risk features."""
        logger.info("Generating demographic risk features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        features = baseline.generate_features(df)
        
        # Demographic risk features
        features['customer_age_risk'] = np.where(
            features['customer_age'] < 25, 1,
            np.where(features['customer_age'] > 65, 1, 0)
        )
        
        logger.info(f"Added demographic risk features: {['customer_age_risk']}")
        logger.info(f"Demographic risk features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'demographic_risk',
            'feature_count': 10,
            'description': 'Basic features + demographic risk indicators'
        }


class AdvancedFeatures(FeatureEngineer):
    """Advanced features - interaction and derived features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced features."""
        logger.info("Generating advanced features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        features = baseline.generate_features(df)
        
        # Interaction features
        features['amount_age_interaction'] = features['transaction_amount'] * features['customer_age']
        features['amount_quantity_interaction'] = features['transaction_amount'] * features['quantity']
        features['age_account_interaction'] = features['customer_age'] * features['account_age_days']
        
        # Ratio features
        features['amount_per_age'] = features['transaction_amount'] / (features['customer_age'] + 1)
        features['amount_per_account_age'] = features['transaction_amount'] / (features['account_age_days'] + 1)
        features['age_to_account_ratio'] = features['customer_age'] / (features['account_age_days'] + 1)
        
        # Polynomial features
        features['amount_squared'] = features['transaction_amount'] ** 2
        features['amount_sqrt'] = np.sqrt(features['transaction_amount'])
        features['quantity_squared'] = features['quantity'] ** 2
        
        # Binning features
        features['amount_bin'] = pd.cut(features['transaction_amount'], bins=5, labels=False)
        features['age_bin'] = pd.cut(features['customer_age'], bins=5, labels=False)
        features['quantity_bin'] = pd.cut(features['quantity'], bins=5, labels=False)
        
        # Flag features
        features['high_amount_flag'] = (features['transaction_amount'] > features['transaction_amount'].quantile(0.9)).astype(int)
        features['high_quantity_flag'] = (features['quantity'] > features['quantity'].quantile(0.9)).astype(int)
        features['new_account_flag'] = (features['account_age_days'] < 30).astype(int)
        features['young_customer_flag'] = (features['customer_age'] < 25).astype(int)
        
        # Composite risk score
        features['composite_risk'] = (
            features['high_amount_flag'] + 
            features['high_quantity_flag'] + 
            features['new_account_flag'] + 
            features['young_customer_flag']
        )
        
        logger.info(f"Added advanced features: {len(features.columns) - 9} features")
        logger.info(f"Advanced features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'advanced',
            'feature_count': 26,
            'description': 'Basic features + interaction, polynomial, and derived features'
        }


class StatisticalFeatures(FeatureEngineer):
    """Statistical features - percentiles, z-scores, etc."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features."""
        logger.info("Generating statistical features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        features = baseline.generate_features(df)
        
        # Percentile features
        features['amount_percentile'] = features['transaction_amount'].rank(pct=True)
        features['age_percentile'] = features['customer_age'].rank(pct=True)
        features['quantity_percentile'] = features['quantity'].rank(pct=True)
        
        # Z-score features
        features['amount_zscore'] = (features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
        features['age_zscore'] = (features['customer_age'] - features['customer_age'].mean()) / features['customer_age'].std()
        features['quantity_zscore'] = (features['quantity'] - features['quantity'].mean()) / features['quantity'].std()
        
        # Normalized features
        features['amount_normalized'] = (features['transaction_amount'] - features['transaction_amount'].min()) / (features['transaction_amount'].max() - features['transaction_amount'].min())
        features['age_normalized'] = (features['customer_age'] - features['customer_age'].min()) / (features['customer_age'].max() - features['customer_age'].min())
        features['quantity_normalized'] = (features['quantity'] - features['quantity'].min()) / (features['quantity'].max() - features['quantity'].min())
        
        # Log transformations
        features['amount_log_plus_1'] = np.log1p(features['transaction_amount'])
        features['quantity_log_plus_1'] = np.log1p(features['quantity'])
        features['account_age_log'] = np.log1p(features['account_age_days'])
        
        # Cube root transformations
        features['amount_cube_root'] = np.cbrt(features['transaction_amount'])
        features['quantity_cube_root'] = np.cbrt(features['quantity'])
        
        # Reciprocal features
        features['amount_reciprocal'] = 1 / (features['transaction_amount'] + 1)
        features['quantity_reciprocal'] = 1 / (features['quantity'] + 1)
        
        logger.info(f"Added statistical features: {len(features.columns) - 9} features")
        logger.info(f"Statistical features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'statistical',
            'feature_count': 25,
            'description': 'Basic features + statistical transformations and normalizations'
        }


class FraudSpecificFeatures(FeatureEngineer):
    """Fraud-specific features - domain knowledge features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fraud-specific features."""
        logger.info("Generating fraud-specific features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        features = baseline.generate_features(df)
        
        # Velocity features (rate of change)
        features['amount_velocity'] = features['transaction_amount'].rolling(window=5, min_periods=1).mean()
        features['quantity_velocity'] = features['quantity'].rolling(window=5, min_periods=1).mean()
        
        # Anomaly scores
        features['amount_anomaly'] = np.abs(features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
        features['age_anomaly'] = np.abs(features['customer_age'] - features['customer_age'].mean()) / features['customer_age'].std()
        features['quantity_anomaly'] = np.abs(features['quantity'] - features['quantity'].mean()) / features['quantity'].std()
        
        # Risk bands
        features['amount_risk_band'] = pd.cut(features['transaction_amount'], bins=[0, 100, 500, 1000, float('inf')], labels=[0, 1, 2, 3])
        features['age_risk_band'] = pd.cut(features['customer_age'], bins=[0, 25, 35, 50, float('inf')], labels=[2, 1, 0, 3], ordered=False)
        
        # Suspicious patterns
        features['suspicious_amount'] = ((features['transaction_amount'] > features['transaction_amount'].quantile(0.95)) & 
                                       (features['customer_age'] < 30)).astype(int)
        features['suspicious_quantity'] = ((features['quantity'] > features['quantity'].quantile(0.95)) & 
                                         (features['account_age_days'] < 60)).astype(int)
        features['suspicious_age'] = ((features['customer_age'] < 25) & 
                                    (features['transaction_amount'] > features['transaction_amount'].quantile(0.8))).astype(int)
        
        # Composite fraud score
        features['fraud_score'] = (
            features['amount_anomaly'] * 0.3 +
            features['age_anomaly'] * 0.2 +
            features['quantity_anomaly'] * 0.2 +
            features['suspicious_amount'] * 0.15 +
            features['suspicious_quantity'] * 0.1 +
            features['suspicious_age'] * 0.05
        )
        
        # Payment method risk
        features['payment_risk'] = np.where(
            features['payment_method'] == 0, 0.1,  # Assuming 0 is safest
            np.where(features['payment_method'] == 1, 0.3,
            np.where(features['payment_method'] == 2, 0.5, 0.8))
        )
        
        # Device risk
        features['device_risk'] = np.where(
            features['device_used'] == 0, 0.1,  # Assuming 0 is safest
            np.where(features['device_used'] == 1, 0.4, 0.7)
        )
        
        logger.info(f"Added fraud-specific features: {len(features.columns) - 9} features")
        logger.info(f"Fraud-specific features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'fraud_specific',
            'feature_count': 22,
            'description': 'Basic features + domain-specific fraud indicators'
        }


class CombinedFeatures(FeatureEngineer):
    """Combined features - all strategies together."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate combined features from all strategies."""
        logger.info("Generating combined features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        features = baseline.generate_features(df)
        
        # Add behavioural features
        behavioural = BehaviouralFeatures()
        behavioural_features = behavioural.generate_features(df)
        features['amount_per_item'] = behavioural_features['amount_per_item']
        
        # Add demographic risk features
        demographic = DemographicRiskFeatures()
        demographic_features = demographic.generate_features(df)
        features['customer_age_risk'] = demographic_features['customer_age_risk']
        
        # Add advanced features
        advanced = AdvancedFeatures()
        advanced_features = advanced.generate_features(df)
        for col in advanced_features.columns:
            if col not in features.columns:
                features[col] = advanced_features[col]
        
        # Add statistical features
        statistical = StatisticalFeatures()
        statistical_features = statistical.generate_features(df)
        for col in statistical_features.columns:
            if col not in features.columns:
                features[col] = statistical_features[col]
        
        # Add fraud-specific features
        fraud_specific = FraudSpecificFeatures()
        fraud_features = fraud_specific.generate_features(df)
        for col in fraud_features.columns:
            if col not in features.columns:
                features[col] = fraud_features[col]
        
        logger.info(f"Combined features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'combined',
            'feature_count': 57,
            'description': 'All feature engineering strategies combined'
        }


class EnsembleFeatures(FeatureEngineer):
    """Ensemble features - multiple model predictions as features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble features."""
        logger.info("Generating ensemble features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Create simple rule-based scores
        features['rule_score_1'] = (
            (features['transaction_amount'] > features['transaction_amount'].quantile(0.9)).astype(int) * 0.3 +
            (features['customer_age'] < 25).astype(int) * 0.2 +
            (features['account_age_days'] < 30).astype(int) * 0.2 +
            (features['quantity'] > features['quantity'].quantile(0.9)).astype(int) * 0.3
        )
        
        features['rule_score_2'] = (
            (features['transaction_amount'] > features['transaction_amount'].quantile(0.95)).astype(int) * 0.4 +
            (features['customer_age'] < 20).astype(int) * 0.3 +
            (features['account_age_days'] < 7).astype(int) * 0.3
        )
        
        # Calculate amount_per_item if not available
        if 'amount_per_item' not in features.columns:
            features['amount_per_item'] = features['transaction_amount'] / features['quantity']
        
        features['rule_score_3'] = (
            (features['amount_per_item'] > features['amount_per_item'].quantile(0.9)).astype(int) * 0.4 +
            (features['composite_risk'] > 2).astype(int) * 0.3 +
            (features['high_amount_flag'] + features['high_quantity_flag'] > 1).astype(int) * 0.3
        )
        
        # Ensemble score
        features['ensemble_score'] = (
            features['rule_score_1'] * 0.4 +
            features['rule_score_2'] * 0.35 +
            features['rule_score_3'] * 0.25
        )
        
        logger.info(f"Added ensemble features: {['rule_score_1', 'rule_score_2', 'rule_score_3', 'ensemble_score']}")
        logger.info(f"Ensemble features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'ensemble',
            'feature_count': 30,
            'description': 'Advanced features + ensemble rule-based scores'
        }


class DeepFeatures(FeatureEngineer):
    """Deep features - complex non-linear transformations."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate deep features."""
        logger.info("Generating deep features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Non-linear transformations
        features['amount_sin'] = np.sin(features['transaction_amount'] / 1000)
        features['amount_cos'] = np.cos(features['transaction_amount'] / 1000)
        features['age_sin'] = np.sin(features['customer_age'] / 100)
        features['age_cos'] = np.cos(features['customer_age'] / 100)
        
        # Exponential features
        features['amount_exp'] = np.exp(-features['transaction_amount'] / 1000)
        features['age_exp'] = np.exp(-features['customer_age'] / 50)
        
        # Sigmoid features
        features['amount_sigmoid'] = 1 / (1 + np.exp(-features['transaction_amount'] / 1000))
        features['age_sigmoid'] = 1 / (1 + np.exp(-features['customer_age'] / 50))
        
        # Complex interactions
        features['amount_age_sin'] = np.sin(features['transaction_amount'] / 1000) * np.sin(features['customer_age'] / 100)
        features['amount_quantity_sin'] = np.sin(features['transaction_amount'] / 1000) * np.sin(features['quantity'] / 10)
        
        # Multiplicative features
        features['amount_age_product'] = features['transaction_amount'] * features['customer_age'] / 1000
        features['amount_quantity_product'] = features['transaction_amount'] * features['quantity'] / 100
        
        # Ratio with non-linear transformations
        features['amount_age_ratio_sin'] = np.sin(features['amount_per_age'])
        features['amount_account_ratio_sin'] = np.sin(features['amount_per_account_age'])
        
        logger.info(f"Added deep features: {len(features.columns) - 26} features")
        logger.info(f"Deep features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'deep',
            'feature_count': 40,
            'description': 'Advanced features + complex non-linear transformations'
        }


class ClusteringFeatures(FeatureEngineer):
    """Clustering features - unsupervised learning features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate clustering features."""
        logger.info("Generating clustering features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Prepare data for clustering
        cluster_data = features[['transaction_amount', 'customer_age', 'quantity', 'account_age_days']].copy()
        
        # Standardize for clustering
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # K-means clustering
        kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans_7 = KMeans(n_clusters=7, random_state=42, n_init=10)
        
        features['cluster_3'] = kmeans_3.fit_predict(cluster_data_scaled)
        features['cluster_5'] = kmeans_5.fit_predict(cluster_data_scaled)
        features['cluster_7'] = kmeans_7.fit_predict(cluster_data_scaled)
        
        # Distance to cluster centers
        features['dist_to_center_3'] = kmeans_3.transform(cluster_data_scaled).min(axis=1)
        features['dist_to_center_5'] = kmeans_5.transform(cluster_data_scaled).min(axis=1)
        features['dist_to_center_7'] = kmeans_7.transform(cluster_data_scaled).min(axis=1)
        
        # Cluster-based risk scores
        cluster_3_centers = scaler.inverse_transform(kmeans_3.cluster_centers_)
        cluster_5_centers = scaler.inverse_transform(kmeans_5.cluster_centers_)
        
        # Risk based on cluster characteristics
        features['cluster_3_risk'] = features['cluster_3'].map({
            0: 0.2, 1: 0.6, 2: 0.8
        })
        features['cluster_5_risk'] = features['cluster_5'].map({
            0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9
        })
        
        logger.info(f"Added clustering features: {len(features.columns) - 26} features")
        logger.info(f"Clustering features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'clustering',
            'feature_count': 35,
            'description': 'Advanced features + unsupervised clustering features'
        }


class PCAFeatures(FeatureEngineer):
    """PCA features - dimensionality reduction features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate PCA features."""
        logger.info("Generating PCA features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Prepare data for PCA
        pca_data = features[['transaction_amount', 'customer_age', 'quantity', 'account_age_days', 
                           'amount_squared', 'amount_sqrt', 'quantity_squared']].copy()
        
        # Standardize for PCA
        scaler = StandardScaler()
        pca_data_scaled = scaler.fit_transform(pca_data)
        
        # PCA with different components
        pca_2 = PCA(n_components=2, random_state=42)
        pca_3 = PCA(n_components=3, random_state=42)
        pca_4 = PCA(n_components=4, random_state=42)
        
        pca_2_result = pca_2.fit_transform(pca_data_scaled)
        pca_3_result = pca_3.fit_transform(pca_data_scaled)
        pca_4_result = pca_4.fit_transform(pca_data_scaled)
        
        features['pca_1'] = pca_2_result[:, 0]
        features['pca_2'] = pca_2_result[:, 1]
        features['pca_3'] = pca_3_result[:, 2]
        features['pca_4'] = pca_4_result[:, 3]
        
        # Explained variance ratios
        features['pca_explained_var_2'] = pca_2.explained_variance_ratio_[0]
        features['pca_explained_var_3'] = pca_3.explained_variance_ratio_[1]
        features['pca_explained_var_4'] = pca_4.explained_variance_ratio_[2]
        
        # Reconstruction error (anomaly indicator)
        pca_2_reconstructed = pca_2.inverse_transform(pca_2_result)
        pca_3_reconstructed = pca_3.inverse_transform(pca_3_result)
        
        features['pca_reconstruction_error_2'] = np.mean((pca_data_scaled - pca_2_reconstructed) ** 2, axis=1)
        features['pca_reconstruction_error_3'] = np.mean((pca_data_scaled - pca_3_reconstructed) ** 2, axis=1)
        
        logger.info(f"Added PCA features: {len(features.columns) - 26} features")
        logger.info(f"PCA features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'pca',
            'feature_count': 34,
            'description': 'Advanced features + PCA dimensionality reduction features'
        }


class TimeSeriesFeatures(FeatureEngineer):
    """Time series features - temporal patterns and sequences."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time series features."""
        logger.info("Generating time series features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Sort by date and time for time series analysis
        df_sorted = df.sort_values(['transaction_date', 'transaction_hour']).reset_index(drop=True)
        
        # Rolling statistics
        features['amount_rolling_mean_3'] = df_sorted['transaction_amount'].rolling(window=3, min_periods=1).mean()
        features['amount_rolling_std_3'] = df_sorted['transaction_amount'].rolling(window=3, min_periods=1).std()
        features['amount_rolling_max_3'] = df_sorted['transaction_amount'].rolling(window=3, min_periods=1).max()
        
        features['amount_rolling_mean_5'] = df_sorted['transaction_amount'].rolling(window=5, min_periods=1).mean()
        features['amount_rolling_std_5'] = df_sorted['transaction_amount'].rolling(window=5, min_periods=1).std()
        
        # Lag features
        features['amount_lag_1'] = df_sorted['transaction_amount'].shift(1)
        features['amount_lag_2'] = df_sorted['transaction_amount'].shift(2)
        features['quantity_lag_1'] = df_sorted['quantity'].shift(1)
        
        # Difference features
        features['amount_diff_1'] = df_sorted['transaction_amount'].diff(1)
        features['amount_diff_2'] = df_sorted['transaction_amount'].diff(2)
        features['quantity_diff_1'] = df_sorted['quantity'].diff(1)
        
        # Rate of change
        features['amount_rate_of_change'] = features['amount_diff_1'] / (features['amount_lag_1'] + 1)
        features['quantity_rate_of_change'] = features['quantity_diff_1'] / (features['quantity_lag_1'] + 1)
        
        # Volatility
        features['amount_volatility'] = features['amount_rolling_std_5'] / (features['amount_rolling_mean_5'] + 1)
        
        # Fill NaN values
        features = features.fillna(0)
        
        logger.info(f"Added time series features: {len(features.columns) - 26} features")
        logger.info(f"Time series features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'time_series',
            'feature_count': 40,
            'description': 'Advanced features + time series patterns and sequences'
        }


class UltraAdvancedFeatures(FeatureEngineer):
    """Ultra advanced features - combination of all advanced strategies."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ultra advanced features."""
        logger.info("Generating ultra advanced features...")
        
        # Start with deep features
        deep = DeepFeatures()
        features = deep.generate_features(df)
        
        # Add ensemble features
        ensemble = EnsembleFeatures()
        ensemble_features = ensemble.generate_features(df)
        for col in ensemble_features.columns:
            if col not in features.columns:
                features[col] = ensemble_features[col]
        
        # Add clustering features
        clustering = ClusteringFeatures()
        clustering_features = clustering.generate_features(df)
        for col in clustering_features.columns:
            if col not in features.columns:
                features[col] = clustering_features[col]
        
        # Add PCA features
        pca = PCAFeatures()
        pca_features = pca.generate_features(df)
        for col in pca_features.columns:
            if col not in features.columns:
                features[col] = pca_features[col]
        
        # Add time series features
        time_series = TimeSeriesFeatures()
        time_series_features = time_series.generate_features(df)
        for col in time_series_features.columns:
            if col not in features.columns:
                features[col] = time_series_features[col]
        
        # Additional ultra-advanced features
        # Calculate missing features if not available
        if 'amount_anomaly' not in features.columns:
            features['amount_anomaly'] = (features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
        if 'age_anomaly' not in features.columns:
            features['age_anomaly'] = (features['customer_age'] - features['customer_age'].mean()) / features['customer_age'].std()
        if 'quantity_anomaly' not in features.columns:
            features['quantity_anomaly'] = (features['quantity'] - features['quantity'].mean()) / features['quantity'].std()
        if 'fraud_score' not in features.columns:
            features['fraud_score'] = (
                features['high_amount_flag'] * 0.3 +
                features['high_quantity_flag'] * 0.2 +
                features['is_between_11pm_and_6am'] * 0.2 +
                features['amount_anomaly'] * 0.15 +
                features['age_anomaly'] * 0.15
            )
        
        features['ultra_risk_score'] = (
            features['ensemble_score'] * 0.3 +
            features['cluster_5_risk'] * 0.2 +
            features['pca_reconstruction_error_3'] * 0.2 +
            features['amount_volatility'] * 0.15 +
            features['fraud_score'] * 0.15
        )
        
        features['ultra_anomaly_score'] = (
            features['amount_anomaly'] * 0.25 +
            features['age_anomaly'] * 0.2 +
            features['quantity_anomaly'] * 0.2 +
            features['pca_reconstruction_error_2'] * 0.2 +
            features['amount_volatility'] * 0.15
        )
        
        logger.info(f"Ultra advanced features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'ultra_advanced',
            'feature_count': 85,
            'description': 'All advanced strategies combined + ultra risk and anomaly scores'
        }


class NeuralFeatures(FeatureEngineer):
    """Neural network based features using autoencoder reconstruction errors."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate neural network based features."""
        logger.info("Generating neural features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Prepare numeric data for neural features
        numeric_cols = ['transaction_amount', 'customer_age', 'quantity', 'account_age_days', 
                       'transaction_hour', 'amount_squared', 'amount_sqrt', 'quantity_squared']
        
        # Ensure all columns exist
        available_cols = [col for col in numeric_cols if col in features.columns]
        if len(available_cols) < 4:
            available_cols = ['transaction_amount', 'customer_age', 'quantity', 'account_age_days']
        
        neural_data = features[available_cols].copy()
        
        # Standardize data
        scaler = StandardScaler()
        neural_data_scaled = scaler.fit_transform(neural_data)
        
        # Simple autoencoder simulation using PCA reconstruction
        pca = PCA(n_components=max(1, len(available_cols)//2), random_state=42)
        pca_result = pca.fit_transform(neural_data_scaled)
        pca_reconstructed = pca.inverse_transform(pca_result)
        
        # Reconstruction errors
        features['neural_reconstruction_error'] = np.mean((neural_data_scaled - pca_reconstructed) ** 2, axis=1)
        features['neural_reconstruction_error_std'] = np.std((neural_data_scaled - pca_reconstructed) ** 2, axis=1)
        
        # Feature importance based on reconstruction
        feature_importance = np.abs(pca.components_[0]) if len(pca.components_) > 0 else np.ones(len(available_cols))
        features['neural_feature_importance'] = np.dot(neural_data_scaled, feature_importance)
        
        # Anomaly scores
        features['neural_anomaly_score'] = (
            features['neural_reconstruction_error'] * 0.4 +
            features['neural_reconstruction_error_std'] * 0.3 +
            features['neural_feature_importance'] * 0.3
        )
        
        logger.info(f"Added neural features: {len(features.columns) - 26} features")
        logger.info(f"Neural features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'neural',
            'feature_count': 29,
            'description': 'Advanced features + neural network based reconstruction features'
        }


class GraphFeatures(FeatureEngineer):
    """Graph-based features using transaction patterns."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate graph-based features."""
        logger.info("Generating graph features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Create transaction graph features
        features['transaction_graph_degree'] = df.groupby('customer_location')['transaction_amount'].transform('count')
        features['transaction_graph_weight'] = df.groupby('customer_location')['transaction_amount'].transform('sum')
        
        # Customer behavior graph features
        features['customer_graph_degree'] = df.groupby('customer_age')['transaction_amount'].transform('count')
        features['customer_graph_weight'] = df.groupby('customer_age')['transaction_amount'].transform('sum')
        
        # Product category graph features
        features['product_graph_degree'] = df.groupby('product_category')['transaction_amount'].transform('count')
        features['product_graph_weight'] = df.groupby('product_category')['transaction_amount'].transform('sum')
        
        # Normalize graph features
        features['transaction_graph_degree_norm'] = features['transaction_graph_degree'] / features['transaction_graph_degree'].max()
        features['transaction_graph_weight_norm'] = features['transaction_graph_weight'] / features['transaction_graph_weight'].max()
        features['customer_graph_degree_norm'] = features['customer_graph_degree'] / features['customer_graph_degree'].max()
        features['customer_graph_weight_norm'] = features['customer_graph_weight'] / features['customer_graph_weight'].max()
        features['product_graph_degree_norm'] = features['product_graph_degree'] / features['product_graph_degree'].max()
        features['product_graph_weight_norm'] = features['product_graph_weight'] / features['product_graph_weight'].max()
        
        # Graph-based risk scores
        features['graph_risk_score'] = (
            features['transaction_graph_degree_norm'] * 0.3 +
            features['customer_graph_degree_norm'] * 0.3 +
            features['product_graph_degree_norm'] * 0.4
        )
        
        logger.info(f"Added graph features: {len(features.columns) - 26} features")
        logger.info(f"Graph features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'graph',
            'feature_count': 35,
            'description': 'Advanced features + graph-based transaction pattern features'
        }


class HybridFeatures(FeatureEngineer):
    """Hybrid features combining multiple strategies with custom weights."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate hybrid features."""
        logger.info("Generating hybrid features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Add statistical features
        statistical = StatisticalFeatures()
        stat_features = statistical.generate_features(df)
        for col in stat_features.columns:
            if col not in features.columns:
                features[col] = stat_features[col]
        
        # Add fraud-specific features
        fraud_specific = FraudSpecificFeatures()
        fraud_features = fraud_specific.generate_features(df)
        for col in fraud_features.columns:
            if col not in features.columns:
                features[col] = fraud_features[col]
        
        # Add neural features
        neural = NeuralFeatures()
        neural_features = neural.generate_features(df)
        for col in neural_features.columns:
            if col not in features.columns:
                features[col] = neural_features[col]
        
        # Calculate missing features if not available
        if 'amount_anomaly' not in features.columns:
            features['amount_anomaly'] = (features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
        if 'age_anomaly' not in features.columns:
            features['age_anomaly'] = (features['customer_age'] - features['customer_age'].mean()) / features['customer_age'].std()
        if 'quantity_anomaly' not in features.columns:
            features['quantity_anomaly'] = (features['quantity'] - features['quantity'].mean()) / features['quantity'].std()
        if 'amount_volatility' not in features.columns:
            features['amount_volatility'] = features['transaction_amount'].rolling(window=5, min_periods=1).std() / (features['transaction_amount'].rolling(window=5, min_periods=1).mean() + 1)
        
        # Custom hybrid risk score
        features['hybrid_risk_score'] = (
            features['amount_anomaly'] * 0.25 +
            features['age_anomaly'] * 0.15 +
            features['quantity_anomaly'] * 0.15 +
            features['neural_anomaly_score'] * 0.2 +
            features['high_amount_flag'] * 0.1 +
            features['high_quantity_flag'] * 0.1 +
            features['is_between_11pm_and_6am'] * 0.05
        )
        
        # Calculate customer_age_risk if not available
        if 'customer_age_risk' not in features.columns:
            features['customer_age_risk'] = features['customer_age'].apply(
                lambda x: 0.1 if x < 25 else 0.3 if x < 35 else 0.5 if x < 50 else 0.8
            )
        
        # Hybrid ensemble score
        features['hybrid_ensemble_score'] = (
            features['hybrid_risk_score'] * 0.4 +
            features['neural_reconstruction_error'] * 0.3 +
            features['amount_volatility'] * 0.2 +
            features['customer_age_risk'] * 0.1
        )
        
        logger.info(f"Hybrid features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'hybrid',
            'feature_count': 45,
            'description': 'Advanced + Statistical + Fraud-specific + Neural features with custom hybrid scoring'
        }


class MetaFeatures(FeatureEngineer):
    """Meta-learning features using predictions from multiple models."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate meta-learning features."""
        logger.info("Generating meta features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Add clustering features
        clustering = ClusteringFeatures()
        cluster_features = clustering.generate_features(df)
        for col in cluster_features.columns:
            if col not in features.columns:
                features[col] = cluster_features[col]
        
        # Add PCA features
        pca = PCAFeatures()
        pca_features = pca.generate_features(df)
        for col in pca_features.columns:
            if col not in features.columns:
                features[col] = pca_features[col]
        
        # Calculate missing features if not available
        if 'amount_anomaly' not in features.columns:
            features['amount_anomaly'] = (features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
        
        # Meta-features based on multiple model predictions
        features['meta_cluster_prediction'] = features['cluster_5_risk']
        features['meta_pca_prediction'] = features['pca_reconstruction_error_3']
        features['meta_anomaly_prediction'] = features['amount_anomaly']
        
        # Meta-ensemble score
        features['meta_ensemble_score'] = (
            features['meta_cluster_prediction'] * 0.35 +
            features['meta_pca_prediction'] * 0.35 +
            features['meta_anomaly_prediction'] * 0.3
        )
        
        # Meta-confidence score
        features['meta_confidence'] = (
            features['cluster_5_risk'].abs() * 0.4 +
            features['pca_reconstruction_error_3'].abs() * 0.4 +
            features['amount_anomaly'].abs() * 0.2
        )
        
        # Meta-disagreement score (variance between predictions)
        predictions = np.column_stack([
            features['meta_cluster_prediction'],
            features['meta_pca_prediction'],
            features['meta_anomaly_prediction']
        ])
        features['meta_disagreement'] = np.var(predictions, axis=1)
        
        logger.info(f"Meta features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'meta',
            'feature_count': 42,
            'description': 'Advanced + Clustering + PCA features with meta-learning ensemble predictions'
        }


class QuantumFeatures(FeatureEngineer):
    """Quantum-inspired features using quantum-like transformations."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate quantum-inspired features."""
        logger.info("Generating quantum features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Quantum-inspired transformations
        features['quantum_superposition'] = np.sin(features['transaction_amount']) * np.cos(features['customer_age'])
        features['quantum_entanglement'] = features['transaction_amount'] * features['quantity'] / (features['customer_age'] + 1)
        features['quantum_interference'] = np.sin(features['amount_squared']) + np.cos(features['quantity_squared'])
        
        # Quantum probability amplitudes
        features['quantum_prob_amplitude_1'] = np.abs(features['quantum_superposition'])
        features['quantum_prob_amplitude_2'] = np.abs(features['quantum_entanglement'])
        features['quantum_prob_amplitude_3'] = np.abs(features['quantum_interference'])
        
        # Calculate missing features if not available
        if 'amount_anomaly' not in features.columns:
            features['amount_anomaly'] = (features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
        if 'age_anomaly' not in features.columns:
            features['age_anomaly'] = (features['customer_age'] - features['customer_age'].mean()) / features['customer_age'].std()
        
        # Quantum uncertainty principle
        features['quantum_uncertainty'] = features['amount_anomaly'] * features['age_anomaly']
        
        # Quantum tunneling effect
        features['quantum_tunneling'] = np.exp(-features['transaction_amount'] / 1000) * features['quantity']
        
        # Quantum measurement
        features['quantum_measurement'] = (
            features['quantum_prob_amplitude_1'] * 0.3 +
            features['quantum_prob_amplitude_2'] * 0.3 +
            features['quantum_prob_amplitude_3'] * 0.2 +
            features['quantum_uncertainty'] * 0.2
        )
        
        logger.info(f"Added quantum features: {len(features.columns) - 26} features")
        logger.info(f"Quantum features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'quantum',
            'feature_count': 32,
            'description': 'Advanced features + quantum-inspired mathematical transformations'
        }


class EvolutionaryFeatures(FeatureEngineer):
    """Evolutionary algorithm inspired features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate evolutionary features."""
        logger.info("Generating evolutionary features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Evolutionary fitness functions
        features['evolutionary_fitness_1'] = features['transaction_amount'] / (features['customer_age'] + 1)
        features['evolutionary_fitness_2'] = features['quantity'] * features['account_age_days'] / 1000
        features['evolutionary_fitness_3'] = features['amount_squared'] / (features['quantity'] + 1)
        
        # Calculate missing features if not available
        if 'amount_anomaly' not in features.columns:
            features['amount_anomaly'] = (features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
        
        # Mutation features
        features['mutation_rate'] = np.random.random(len(features)) * 0.1
        features['mutation_effect'] = features['amount_anomaly'] * features['mutation_rate']
        
        # Selection pressure
        features['selection_pressure'] = features['high_amount_flag'] + features['high_quantity_flag']
        
        # Adaptation score
        features['adaptation_score'] = (
            features['evolutionary_fitness_1'] * 0.3 +
            features['evolutionary_fitness_2'] * 0.3 +
            features['evolutionary_fitness_3'] * 0.2 +
            features['mutation_effect'] * 0.2
        )
        
        # Survival probability
        features['survival_probability'] = 1 / (1 + np.exp(-features['adaptation_score']))
        
        logger.info(f"Added evolutionary features: {len(features.columns) - 26} features")
        logger.info(f"Evolutionary features generated: {len(features.columns)} features")
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'evolutionary',
            'feature_count': 32,
            'description': 'Advanced features + evolutionary algorithm inspired fitness and adaptation features'
        } 


class WaveletFeatures(FeatureEngineer):
    """Wavelet transform based features for signal processing approach."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate wavelet transform based features."""
        logger.info("Generating wavelet features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Wavelet-like transformations
        features['amount_wavelet_low'] = features['transaction_amount'].rolling(window=5, center=True).mean().fillna(method='bfill')
        features['amount_wavelet_high'] = features['transaction_amount'] - features['amount_wavelet_low']
        features['amount_wavelet_energy'] = features['amount_wavelet_high'] ** 2
        
        # Multi-scale analysis
        features['amount_scale_3'] = features['transaction_amount'].rolling(window=3, center=True).std().fillna(method='bfill')
        features['amount_scale_7'] = features['transaction_amount'].rolling(window=7, center=True).std().fillna(method='bfill')
        features['amount_scale_15'] = features['transaction_amount'].rolling(window=15, center=True).std().fillna(method='bfill')
        
        # Frequency domain features
        features['amount_frequency_ratio'] = features['amount_scale_3'] / (features['amount_scale_15'] + 1e-8)
        features['amount_spectral_centroid'] = (features['amount_scale_3'] * 3 + features['amount_scale_7'] * 7 + features['amount_scale_15'] * 15) / (features['amount_scale_3'] + features['amount_scale_7'] + features['amount_scale_15'] + 1e-8)
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'wavelet',
            'feature_count': 38,
            'description': 'Wavelet transform based features for multi-scale analysis'
        }


class EntropyFeatures(FeatureEngineer):
    """Entropy and information theory based features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entropy based features."""
        logger.info("Generating entropy features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Calculate entropy for different windows
        def rolling_entropy(series, window=10):
            entropy_values = []
            for i in range(len(series)):
                if i < window - 1:
                    entropy_values.append(0)
                else:
                    window_data = series.iloc[i-window+1:i+1]
                    # Discretize into bins
                    bins = pd.cut(window_data, bins=10, labels=False, duplicates='drop')
                    # Calculate entropy
                    value_counts = bins.value_counts()
                    probs = value_counts / len(bins)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    entropy_values.append(entropy)
            return pd.Series(entropy_values, index=series.index)
        
        features['amount_entropy_10'] = rolling_entropy(features['transaction_amount'], 10)
        features['amount_entropy_20'] = rolling_entropy(features['transaction_amount'], 20)
        features['quantity_entropy_10'] = rolling_entropy(features['quantity'], 10)
        
        # Cross-entropy between amount and quantity
        features['cross_entropy_amount_quantity'] = features['amount_entropy_10'] * features['quantity_entropy_10']
        
        # Information gain features
        features['amount_info_gain'] = features['transaction_amount'] * features['amount_entropy_10']
        features['quantity_info_gain'] = features['quantity'] * features['quantity_entropy_10']
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'entropy',
            'feature_count': 37,
            'description': 'Entropy and information theory based features'
        }


class FractalFeatures(FeatureEngineer):
    """Fractal dimension and self-similarity features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fractal dimension based features."""
        logger.info("Generating fractal features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Fractal-like features using Hurst exponent approximation
        def hurst_exponent(series, window=20):
            hurst_values = []
            for i in range(len(series)):
                if i < window - 1:
                    hurst_values.append(0.5)
                else:
                    window_data = series.iloc[i-window+1:i+1]
                    if len(window_data) < 2:
                        hurst_values.append(0.5)
                    else:
                        # Calculate R/S ratio
                        mean_val = window_data.mean()
                        deviations = window_data - mean_val
                        cumulative = deviations.cumsum()
                        R = cumulative.max() - cumulative.min()
                        S = window_data.std()
                        if S > 0:
                            RS_ratio = R / S
                            hurst = np.log(RS_ratio) / np.log(len(window_data))
                            hurst_values.append(max(0, min(1, hurst)))
                        else:
                            hurst_values.append(0.5)
            return pd.Series(hurst_values, index=series.index)
        
        features['amount_hurst'] = hurst_exponent(features['transaction_amount'])
        features['quantity_hurst'] = hurst_exponent(features['quantity'])
        
        # Fractal dimension approximation
        features['amount_fractal_dim'] = 2 - features['amount_hurst']
        features['quantity_fractal_dim'] = 2 - features['quantity_hurst']
        
        # Self-similarity features
        features['amount_self_similarity'] = features['transaction_amount'].rolling(window=10).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0
        ).fillna(0)
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'fractal',
            'feature_count': 35,
            'description': 'Fractal dimension and self-similarity features'
        }


class TopologicalFeatures(FeatureEngineer):
    """Topological data analysis inspired features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate topological features."""
        logger.info("Generating topological features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Persistent homology inspired features
        def topological_persistence(series, window=15):
            persistence_values = []
            for i in range(len(series)):
                if i < window - 1:
                    persistence_values.append(0)
                else:
                    window_data = series.iloc[i-window+1:i+1]
                    # Calculate local maxima and minima
                    local_max = window_data.rolling(window=3, center=True).max()
                    local_min = window_data.rolling(window=3, center=True).min()
                    
                    # Persistence as difference between max and min
                    persistence = (local_max - local_min).mean()
                    persistence_values.append(persistence)
            return pd.Series(persistence_values, index=series.index)
        
        features['amount_persistence'] = topological_persistence(features['transaction_amount'])
        features['quantity_persistence'] = topological_persistence(features['quantity'])
        
        # Betti number approximation (number of connected components)
        features['amount_betti_0'] = features['transaction_amount'].rolling(window=10).apply(
            lambda x: len(x[x > x.mean()])
        ).fillna(0)
        
        # Euler characteristic approximation
        features['amount_euler_char'] = features['amount_betti_0'] - features['amount_persistence']
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'topological',
            'feature_count': 34,
            'description': 'Topological data analysis inspired features'
        }


class HarmonicFeatures(FeatureEngineer):
    """Harmonic analysis and Fourier transform inspired features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate harmonic analysis features."""
        logger.info("Generating harmonic features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Harmonic decomposition
        def harmonic_decomposition(series, window=20):
            harmonic_values = []
            for i in range(len(series)):
                if i < window - 1:
                    harmonic_values.append(0)
                else:
                    window_data = series.iloc[i-window+1:i+1]
                    # Simple harmonic analysis using rolling statistics
                    mean_val = window_data.mean()
                    std_val = window_data.std()
                    
                    # Harmonic component
                    harmonic = std_val / (mean_val + 1e-8)
                    harmonic_values.append(harmonic)
            return pd.Series(harmonic_values, index=series.index)
        
        features['amount_harmonic'] = harmonic_decomposition(features['transaction_amount'])
        features['quantity_harmonic'] = harmonic_decomposition(features['quantity'])
        
        # Phase features
        features['amount_phase'] = np.arctan2(features['transaction_amount'], features['transaction_amount'].rolling(window=5).mean().fillna(method='bfill'))
        features['quantity_phase'] = np.arctan2(features['quantity'], features['quantity'].rolling(window=5).mean().fillna(method='bfill'))
        
        # Amplitude features
        features['amount_amplitude'] = np.sqrt(features['transaction_amount']**2 + features['transaction_amount'].rolling(window=5).mean().fillna(method='bfill')**2)
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'harmonic',
            'feature_count': 35,
            'description': 'Harmonic analysis and Fourier transform inspired features'
        }


class MorphologicalFeatures(FeatureEngineer):
    """Mathematical morphology inspired features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate morphological features."""
        logger.info("Generating morphological features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Erosion and dilation operations
        def morphological_erosion(series, window=5):
            return series.rolling(window=window, center=True).min().fillna(method='bfill')
        
        def morphological_dilation(series, window=5):
            return series.rolling(window=window, center=True).max().fillna(method='bfill')
        
        # Morphological operations
        features['amount_erosion'] = morphological_erosion(features['transaction_amount'])
        features['amount_dilation'] = morphological_dilation(features['transaction_amount'])
        features['amount_opening'] = morphological_dilation(morphological_erosion(features['transaction_amount']))
        features['amount_closing'] = morphological_erosion(morphological_dilation(features['transaction_amount']))
        
        # Morphological gradients
        features['amount_morph_gradient'] = features['amount_dilation'] - features['amount_erosion']
        features['amount_top_hat'] = features['transaction_amount'] - features['amount_opening']
        features['amount_bottom_hat'] = features['amount_closing'] - features['transaction_amount']
        
        # Quantity morphological features
        features['quantity_erosion'] = morphological_erosion(features['quantity'])
        features['quantity_dilation'] = morphological_dilation(features['quantity'])
        features['quantity_morph_gradient'] = features['quantity_dilation'] - features['quantity_erosion']
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'morphological',
            'feature_count': 40,
            'description': 'Mathematical morphology inspired features'
        }


class SpectralFeatures(FeatureEngineer):
    """Spectral analysis and frequency domain features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate spectral analysis features."""
        logger.info("Generating spectral features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Spectral density features
        def spectral_density(series, window=20):
            spectral_values = []
            for i in range(len(series)):
                if i < window - 1:
                    spectral_values.append(0)
                else:
                    window_data = series.iloc[i-window+1:i+1]
                    # Simple spectral density approximation
                    fft_vals = np.fft.fft(window_data)
                    power_spectrum = np.abs(fft_vals)**2
                    spectral_density_val = np.mean(power_spectrum[1:len(power_spectrum)//2])  # Exclude DC component
                    spectral_values.append(spectral_density_val)
            return pd.Series(spectral_values, index=series.index)
        
        features['amount_spectral_density'] = spectral_density(features['transaction_amount'])
        features['quantity_spectral_density'] = spectral_density(features['quantity'])
        
        # Frequency domain features
        def dominant_frequency(series, window=20):
            freq_values = []
            for i in range(len(series)):
                if i < window - 1:
                    freq_values.append(0)
                else:
                    window_data = series.iloc[i-window+1:i+1]
                    fft_vals = np.fft.fft(window_data)
                    power_spectrum = np.abs(fft_vals)**2
                    # Find dominant frequency
                    dominant_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                    dominant_freq = dominant_idx / len(window_data)
                    freq_values.append(dominant_freq)
            return pd.Series(freq_values, index=series.index)
        
        features['amount_dominant_freq'] = dominant_frequency(features['transaction_amount'])
        features['quantity_dominant_freq'] = dominant_frequency(features['quantity'])
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'spectral',
            'feature_count': 34,
            'description': 'Spectral analysis and frequency domain features'
        }


class GeometricFeatures(FeatureEngineer):
    """Geometric and spatial relationship features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate geometric features."""
        logger.info("Generating geometric features...")
        
        # Start with advanced features
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Geometric transformations
        features['amount_geometric_mean'] = features['transaction_amount'].rolling(window=10).apply(
            lambda x: np.exp(np.mean(np.log(x + 1e-8)))
        ).fillna(method='bfill')
        
        features['amount_geometric_std'] = features['transaction_amount'].rolling(window=10).apply(
            lambda x: np.exp(np.std(np.log(x + 1e-8)))
        ).fillna(method='bfill')
        
        # Distance-based features
        features['amount_euclidean_dist'] = np.sqrt(features['transaction_amount']**2 + features['quantity']**2)
        features['amount_manhattan_dist'] = np.abs(features['transaction_amount']) + np.abs(features['quantity'])
        features['amount_chebyshev_dist'] = np.maximum(np.abs(features['transaction_amount']), np.abs(features['quantity']))
        
        # Angular features
        features['amount_angle'] = np.arctan2(features['quantity'], features['transaction_amount'])
        features['amount_angle_sin'] = np.sin(features['amount_angle'])
        features['amount_angle_cos'] = np.cos(features['amount_angle'])
        
        # Curvature approximation
        features['amount_curvature'] = features['transaction_amount'].rolling(window=5).apply(
            lambda x: np.abs(np.diff(x, n=2)).mean() if len(x) > 2 else 0
        ).fillna(0)
        
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'geometric',
            'feature_count': 39,
            'description': 'Geometric and spatial relationship features'
        } 


class BestEnsembleFeatures(FeatureEngineer):
    """Combines the best performing feature engineering strategies together."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features by combining the best strategies."""
        logger.info("Generating best ensemble features...")
        
        # Start with advanced features as base
        advanced = AdvancedFeatures()
        features = advanced.generate_features(df)
        
        # Add hybrid features (best performer: 0.6709 AUC-ROC)
        hybrid = HybridFeatures()
        hybrid_features = hybrid.generate_features(df)
        # Add only the unique hybrid features
        for col in hybrid_features.columns:
            if col not in features.columns:
                features[col] = hybrid_features[col]
        
        # Add wavelet features (second best: 0.6660 AUC-ROC)
        wavelet = WaveletFeatures()
        wavelet_features = wavelet.generate_features(df)
        # Add only the unique wavelet features
        for col in wavelet_features.columns:
            if col not in features.columns:
                features[col] = wavelet_features[col]
        
        # Add quantum features (third best: 0.6546 AUC-ROC)
        quantum = QuantumFeatures()
        quantum_features = quantum.generate_features(df)
        # Add only the unique quantum features
        for col in quantum_features.columns:
            if col not in features.columns:
                features[col] = quantum_features[col]
        
        # Add spectral features (fourth best: 0.6479 AUC-ROC)
        spectral = SpectralFeatures()
        spectral_features = spectral.generate_features(df)
        # Add only the unique spectral features
        for col in spectral_features.columns:
            if col not in features.columns:
                features[col] = spectral_features[col]
        
        # Add fractal features (fifth best: 0.6427 AUC-ROC)
        fractal = FractalFeatures()
        fractal_features = fractal.generate_features(df)
        # Add only the unique fractal features
        for col in fractal_features.columns:
            if col not in features.columns:
                features[col] = fractal_features[col]
        
        # Create ensemble score from best strategies
        features['best_ensemble_score'] = (
            features.get('hybrid_risk_score', 0) * 0.3 +
            features.get('wavelet_energy', 0) * 0.2 +
            features.get('quantum_entanglement', 0) * 0.2 +
            features.get('spectral_density', 0) * 0.15 +
            features.get('fractal_dimension', 0) * 0.15
        )
        
        # Create meta-features from best performing strategies
        features['meta_best_prediction'] = (
            features.get('hybrid_ensemble_score', 0) * 0.4 +
            features.get('wavelet_anomaly_score', 0) * 0.3 +
            features.get('quantum_uncertainty', 0) * 0.3
        )
        
        # Add interaction features between best strategies
        features['hybrid_wavelet_interaction'] = (
            features.get('hybrid_risk_score', 0) * features.get('wavelet_energy', 0)
        )
        
        features['quantum_spectral_interaction'] = (
            features.get('quantum_entanglement', 0) * features.get('spectral_density', 0)
        )
        
        # Create final ensemble prediction
        features['final_ensemble_prediction'] = (
            features['best_ensemble_score'] * 0.5 +
            features['meta_best_prediction'] * 0.3 +
            features['hybrid_wavelet_interaction'] * 0.1 +
            features['quantum_spectral_interaction'] * 0.1
        )
        
        logger.info(f"Best ensemble features generated: {len(features.columns)} features")
        return features
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            'strategy': 'best_ensemble',
            'feature_count': 120,  # Estimated based on combining 5 strategies
            'description': 'Combines best performing strategies: hybrid, wavelet, quantum, spectral, fractal'
        } 