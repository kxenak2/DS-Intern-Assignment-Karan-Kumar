"""
Smart Factory Energy Prediction - Data Preprocessing and Feature Engineering Module
Author: Karan Kumar
Date: May 27, 2025

This module contains data preprocessing and feature engineering functions
for the smart factory energy prediction challenge.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Class for comprehensive data preprocessing and feature engineering"""
    
    def __init__(self, target_col='equipment_energy_consumption'):
        """Initialize preprocessor"""
        self.target_col = target_col
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
        self.selected_features = None
        
    def handle_missing_values(self, df, strategy='median', n_neighbors=5):
        """Handle missing values in the dataset"""
        print(f"\n=== HANDLING MISSING VALUES ===")
        print(f"Strategy: {strategy}")
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        df_processed = df.copy()
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if strategy == 'knn':
                self.imputer = KNNImputer(n_neighbors=n_neighbors)
                df_processed[numeric_cols] = self.imputer.fit_transform(df_processed[numeric_cols])
            else:
                self.imputer = SimpleImputer(strategy=strategy)
                df_processed[numeric_cols] = self.imputer.fit_transform(df_processed[numeric_cols])
        
        # Handle categorical columns (if any)
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if col != 'timestamp':  # Don't impute timestamp
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        missing_after = df_processed.isnull().sum().sum()
        print(f"Missing values after imputation: {missing_after}")
        
        return df_processed
    
    def remove_outliers(self, df, method='iqr', threshold=1.5):
        """Remove outliers from the dataset"""
        print(f"\n=== OUTLIER REMOVAL ===")
        print(f"Method: {method}, Threshold: {threshold}")
        
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            if col == self.target_col:
                continue  # Don't remove outliers from target variable
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = z_scores > threshold
            
            outlier_count = outliers.sum()
            outlier_counts[col] = outlier_count
            
            if outlier_count > 0:
                df_clean = df_clean[~outliers]
        
        print(f"Outliers removed per column:")
        for col, count in outlier_counts.items():
            if count > 0:
                print(f"  {col}: {count}")
        
        print(f"Dataset shape before outlier removal: {df.shape}")
        print(f"Dataset shape after outlier removal: {df_clean.shape}")
        
        return df_clean
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        print(f"\n=== FEATURE ENGINEERING ===")
        
        df_features = df.copy()
        
        # Time-based features (if not already created)
        if 'timestamp' in df_features.columns:
            df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
            
            if 'hour' not in df_features.columns:
                df_features['hour'] = df_features['timestamp'].dt.hour
            if 'day_of_week' not in df_features.columns:
                df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            if 'month' not in df_features.columns:
                df_features['month'] = df_features['timestamp'].dt.month
            if 'is_weekend' not in df_features.columns:
                df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time features
        if 'hour' in df_features.columns:
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        
        if 'day_of_week' in df_features.columns:
            df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        if 'month' in df_features.columns:
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Zone-based features
        zone_temp_cols = [col for col in df_features.columns if 'zone' in col and 'temperature' in col]
        zone_humidity_cols = [col for col in df_features.columns if 'zone' in col and 'humidity' in col]
        
        if zone_temp_cols:
            # Average temperature across all zones
            df_features['avg_zone_temperature'] = df_features[zone_temp_cols].mean(axis=1)
            # Temperature range (max - min)
            df_features['temp_range'] = df_features[zone_temp_cols].max(axis=1) - df_features[zone_temp_cols].min(axis=1)
            # Temperature variance
            df_features['temp_variance'] = df_features[zone_temp_cols].var(axis=1)
        
        if zone_humidity_cols:
            # Average humidity across all zones
            df_features['avg_zone_humidity'] = df_features[zone_humidity_cols].mean(axis=1)
            # Humidity range
            df_features['humidity_range'] = df_features[zone_humidity_cols].max(axis=1) - df_features[zone_humidity_cols].min(axis=1)
            # Humidity variance
            df_features['humidity_variance'] = df_features[zone_humidity_cols].var(axis=1)
        
        # Weather-based features
        if 'outdoor_temperature' in df_features.columns and zone_temp_cols:
            # Temperature difference between outdoor and average indoor
            df_features['temp_diff_outdoor_indoor'] = df_features['outdoor_temperature'] - df_features['avg_zone_temperature']
        
        if 'outdoor_humidity' in df_features.columns and zone_humidity_cols:
            # Humidity difference between outdoor and average indoor
            df_features['humidity_diff_outdoor_indoor'] = df_features['outdoor_humidity'] - df_features['avg_zone_humidity']
        
        # Comfort index (combining temperature and humidity)
        if 'avg_zone_temperature' in df_features.columns and 'avg_zone_humidity' in df_features.columns:
            # Heat index approximation
            df_features['comfort_index'] = df_features['avg_zone_temperature'] + 0.5 * df_features['avg_zone_humidity']
        
        # Energy efficiency ratios
        if 'lighting_energy' in df_features.columns:
            # Total energy (equipment + lighting)
            df_features['total_energy'] = df_features[self.target_col] + df_features['lighting_energy']
            # Equipment energy ratio
            df_features['equipment_energy_ratio'] = df_features[self.target_col] / (df_features['total_energy'] + 1e-6)
        
        # Pressure and weather interaction
        if 'atmospheric_pressure' in df_features.columns and 'wind_speed' in df_features.columns:
            df_features['pressure_wind_interaction'] = df_features['atmospheric_pressure'] * df_features['wind_speed']
        
        # Working hours indicator
        if 'hour' in df_features.columns:
            df_features['is_working_hours'] = ((df_features['hour'] >= 8) & (df_features['hour'] <= 17)).astype(int)
        
        # Lag features for time series (if data is sequential)
        if self.target_col in df_features.columns:
            df_features['target_lag1'] = df_features[self.target_col].shift(1)
            df_features['target_lag2'] = df_features[self.target_col].shift(2)
            df_features['target_rolling_mean_3'] = df_features[self.target_col].rolling(window=3).mean()
        
        new_features = [col for col in df_features.columns if col not in df.columns]
        print(f"Created {len(new_features)} new features:")
        for feature in new_features:
            print(f"  - {feature}")
        
        return df_features
    
    def select_features(self, X, y, method='all', k=20, threshold=0.01):
        """Select the most important features"""
        print(f"\n=== FEATURE SELECTION ===")
        print(f"Method: {method}, k={k}, threshold={threshold}")
        print(f"Initial feature count: {X.shape[1]}")
        
        if method == 'correlation':
            # Remove features with low correlation to target
            correlations = pd.DataFrame(X).corrwith(pd.Series(y)).abs()
            selected_features = correlations[correlations > threshold].index.tolist()
            X_selected = X[selected_features]
            
        elif method == 'univariate':
            # Select k best features using f_regression
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selector = selector
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selector = selector
            
        elif method == 'tree_based':
            # Tree-based feature selection
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator, threshold=threshold)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selector = selector
            
        elif method == 'all':
            # Use all features
            X_selected = X
            selected_features = X.columns.tolist()
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.selected_features = selected_features
        
        print(f"Selected {len(selected_features)} features:")
        for feature in selected_features:
            print(f"  - {feature}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """Scale features using specified method"""
        print(f"\n=== FEATURE SCALING ===")
        print(f"Method: {method}")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data(self, df, test_size=0.2, random_state=42, 
                    handle_missing=True, remove_outliers_flag=True,
                    engineer_features_flag=True, select_features_flag=True,
                    scale_features_flag=True):
        """Complete data preparation pipeline"""
        print("="*60)
        print("STARTING DATA PREPARATION PIPELINE")
        print("="*60)
        
        df_processed = df.copy()
        
        # 1. Handle missing values
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed)
        
        # 2. Feature engineering
        if engineer_features_flag:
            df_processed = self.engineer_features(df_processed)
        
        # 3. Remove outliers
        if remove_outliers_flag:
            df_processed = self.remove_outliers(df_processed)
        
        # 4. Prepare features and target
        # Remove non-predictive columns
        cols_to_remove = ['timestamp']
        # Handle random variables decision
        random_vars = ['random_variable1', 'random_variable2']
        
        # Analyze random variables correlation with target
        for var in random_vars:
            if var in df_processed.columns:
                correlation = abs(df_processed[var].corr(df_processed[self.target_col]))
                print(f"\n{var} correlation with target: {correlation:.4f}")
                if correlation < 0.05:  # Very weak correlation
                    cols_to_remove.append(var)
                    print(f"Removing {var} due to weak correlation")
        
        # Remove specified columns
        available_cols_to_remove = [col for col in cols_to_remove if col in df_processed.columns]
        df_processed = df_processed.drop(columns=available_cols_to_remove)
        
        # Separate features and target
        X = df_processed.drop(columns=[self.target_col])
        y = df_processed[self.target_col]
        
        # Handle any remaining non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        print(f"\nFeatures shape before selection: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # 5. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # 6. Feature selection
        if select_features_flag:
            X_train = self.select_features(X_train, y_train, method='tree_based')
            X_test = X_test[self.selected_features]
        
        # 7. Feature scaling
        if scale_features_flag:
            X_train, X_test = self.scale_features(X_train, X_test, method='robust')
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETED")
        print("="*60)
        print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_analysis(self, X, y):
        """Analyze feature importance using multiple methods"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 features by Random Forest importance:")
        print(rf_importance.head(15))
        
        # Correlation-based importance
        correlations = pd.DataFrame({
            'feature': X.columns,
            'correlation': [abs(X[col].corr(y)) for col in X.columns]
        }).sort_values('correlation', ascending=False)
        
        print("\nTop 15 features by correlation with target:")
        print(correlations.head(15))
        
        return {
            'rf_importance': rf_importance,
            'correlations': correlations
        }


def main():
    """Test the preprocessing pipeline"""
    from data_analysis import DataAnalyzer
    
    # Load and analyze data
    analyzer = DataAnalyzer('data/data.csv')
    analyzer.load_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(analyzer.df)
    
    # Feature importance analysis
    importance_analysis = preprocessor.get_feature_importance_analysis(X_train, y_train)
    
    return preprocessor, X_train, X_test, y_train, y_test, importance_analysis

if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test, importance_analysis = main() 