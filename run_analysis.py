#!/usr/bin/env python3
"""
Smart Factory Energy Prediction - Quick Analysis Demo
Author: Karan Kumar
Date: May 27, 2025

This script demonstrates the energy prediction system with a quick analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Import our modules
from data_analysis import DataAnalyzer
from preprocessing import DataPreprocessor
from models import ModelTrainer

warnings.filterwarnings('ignore')
plt.style.use('default')  # Use default style for compatibility

def run_quick_demo():
    """Run a quick demonstration of the energy prediction system"""
    
    print("🚀 Smart Factory Energy Prediction - Quick Demo")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Author: Karan Kumar")
    
    try:
        # Step 1: Load and analyze data
        print("\n📊 Step 1: Loading and analyzing data...")
        analyzer = DataAnalyzer('data/data.csv')
        df = analyzer.load_data()
        
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"🎯 Target variable: {analyzer.target_col}")
        print(f"📈 Energy consumption range: {df[analyzer.target_col].min():.1f} - {df[analyzer.target_col].max():.1f} Wh")
        
        # Quick missing value check
        missing_total = df.isnull().sum().sum()
        print(f"❓ Missing values: {missing_total}")
        
        # Step 2: Basic preprocessing
        print("\n🔧 Step 2: Data preprocessing...")
        preprocessor = DataPreprocessor()
        
        # Quick feature engineering demo
        df_processed = preprocessor.engineer_features(df.head(1000))  # Small sample for demo
        print(f"✅ Feature engineering completed")
        print(f"📊 Features before: {df.shape[1]}, after: {df_processed.shape[1]}")
        
        # Step 3: Sample prediction with a simple model
        print("\n🤖 Step 3: Quick model demonstration...")
        
        # Prepare a small sample for quick demo
        sample_df = df.sample(min(1000, len(df)), random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            sample_df, 
            test_size=0.3, 
            random_state=42,
            remove_outliers_flag=False,  # Skip for speed
            scale_features_flag=False    # Skip for speed
        )
        
        # Train a quick Random Forest model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error
        
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"✅ Quick Random Forest model trained")
        print(f"📈 R² Score: {r2:.4f}")
        print(f"📉 RMSE: {rmse:.2f} Wh")
        
        # Feature importance preview
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 5 Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
            print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
        
        # Step 4: Business insights
        print("\n💼 Step 4: Business insights...")
        
        # Time pattern analysis
        if 'hour' in df.columns:
            hourly_avg = df.groupby('hour')[analyzer.target_col].mean()
            peak_hour = hourly_avg.idxmax()
            min_hour = hourly_avg.idxmin()
            
            print(f"⏰ Peak energy consumption hour: {peak_hour}:00 ({hourly_avg[peak_hour]:.1f} Wh)")
            print(f"⏰ Minimum energy consumption hour: {min_hour}:00 ({hourly_avg[min_hour]:.1f} Wh)")
            
            potential_savings = (hourly_avg[peak_hour] - hourly_avg[min_hour]) * 0.1  # 10% optimization
            print(f"💰 Potential savings through scheduling: {potential_savings:.1f} Wh per hour")
        
        # Correlation insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()[analyzer.target_col].sort_values(key=abs, ascending=False)
            strongest_predictor = correlations.index[1]  # Skip target itself
            print(f"🎯 Strongest predictor: {strongest_predictor} (correlation: {correlations[strongest_predictor]:.3f})")
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📋 Summary:")
        print(f"  • Dataset: {df.shape[0]:,} records, {df.shape[1]} features")
        print(f"  • Model Performance: R² = {r2:.4f}, RMSE = {rmse:.2f} Wh")
        print(f"  • System Status: ✅ All modules working correctly")
        print(f"  • Recommendation: Ready for full analysis with main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print(f"Please check your data file and dependencies.")
        return False

if __name__ == "__main__":
    success = run_quick_demo()
    if success:
        print(f"\n🚀 To run the full analysis, execute: python main.py")
    else:
        print(f"\n🔧 Please fix the issues above and try again.") 