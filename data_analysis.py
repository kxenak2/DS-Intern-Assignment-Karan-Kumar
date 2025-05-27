"""
Smart Factory Energy Prediction - Data Analysis Module
Author: Karan Kumar
Date: May 27, 2025

This module contains comprehensive data analysis and preprocessing functions
for the smart factory energy prediction challenge.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

class DataAnalyzer:
    """Class for comprehensive data analysis and preprocessing"""
    
    def __init__(self, filepath):
        """Initialize with data filepath"""
        self.filepath = filepath
        self.df = None
        self.target_col = 'equipment_energy_consumption'
        
    def load_data(self):
        """Load and basic preprocessing of the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Extract time-based features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n=== DATASET OVERVIEW ===")
        print(f"Number of rows: {self.df.shape[0]:,}")
        print(f"Number of columns: {self.df.shape[1]}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n=== DATA TYPES ===")
        print(self.df.dtypes)
        
        print("\n=== BASIC STATISTICS ===")
        print(self.df.describe())
        
    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Percentage': missing_percentage
        }).sort_values('Missing Count', ascending=False)
        
        print("\n=== MISSING VALUES ANALYSIS ===")
        missing_cols = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_cols) > 0:
            print(missing_cols)
            
            # Visualize missing values
            plt.figure(figsize=(12, 6))
            plt.bar(missing_cols.index, missing_cols['Percentage'])
            plt.title('Missing Values Percentage by Column')
            plt.xlabel('Columns')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No missing values found in the dataset!")
            
        return missing_df
    
    def analyze_target_variable(self):
        """Comprehensive analysis of the target variable"""
        target = self.target_col
        target_data = self.df[target].dropna()
        
        print(f"\n=== TARGET VARIABLE ANALYSIS: {target} ===")
        print(f"Count: {len(target_data):,}")
        print(f"Mean: {target_data.mean():.2f}")
        print(f"Median: {target_data.median():.2f}")
        print(f"Standard Deviation: {target_data.std():.2f}")
        print(f"Min: {target_data.min():.2f}")
        print(f"Max: {target_data.max():.2f}")
        print(f"Skewness: {target_data.skew():.2f}")
        print(f"Kurtosis: {target_data.kurtosis():.2f}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0,0].hist(target_data, bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title(f'Distribution of {target}')
        axes[0,0].set_xlabel('Energy Consumption (Wh)')
        axes[0,0].set_ylabel('Frequency')
        
        # Box plot
        axes[0,1].boxplot(target_data)
        axes[0,1].set_title(f'Box Plot of {target}')
        axes[0,1].set_ylabel('Energy Consumption (Wh)')
        
        # Q-Q plot
        stats.probplot(target_data, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot (Normal Distribution)')
        
        # Time series (sample)
        sample_data = self.df.sample(min(1000, len(self.df))).sort_values('timestamp')
        axes[1,1].plot(sample_data['timestamp'], sample_data[target], alpha=0.7)
        axes[1,1].set_title('Energy Consumption Over Time (Sample)')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Energy Consumption (Wh)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return target_data.describe()
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Correlations with target
        target_correlations = correlation_matrix[self.target_col].sort_values(
            key=abs, ascending=False
        )
        
        print("\n=== CORRELATIONS WITH TARGET VARIABLE ===")
        print(target_correlations)
        
        # Correlation heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f',
                    square=True,
                    cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap of All Numeric Variables')
        plt.tight_layout()
        plt.show()
        
        return target_correlations
    
    def analyze_zones(self):
        """Analyze temperature and humidity patterns across zones"""
        zone_temp_cols = [col for col in self.df.columns if 'zone' in col and 'temperature' in col]
        zone_humidity_cols = [col for col in self.df.columns if 'zone' in col and 'humidity' in col]
        
        print(f"\n=== ZONE ANALYSIS ===")
        print(f"Temperature sensors: {len(zone_temp_cols)}")
        print(f"Humidity sensors: {len(zone_humidity_cols)}")
        
        # Zone statistics
        zone_stats = {}
        for col in zone_temp_cols + zone_humidity_cols:
            zone_stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'missing_pct': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
        
        zone_stats_df = pd.DataFrame(zone_stats).T
        print("\nZone Statistics:")
        print(zone_stats_df)
        
        # Visualizations
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Temperature comparison
        temp_data = []
        temp_labels = []
        for col in zone_temp_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                temp_data.append(data)
                temp_labels.append(col.replace('zone', 'Zone ').replace('_temperature', ''))
        
        if temp_data:
            axes[0].boxplot(temp_data, labels=temp_labels)
            axes[0].set_title('Temperature Distribution Across Factory Zones')
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Humidity comparison
        humidity_data = []
        humidity_labels = []
        for col in zone_humidity_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                humidity_data.append(data)
                humidity_labels.append(col.replace('zone', 'Zone ').replace('_humidity', ''))
        
        if humidity_data:
            axes[1].boxplot(humidity_data, labels=humidity_labels)
            axes[1].set_title('Humidity Distribution Across Factory Zones')
            axes[1].set_ylabel('Humidity (%)')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return zone_stats_df
    
    def analyze_random_variables(self):
        """Analyze the random variables to determine their usefulness"""
        print("\n=== RANDOM VARIABLES ANALYSIS ===")
        
        random_vars = ['random_variable1', 'random_variable2']
        random_analysis = {}
        
        for var in random_vars:
            if var in self.df.columns:
                data = self.df[var].dropna()
                target_data = self.df[self.target_col].dropna()
                
                # Basic statistics
                stats_dict = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'correlation_with_target': self.df[var].corr(self.df[self.target_col])
                }
                
                # Statistical tests
                # Test for normality
                _, p_normal = stats.normaltest(data)
                stats_dict['is_normal'] = p_normal > 0.05
                
                # Test correlation significance
                corr_val, p_corr = stats.pearsonr(
                    self.df[var].dropna(), 
                    self.df.loc[self.df[var].notna(), self.target_col].dropna()
                )
                stats_dict['correlation_p_value'] = p_corr
                stats_dict['significant_correlation'] = p_corr < 0.05
                
                random_analysis[var] = stats_dict
                
                print(f"\n{var}:")
                for key, value in stats_dict.items():
                    print(f"  {key}: {value}")
        
        # Visualizations
        if len(random_vars) >= 2 and all(var in self.df.columns for var in random_vars):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Distribution plots
            axes[0,0].hist(self.df['random_variable1'].dropna(), bins=50, alpha=0.7)
            axes[0,0].set_title('Random Variable 1 Distribution')
            axes[0,0].set_xlabel('Value')
            axes[0,0].set_ylabel('Frequency')
            
            axes[0,1].hist(self.df['random_variable2'].dropna(), bins=50, alpha=0.7)
            axes[0,1].set_title('Random Variable 2 Distribution')
            axes[0,1].set_xlabel('Value')
            axes[0,1].set_ylabel('Frequency')
            
            # Scatter plots with target
            axes[1,0].scatter(self.df['random_variable1'], self.df[self.target_col], alpha=0.5)
            axes[1,0].set_title('Random Variable 1 vs Target')
            axes[1,0].set_xlabel('Random Variable 1')
            axes[1,0].set_ylabel('Equipment Energy Consumption')
            
            axes[1,1].scatter(self.df['random_variable2'], self.df[self.target_col], alpha=0.5)
            axes[1,1].set_title('Random Variable 2 vs Target')
            axes[1,1].set_xlabel('Random Variable 2')
            axes[1,1].set_ylabel('Equipment Energy Consumption')
            
            plt.tight_layout()
            plt.show()
            
            # Test correlation between random variables
            corr_between = self.df['random_variable1'].corr(self.df['random_variable2'])
            print(f"\nCorrelation between random variables: {corr_between:.4f}")
        
        return random_analysis
    
    def time_patterns_analysis(self):
        """Analyze time-based patterns in energy consumption"""
        print("\n=== TIME-BASED PATTERNS ANALYSIS ===")
        
        target = self.target_col
        
        # Hourly patterns
        hourly_stats = self.df.groupby('hour')[target].agg(['mean', 'std', 'count'])
        print("\nHourly energy consumption patterns:")
        print(hourly_stats)
        
        # Daily patterns
        daily_stats = self.df.groupby('day_of_week')[target].agg(['mean', 'std', 'count'])
        daily_stats.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday']
        print("\nDaily energy consumption patterns:")
        print(daily_stats)
        
        # Monthly patterns
        monthly_stats = self.df.groupby('month')[target].agg(['mean', 'std', 'count'])
        print("\nMonthly energy consumption patterns:")
        print(monthly_stats)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hourly pattern
        axes[0,0].plot(hourly_stats.index, hourly_stats['mean'], marker='o')
        axes[0,0].fill_between(hourly_stats.index, 
                              hourly_stats['mean'] - hourly_stats['std'],
                              hourly_stats['mean'] + hourly_stats['std'], 
                              alpha=0.3)
        axes[0,0].set_title('Average Energy Consumption by Hour')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Energy Consumption (Wh)')
        axes[0,0].grid(True)
        
        # Daily pattern
        axes[0,1].bar(range(len(daily_stats)), daily_stats['mean'])
        axes[0,1].set_title('Average Energy Consumption by Day of Week')
        axes[0,1].set_xlabel('Day of Week')
        axes[0,1].set_ylabel('Energy Consumption (Wh)')
        axes[0,1].set_xticks(range(len(daily_stats)))
        axes[0,1].set_xticklabels(daily_stats.index, rotation=45)
        
        # Monthly pattern
        axes[1,0].plot(monthly_stats.index, monthly_stats['mean'], marker='o')
        axes[1,0].set_title('Average Energy Consumption by Month')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Energy Consumption (Wh)')
        axes[1,0].grid(True)
        
        # Heatmap: Hour vs Day of Week
        pivot_data = self.df.groupby(['hour', 'day_of_week'])[target].mean().unstack()
        if not pivot_data.empty:
            sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.1f', ax=axes[1,1])
            axes[1,1].set_title('Energy Consumption Heatmap (Hour vs Day)')
            axes[1,1].set_xlabel('Day of Week')
            axes[1,1].set_ylabel('Hour of Day')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'hourly': hourly_stats,
            'daily': daily_stats,
            'monthly': monthly_stats
        }
    
    def comprehensive_analysis(self):
        """Run all analysis methods"""
        print("Starting comprehensive data analysis...")
        print("="*50)
        
        # Load data
        self.load_data()
        
        # Basic information
        self.basic_info()
        
        # Missing values
        missing_analysis = self.analyze_missing_values()
        
        # Target variable
        target_analysis = self.analyze_target_variable()
        
        # Correlations
        correlation_analysis = self.correlation_analysis()
        
        # Zone analysis
        zone_analysis = self.analyze_zones()
        
        # Random variables
        random_analysis = self.analyze_random_variables()
        
        # Time patterns
        time_analysis = self.time_patterns_analysis()
        
        print("\n" + "="*50)
        print("Comprehensive analysis completed!")
        
        return {
            'missing_values': missing_analysis,
            'target_stats': target_analysis,
            'correlations': correlation_analysis,
            'zone_stats': zone_analysis,
            'random_vars': random_analysis,
            'time_patterns': time_analysis
        }

def main():
    """Main function to run the analysis"""
    analyzer = DataAnalyzer('data/data.csv')
    results = analyzer.comprehensive_analysis()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main() 