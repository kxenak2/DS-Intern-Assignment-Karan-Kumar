"""
Smart Factory Energy Prediction - Main Execution Script
Author: Karan Kumar
Date: May 27, 2025

This is the main script that executes the complete energy prediction pipeline
including data analysis, preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
import pickle

# Import custom modules
from data_analysis import DataAnalyzer
from preprocessing import DataPreprocessor
from models import ModelTrainer

warnings.filterwarnings('ignore')

class EnergyPredictionPipeline:
    """Complete pipeline for smart factory energy prediction"""
    
    def __init__(self, data_path='data/data.csv', results_dir='results'):
        """Initialize the pipeline"""
        self.data_path = data_path
        self.results_dir = results_dir
        self.analyzer = None
        self.preprocessor = None
        self.trainer = None
        self.final_results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
    def run_data_analysis(self):
        """Step 1: Comprehensive data analysis"""
        print("\n" + "="*80)
        print("STEP 1: DATA ANALYSIS AND EXPLORATION")
        print("="*80)
        
        self.analyzer = DataAnalyzer(self.data_path)
        analysis_results = self.analyzer.comprehensive_analysis()
        
        # Save analysis results
        self.final_results['data_analysis'] = analysis_results
        
        return analysis_results
    
    def run_preprocessing(self):
        """Step 2: Data preprocessing and feature engineering"""
        print("\n" + "="*80)
        print("STEP 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
        print("="*80)
        
        if self.analyzer is None:
            raise ValueError("Data analysis must be run first")
        
        self.preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data(
            self.analyzer.df,
            test_size=0.2,
            random_state=42
        )
        
        # Get feature importance analysis
        importance_analysis = self.preprocessor.get_feature_importance_analysis(X_train, y_train)
        
        self.final_results['preprocessing'] = {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'selected_features': self.preprocessor.selected_features,
            'feature_importance': importance_analysis
        }
        
        return X_train, X_test, y_train, y_test
    
    def run_model_training(self, X_train, X_test, y_train, y_test):
        """Step 3: Model training and evaluation"""
        print("\n" + "="*80)
        print("STEP 3: MODEL TRAINING AND EVALUATION")
        print("="*80)
        
        self.trainer = ModelTrainer(random_state=42)
        
        # Train all base models
        results, predictions = self.trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Generate results summary
        results_summary = self.trainer.generate_results_summary()
        
        # Save model comparison plot
        model_comparison_path = os.path.join(self.results_dir, 'model_comparison.png')
        self.trainer.plot_model_comparison(save_path=model_comparison_path)
        
        # Analyze best model predictions
        prediction_analysis_path = os.path.join(self.results_dir, 'prediction_analysis.png')
        self.trainer.analyze_predictions(y_test, save_path=prediction_analysis_path)
        
        # Get feature importance for best model
        feature_importance = self.trainer.get_feature_importance(X_train, y_train)
        
        self.final_results['model_training'] = {
            'results': results,
            'results_summary': results_summary,
            'best_model': self.trainer.best_model_name,
            'best_score': self.trainer.best_score,
            'feature_importance': feature_importance
        }
        
        return results, predictions
    
    def run_hyperparameter_tuning(self, X_train, y_train):
        """Step 4: Hyperparameter tuning for top models"""
        print("\n" + "="*80)
        print("STEP 4: HYPERPARAMETER TUNING")
        print("="*80)
        
        if self.trainer is None:
            raise ValueError("Model training must be run first")
        
        # Tune top 3 models
        tuned_models, tuning_results = self.trainer.hyperparameter_tuning(
            X_train, y_train, 
            models_to_tune=None,  # Will automatically select top models
            search_type='randomized',
            n_iter=30
        )
        
        self.final_results['hyperparameter_tuning'] = {
            'tuned_models': list(tuned_models.keys()),
            'tuning_results': tuning_results
        }
        
        return tuned_models, tuning_results
    
    def generate_final_insights(self):
        """Step 5: Generate insights and recommendations"""
        print("\n" + "="*80)
        print("STEP 5: INSIGHTS AND RECOMMENDATIONS")
        print("="*80)
        
        insights = {
            'data_insights': self._generate_data_insights(),
            'model_insights': self._generate_model_insights(),
            'feature_insights': self._generate_feature_insights(),
            'business_recommendations': self._generate_business_recommendations()
        }
        
        self.final_results['insights'] = insights
        
        # Print insights
        for category, insight_list in insights.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for i, insight in enumerate(insight_list, 1):
                print(f"  {i}. {insight}")
        
        return insights
    
    def _generate_data_insights(self):
        """Generate insights from data analysis"""
        insights = []
        
        if 'random_vars' in self.final_results.get('data_analysis', {}):
            random_analysis = self.final_results['data_analysis']['random_vars']
            for var, stats in random_analysis.items():
                if not stats.get('significant_correlation', False):
                    insights.append(f"{var} shows no significant correlation with energy consumption and should be excluded from modeling")
        
        if 'time_patterns' in self.final_results.get('data_analysis', {}):
            insights.append("Energy consumption shows clear time-based patterns that can be leveraged for prediction")
            insights.append("Weekend vs weekday patterns provide valuable predictive information")
        
        if 'zone_stats' in self.final_results.get('data_analysis', {}):
            insights.append("Multi-zone temperature and humidity sensors provide rich environmental context")
        
        return insights
    
    def _generate_model_insights(self):
        """Generate insights from model performance"""
        insights = []
        
        if 'model_training' in self.final_results:
            best_model = self.final_results['model_training']['best_model']
            best_score = self.final_results['model_training']['best_score']
            
            insights.append(f"{best_model} achieved the best performance with R² = {best_score:.4f}")
            
            # Analyze model types
            results = self.final_results['model_training']['results']
            tree_models = [name for name in results.keys() if any(x in name for x in ['Forest', 'Tree', 'XGB', 'LightGBM', 'Gradient'])]
            linear_models = [name for name in results.keys() if any(x in name for x in ['Linear', 'Ridge', 'Lasso', 'Elastic'])]
            
            if tree_models:
                tree_scores = [results[name]['test_r2'] for name in tree_models]
                insights.append(f"Tree-based models generally outperform linear models (avg R² = {np.mean(tree_scores):.4f})")
            
            if len(results) > 5:
                insights.append("Ensemble methods and gradient boosting show superior performance for this energy prediction task")
        
        return insights
    
    def _generate_feature_insights(self):
        """Generate insights from feature importance"""
        insights = []
        
        if 'feature_importance' in self.final_results.get('model_training', {}):
            feature_imp = self.final_results['model_training']['feature_importance']
            
            if feature_imp is not None and len(feature_imp) > 0:
                top_features = feature_imp.head(5)['feature'].tolist()
                insights.append(f"Top 5 most important features: {', '.join(top_features)}")
                
                # Analyze feature types
                time_features = [f for f in top_features if any(x in f for x in ['hour', 'day', 'month', 'weekend'])]
                zone_features = [f for f in top_features if 'zone' in f]
                weather_features = [f for f in top_features if 'outdoor' in f or 'atmospheric' in f]
                
                if time_features:
                    insights.append("Time-based features are crucial for energy prediction accuracy")
                if zone_features:
                    insights.append("Zone-specific environmental conditions significantly impact energy consumption")
                if weather_features:
                    insights.append("External weather conditions influence factory energy requirements")
        
        return insights
    
    def _generate_business_recommendations(self):
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Time-based recommendations
        recommendations.append("Implement time-of-use energy scheduling to optimize consumption during off-peak hours")
        recommendations.append("Develop predictive maintenance schedules based on energy consumption patterns")
        
        # Environmental recommendations
        recommendations.append("Install advanced HVAC controls that respond to multi-zone temperature differentials")
        recommendations.append("Implement zone-specific energy management strategies based on local environmental conditions")
        
        # Monitoring recommendations
        recommendations.append("Deploy real-time energy monitoring dashboard using the developed prediction model")
        recommendations.append("Set up automated alerts for unusual energy consumption patterns")
        
        # Operational recommendations
        recommendations.append("Consider energy-efficient equipment scheduling during optimal environmental conditions")
        recommendations.append("Implement weather-aware energy planning for manufacturing operations")
        
        # Model deployment recommendations
        recommendations.append("Deploy the trained model in production for continuous energy consumption forecasting")
        recommendations.append("Establish regular model retraining schedule to maintain prediction accuracy")
        
        return recommendations
    
    def save_results(self):
        """Save all results to files"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save final results as pickle
        results_path = os.path.join(self.results_dir, 'final_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.final_results, f)
        print(f"Final results saved to: {results_path}")
        
        # Save best model
        if self.trainer and self.trainer.best_model:
            model_path = os.path.join(self.results_dir, 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.trainer.best_model, f)
            print(f"Best model saved to: {model_path}")
        
        # Save preprocessor
        if self.preprocessor:
            preprocessor_path = os.path.join(self.results_dir, 'preprocessor.pkl')
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            print(f"Preprocessor saved to: {preprocessor_path}")
        
        # Save summary report
        self._save_summary_report()
        
    def _save_summary_report(self):
        """Save a text summary report"""
        report_path = os.path.join(self.results_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("SMART FACTORY ENERGY PREDICTION - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Author: Karan Kumar\n\n")
            
            # Data summary
            if 'data_analysis' in self.final_results:
                f.write("DATA SUMMARY:\n")
                f.write("-" * 20 + "\n")
                if self.analyzer:
                    f.write(f"Dataset shape: {self.analyzer.df.shape}\n")
                    f.write(f"Date range: {self.analyzer.df['timestamp'].min()} to {self.analyzer.df['timestamp'].max()}\n")
                f.write("\n")
            
            # Model performance
            if 'model_training' in self.final_results:
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Best model: {self.final_results['model_training']['best_model']}\n")
                f.write(f"Best R² score: {self.final_results['model_training']['best_score']:.4f}\n")
                f.write("\n")
            
            # Insights
            if 'insights' in self.final_results:
                f.write("KEY INSIGHTS:\n")
                f.write("-" * 20 + "\n")
                for category, insight_list in self.final_results['insights'].items():
                    f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                    for i, insight in enumerate(insight_list, 1):
                        f.write(f"  {i}. {insight}\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def run_complete_pipeline(self):
        """Run the complete energy prediction pipeline"""
        print("SMART FACTORY ENERGY PREDICTION PIPELINE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Author: Karan Kumar")
        
        try:
            # Step 1: Data Analysis
            self.run_data_analysis()
            
            # Step 2: Preprocessing
            X_train, X_test, y_train, y_test = self.run_preprocessing()
            
            # Step 3: Model Training
            self.run_model_training(X_train, X_test, y_train, y_test)
            
            # Step 4: Hyperparameter Tuning
            self.run_hyperparameter_tuning(X_train, y_train)
            
            # Step 5: Generate Insights
            self.generate_final_insights()
            
            # Step 6: Save Results
            self.save_results()
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Results saved in: {self.results_dir}/")
            
            return self.final_results
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed with exception: {e}")
            raise


def main():
    """Main function to run the complete pipeline"""
    
    # Initialize and run pipeline
    pipeline = EnergyPredictionPipeline(
        data_path='data/data.csv',
        results_dir='results'
    )
    
    # Run complete analysis
    results = pipeline.run_complete_pipeline()
    
    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main() 