"""
Smart Factory Energy Prediction - Machine Learning Models Module
Author: Karan Kumar
Date: May 27, 2025

This module contains various machine learning models, hyperparameter tuning,
cross-validation, and evaluation methods for the energy prediction challenge.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class for training and evaluating multiple regression models"""
    
    def __init__(self, random_state=42):
        """Initialize model trainer"""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def get_base_models(self):
        """Get dictionary of base models to train"""
        models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(random_state=self.random_state),
            
            'Lasso Regression': Lasso(random_state=self.random_state),
            
            'ElasticNet': ElasticNet(random_state=self.random_state),
            
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                random_state=self.random_state
            ),
            
            'XGBoost': xgb.XGBRegressor(
                random_state=self.random_state,
                eval_metric='rmse',
                verbosity=0
            ),
            
            'LightGBM': lgb.LGBMRegressor(
                random_state=self.random_state,
                verbosity=-1
            ),
            
            'K-Nearest Neighbors': KNeighborsRegressor(),
            
            'Support Vector Regression': SVR(),
            
            'Neural Network': MLPRegressor(
                random_state=self.random_state,
                max_iter=1000
            )
        }
        
        return models
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, cv=5):
        """Evaluate a single model using multiple metrics"""
        
        # Fit the model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'training_time': training_time,
            'prediction_time': prediction_time,
            
            # Training metrics
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            
            # Test metrics
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
        }
        
        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_root_mean_squared_error')
            metrics['cv_rmse_mean'] = -cv_scores.mean()
            metrics['cv_rmse_std'] = cv_scores.std()
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            metrics['cv_rmse_mean'] = np.nan
            metrics['cv_rmse_std'] = np.nan
        
        return metrics, y_test_pred
    
    def train_all_models(self, X_train, y_train, X_test, y_test, cv=5):
        """Train and evaluate all base models"""
        print("="*60)
        print("TRAINING AND EVALUATING MODELS")
        print("="*60)
        
        models = self.get_base_models()
        results = {}
        predictions = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                metrics, y_pred = self.evaluate_model(model, X_train, y_train, X_test, y_test, cv)
                results[name] = metrics
                predictions[name] = y_pred
                self.models[name] = model
                
                # Track best model based on test R²
                if metrics['test_r2'] > self.best_score:
                    self.best_score = metrics['test_r2']
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"  Test R²: {metrics['test_r2']:.4f}")
                print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
                print(f"  Test MAE: {metrics['test_mae']:.4f}")
                
            except Exception as e:
                print(f"  Failed to train {name}: {e}")
                continue
        
        self.results = results
        self.predictions = predictions
        
        print(f"\nBest model: {self.best_model_name} (R² = {self.best_score:.4f})")
        
        return results, predictions
    
    def get_hyperparameter_grids(self):
        """Get hyperparameter grids for tuning"""
        param_grids = {
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            
            'Lasso Regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            
            'ElasticNet': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'Support Vector Regression': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        return param_grids
    
    def hyperparameter_tuning(self, X_train, y_train, models_to_tune=None, cv=5, 
                            search_type='grid', n_iter=50):
        """Perform hyperparameter tuning for selected models"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        if models_to_tune is None:
            # Tune top 5 models based on previous results
            if self.results:
                top_models = sorted(self.results.items(), 
                                  key=lambda x: x[1]['test_r2'], reverse=True)[:5]
                models_to_tune = [name for name, _ in top_models]
            else:
                models_to_tune = ['Random Forest', 'XGBoost', 'LightGBM', 
                                'Gradient Boosting', 'Ridge Regression']
        
        param_grids = self.get_hyperparameter_grids()
        base_models = self.get_base_models()
        tuned_models = {}
        tuning_results = {}
        
        for model_name in models_to_tune:
            if model_name not in param_grids:
                print(f"No parameter grid defined for {model_name}, skipping...")
                continue
                
            print(f"\nTuning {model_name}...")
            
            model = base_models[model_name]
            param_grid = param_grids[model_name]
            
            try:
                if search_type == 'grid':
                    search = GridSearchCV(
                        model, param_grid, cv=cv, scoring='neg_root_mean_squared_error',
                        n_jobs=-1, verbose=0
                    )
                else:  # randomized search
                    search = RandomizedSearchCV(
                        model, param_grid, cv=cv, scoring='neg_root_mean_squared_error',
                        n_iter=n_iter, n_jobs=-1, random_state=self.random_state, verbose=0
                    )
                
                search.fit(X_train, y_train)
                
                tuned_models[model_name] = search.best_estimator_
                tuning_results[model_name] = {
                    'best_params': search.best_params_,
                    'best_cv_score': -search.best_score_,
                    'best_estimator': search.best_estimator_
                }
                
                print(f"  Best CV RMSE: {-search.best_score_:.4f}")
                print(f"  Best parameters: {search.best_params_}")
                
            except Exception as e:
                print(f"  Failed to tune {model_name}: {e}")
                continue
        
        return tuned_models, tuning_results
    
    def create_ensemble_model(self, X_train, y_train, top_n=3):
        """Create an ensemble model using top performing models"""
        print(f"\n=== CREATING ENSEMBLE MODEL ===")
        
        # Get top n models based on test R²
        if not self.results:
            print("No model results available for ensemble creation")
            return None
            
        top_models = sorted(self.results.items(), 
                          key=lambda x: x[1]['test_r2'], reverse=True)[:top_n]
        
        print(f"Top {top_n} models for ensemble:")
        ensemble_models = []
        for name, metrics in top_models:
            print(f"  {name}: R² = {metrics['test_r2']:.4f}")
            ensemble_models.append(self.models[name])
        
        # Simple averaging ensemble
        class SimpleEnsemble:
            def __init__(self, models):
                self.models = models
            
            def fit(self, X, y):
                for model in self.models:
                    model.fit(X, y)
                return self
            
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models])
                return np.mean(predictions, axis=0)
        
        ensemble = SimpleEnsemble(ensemble_models)
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def generate_results_summary(self):
        """Generate a comprehensive summary of all model results"""
        if not self.results:
            print("No results available to summarize")
            return None
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # Sort by test R²
        results_df = results_df.sort_values('test_r2', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Display top 10 models
        display_cols = ['test_r2', 'test_rmse', 'test_mae', 'cv_rmse_mean', 'training_time']
        print(results_df[display_cols].head(10).round(4))
        
        return results_df
    
    def plot_model_comparison(self, save_path=None):
        """Create visualization comparing model performances"""
        if not self.results:
            print("No results available for plotting")
            return
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('test_r2', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² comparison
        axes[0,0].barh(results_df.index, results_df['test_r2'])
        axes[0,0].set_title('Model Comparison - Test R²')
        axes[0,0].set_xlabel('R² Score')
        
        # RMSE comparison
        axes[0,1].barh(results_df.index, results_df['test_rmse'])
        axes[0,1].set_title('Model Comparison - Test RMSE')
        axes[0,1].set_xlabel('RMSE')
        
        # MAE comparison
        axes[1,0].barh(results_df.index, results_df['test_mae'])
        axes[1,0].set_title('Model Comparison - Test MAE')
        axes[1,0].set_xlabel('MAE')
        
        # Training time comparison
        axes[1,1].barh(results_df.index, results_df['training_time'])
        axes[1,1].set_title('Model Comparison - Training Time')
        axes[1,1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def analyze_predictions(self, y_true, model_name=None, save_path=None):
        """Analyze predictions vs actual values"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.predictions:
            print(f"No predictions available for {model_name}")
            return
        
        y_pred = self.predictions[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Predicted vs Actual
        axes[0,0].scatter(y_true, y_pred, alpha=0.6)
        axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Values')
        axes[0,0].set_ylabel('Predicted Values')
        axes[0,0].set_title(f'{model_name}: Predicted vs Actual')
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title(f'{model_name}: Residuals Plot')
        
        # Residuals histogram
        axes[1,0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title(f'{model_name}: Residuals Distribution')
        
        # Q-Q plot for residuals
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title(f'{model_name}: Q-Q Plot of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Calculate additional metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{model_name} Detailed Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Mean Residual: {residuals.mean():.4f}")
        print(f"  Std Residual: {residuals.std():.4f}")
        
        return fig
    
    def get_feature_importance(self, X_train, y_train, model_name=None, top_n=15):
        """Get and visualize feature importance for tree-based models"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        # Try to get feature importance
        importance_data = None
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_data = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_data = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(model.coef_)
            }).sort_values('importance', ascending=False)
        
        if importance_data is not None:
            print(f"\n{model_name} - Top {top_n} Feature Importances:")
            print(importance_data.head(top_n))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = importance_data.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'{model_name} - Top {top_n} Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return importance_data
        else:
            print(f"Feature importance not available for {model_name}")
            return None


def main():
    """Test the model training pipeline"""
    from data_analysis import DataAnalyzer
    from preprocessing import DataPreprocessor
    
    # Load and prepare data
    analyzer = DataAnalyzer('data/data.csv')
    analyzer.load_data()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(analyzer.df)
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Train all models
    results, predictions = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Generate summary
    results_summary = trainer.generate_results_summary()
    
    # Plot comparisons
    trainer.plot_model_comparison()
    
    # Analyze best model predictions
    trainer.analyze_predictions(y_test)
    
    # Get feature importance
    trainer.get_feature_importance(X_train, y_train)
    
    return trainer, results, predictions

if __name__ == "__main__":
    trainer, results, predictions = main() 