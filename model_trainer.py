import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

class IPOModelTrainer:
    """
    Class to train and evaluate machine learning models for IPO gains prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        self.results = {}
        
    def prepare_features(self, df, target_col='listing_gains'):
        """
        Prepare features for modeling
        """
        print("="*50)
        print("FEATURE PREPARATION FOR MODELING")
        print("="*50)
        
        # Define feature categories
        feature_categories = {
            'basic': ['issue_size_cr', 'issue_size_log', 'issue_price', 'lot_size', 'pe_ratio'],
            'subscription': ['qib_subscription', 'nii_subscription', 'rii_subscription', 
                           'emp_subscription', 'total_subscription'],
            'temporal': ['listing_year', 'listing_month', 'listing_quarter', 'listing_day_of_week'],
            'engineered': ['opening_premium', 'listing_volatility', 'qib_rii_ratio', 
                         'institutional_ratio', 'price_pe_ratio'],
            'categorical': ['sector_encoded', 'subscription_category_encoded', 'market_period_encoded'],
            'flags': ['is_oversubscribed']
        }
        
        # Select available features
        selected_features = []
        for category, features in feature_categories.items():
            available = [f for f in features if f in df.columns]
            selected_features.extend(available)
            if available:
                print(f"{category.title()} features ({len(available)}): {available}")
        
        self.feature_names = selected_features
        
        # Prepare feature matrix and target
        X = df[selected_features].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Total features selected: {len(selected_features)}")
        
        return X, y
    
    def initialize_models(self):
        """
        Initialize different regression models
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Elastic Net': ElasticNet(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        print(f"\nInitialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  • {name}")
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluate a single model and return metrics
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_mape': mean_absolute_percentage_error(y_train, y_pred_train),
            'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'model': model
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        return metrics
    
    def train_all_models(self, X, y, test_size=0.2):
        """
        Train all models and compare performance
        """
        print("\n" + "="*50)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*50)
        
        # Initialize models
        self.initialize_models()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for linear models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']:
                metrics = self.evaluate_model(model, X_train_scaled, X_test_scaled, 
                                            y_train, y_test, name)
            else:
                metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test, name)
            
            self.results[name] = metrics
            
            # Print results
            print(f"  Train R²: {metrics['train_r2']:.4f} | Test R²: {metrics['test_r2']:.4f}")
            print(f"  Train RMSE: {metrics['train_rmse']:.2f} | Test RMSE: {metrics['test_rmse']:.2f}")
            print(f"  CV R² (mean±std): {metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}")
        
        # Identify best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_r2'])
        self.best_model = self.results[best_model_name]['model']
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"Test R² Score: {self.results[best_model_name]['test_r2']:.4f}")
        print(f"Test RMSE: {self.results[best_model_name]['test_rmse']:.2f}%")
        print(f"Test MAE: {self.results[best_model_name]['test_mae']:.2f}%")
        print(f"{'='*50}")
        
        return X_train, X_test, y_train, y_test, best_model_name
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for the specified model
        """
        print(f"\n{'='*50}")
        print(f"HYPERPARAMETER TUNING FOR {model_name.upper()}")
        print(f"{'='*50}")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        print(f"Performing grid search with {len(param_grid)} hyperparameters...")
        print(f"Total combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Use RandomizedSearchCV for efficiency with large parameter spaces
        if np.prod([len(v) for v in param_grid.values()]) > 100:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=50, cv=5, scoring='r2', 
                n_jobs=-1, random_state=42, verbose=1
            )
            print("Using RandomizedSearchCV for efficiency...")
        else:
            search = GridSearchCV(
                model, param_grid, cv=5, scoring='r2', 
                n_jobs=-1, verbose=1
            )
            print("Using GridSearchCV for comprehensive search...")
        
        # Perform search
        search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best cross-validation R²: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def evaluate_final_model(self, model, X_train, X_test, y_train, y_test):
        """
        Comprehensive evaluation of the final model
        """
        print(f"\n{'='*50}")
        print("FINAL MODEL EVALUATION")
        print(f"{'='*50}")
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate comprehensive metrics
        metrics = {
            'R² Score (Train)': r2_score(y_train, y_pred_train),
            'R² Score (Test)': r2_score(y_test, y_pred_test),
            'RMSE (Train)': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'RMSE (Test)': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE (Train)': mean_absolute_error(y_train, y_pred_train),
            'MAE (Test)': mean_absolute_error(y_test, y_pred_test),
            'MAPE (Train)': mean_absolute_percentage_error(y_train, y_pred_train) * 100,
            'MAPE (Test)': mean_absolute_percentage_error(y_test, y_pred_test) * 100
        }
        
        print("Performance Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            if 'R²' in metric:
                print(f"{metric:<20}: {value:.4f}")
            elif 'MAPE' in metric:
                print(f"{metric:<20}: {value:.2f}%")
            else:
                print(f"{metric:<20}: {value:.2f}")
        
        # Calculate prediction intervals
        residuals = y_test - y_pred_test
        residual_std = np.std(residuals)
        
        print(f"\nPrediction Accuracy:")
        print("-" * 40)
        print(f"Standard Error: ±{residual_std:.2f}%")
        print(f"68% Confidence Interval: ±{residual_std:.2f}%")
        print(f"95% Confidence Interval: ±{1.96 * residual_std:.2f}%")
        
        return metrics, y_pred_test, residuals
    
    def analyze_feature_importance(self, model, X_train, y_train):
        """
        Analyze and visualize feature importance
        """
        print(f"\n{'='*50}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*50}")
        
        # Get feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Features:")
            print("-" * 40)
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        # Permutation importance (works for all models)
        print(f"\nCalculating permutation importance...")
        perm_importance = permutation_importance(model, X_train, y_train, 
                                               n_repeats=10, random_state=42, n_jobs=-1)
        
        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print(f"\nTop 15 Features (Permutation Importance):")
        print("-" * 50)
        for i, (_, row) in enumerate(perm_df.head(15).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        # Visualize feature importance
        self._plot_feature_importance(importance_df if hasattr(model, 'feature_importances_') else None, 
                                    perm_df)
        
        return perm_df
    
    def _plot_feature_importance(self, tree_importance, perm_importance):
        """
        Plot feature importance
        """
        fig, axes = plt.subplots(1, 2 if tree_importance is not None else 1, 
                               figsize=(15, 8))
        if tree_importance is not None and len(axes) == 2:
            ax1, ax2 = axes
        else:
            ax1 = axes if tree_importance is not None else axes
            ax2 = None
        
        # Tree-based importance
        if tree_importance is not None:
            top_features = tree_importance.head(15)
            y_pos = np.arange(len(top_features))
            
            ax1.barh(y_pos, top_features['importance'], color='skyblue', alpha=0.8)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_features['feature'])
            ax1.invert_yaxis()
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Feature Importance (Tree-based)', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
        
        if ax2 is not None:
            top_perm = perm_importance.head(15)
            y_pos = np.arange(len(top_perm))
            
            ax2.barh(y_pos, top_perm['importance_mean'], 
                    xerr=top_perm['importance_std'],
                    color='lightcoral', alpha=0.8, capsize=3)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_perm['feature'])
            ax2.invert_yaxis()
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Feature Importance (Permutation)', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            # If only permutation importance
            top_perm = perm_importance.head(15)
            y_pos = np.arange(len(top_perm))
            
            ax1.barh(y_pos, top_perm['importance_mean'], 
                    xerr=top_perm['importance_std'],
                    color='lightcoral', alpha=0.8, capsize=3)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_perm['feature'])
            ax1.invert_yaxis()
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Feature Importance (Permutation)', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance(self, y_test, y_pred, residuals, model_name):
        """
        Create comprehensive performance plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(y_test, y_pred)
        ax1.set_xlabel('Actual Listing Gains (%)')
        ax1.set_ylabel('Predicted Listing Gains (%)')
        ax1.set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        ax2 = axes[0, 1]
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Listing Gains (%)')
        ax2.set_ylabel('Residuals (%)')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals distribution
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(residuals.mean(), color='red', linestyle='--', 
                   label=f'Mean: {residuals.mean():.2f}%')
        ax3.set_xlabel('Residuals (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction intervals
        ax4 = axes[1, 1]
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        y_test_sorted = y_test.iloc[sorted_indices]
        residual_std = np.std(residuals)
        
        ax4.plot(y_pred_sorted, y_test_sorted, 'bo', alpha=0.6, label='Actual')
        ax4.plot(y_pred_sorted, y_pred_sorted, 'r-', linewidth=2, label='Predicted')
        ax4.fill_between(y_pred_sorted, 
                        y_pred_sorted - 1.96*residual_std, 
                        y_pred_sorted + 1.96*residual_std, 
                        alpha=0.2, color='red', label='95% Confidence Interval')
        
        ax4.set_xlabel('Predicted Listing Gains (%)')
        ax4.set_ylabel('Listing Gains (%)')
        ax4.set_title('Prediction with Confidence Intervals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model, filename='ipo_prediction_model.pkl'):
        """
        Save the trained model
        """
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'results': self.results
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='ipo_prediction_model.pkl'):
        """
        Load a saved model
        """
        model_data = joblib.load(filename)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.results = model_data.get('results', {})
        
        print(f"Model loaded from {filename}")
        return self.best_model
    
    def predict_new_ipo(self, ipo_data):
        """
        Predict listing gains for new IPO data
        """
        if self.best_model is None:
            raise ValueError("No model trained. Please train a model first.")
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(ipo_data.columns)
        if missing_features:
            print(f"Warning: Missing features will be filled with median values: {missing_features}")
        
        # Prepare features
        X_new = ipo_data[self.feature_names].copy()
        X_new = X_new.fillna(X_new.median())
        
        # Make prediction
        prediction = self.best_model.predict(X_new)
        
        return prediction

# Example usage
if __name__ == "__main__":
    trainer = IPOModelTrainer()
    print("IPO Model Trainer ready to use!")
    print("Usage:")
    print("1. X, y = trainer.prepare_features(df)")
    print("2. X_train, X_test, y_train, y_test, best_model = trainer.train_all_models(X, y)")
    print("3. tuned_model = trainer.hyperparameter_tuning(X_train, y_train)")
