"""
IPO Gains Prediction - Complete Pipeline
========================================

This script demonstrates the complete pipeline for IPO gains prediction using Indian IPO data.
It includes data processing, exploratory analysis, model training and evaluation.

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_processor import IPODataProcessor
from eda_analyzer import IPOExploratoryAnalyzer
from model_trainer import IPOModelTrainer

class IPOPredictionPipeline:
    """
    Complete pipeline for IPO gains prediction
    """
    
    def __init__(self):
        self.processor = IPODataProcessor()
        self.analyzer = IPOExploratoryAnalyzer()
        self.trainer = IPOModelTrainer()
        self.df = None
        self.processed_df = None
        self.model = None
        
    def run_complete_pipeline(self, excel_file_path):
        """
        Run the complete IPO prediction pipeline
        """
        print("Starting IPO Gains Prediction Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Data Processing
            print("\nSTEP 1: DATA PROCESSING")
            print("-" * 40)
            self.processed_df = self.processor.process_data(excel_file_path)
            
            if self.processed_df is None:
                print("Error: Could not load or process data")
                return None
                
            print(f"Successfully processed {len(self.processed_df)} IPO records")
            
            # Step 2: Exploratory Data Analysis
            print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")
            print("-" * 40)
            eda_report = self.analyzer.create_comprehensive_report(self.processed_df)
            print("EDA completed successfully")
            
            # Step 3: Model Training
            print("\nSTEP 3: MODEL TRAINING AND EVALUATION")
            print("-" * 40)
            
            # Prepare features
            X, y = self.trainer.prepare_features(self.processed_df)
            
            # Train all models
            X_train, X_test, y_train, y_test, best_model_name = self.trainer.train_all_models(X, y)
            
            # Hyperparameter tuning for best model
            if best_model_name in ['Random Forest', 'Gradient Boosting']:
                print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
                tuned_model = self.trainer.hyperparameter_tuning(X_train, y_train, best_model_name)
                
                # Evaluate tuned model
                metrics, y_pred, residuals = self.trainer.evaluate_final_model(
                    tuned_model, X_train, X_test, y_train, y_test)
                
                self.model = tuned_model
                model_name = f"Tuned {best_model_name}"
            else:
                # Use the best model as is
                self.model = self.trainer.best_model
                metrics, y_pred, residuals = self.trainer.evaluate_final_model(
                    self.model, X_train, X_test, y_train, y_test)
                model_name = best_model_name
            
            # Feature importance analysis
            feature_importance = self.trainer.analyze_feature_importance(self.model, X_train, y_train)
            
            # Performance visualization
            self.trainer.plot_model_performance(y_test, y_pred, residuals, model_name)
            
            # Step 4: Save Model
            print(f"\nSTEP 4: SAVING MODEL")
            print("-" * 40)
            model_filename = f"ipo_prediction_model_{best_model_name.lower().replace(' ', '_')}.pkl"
            self.trainer.save_model(self.model, model_filename)
            
            # Step 5: Generate Final Report
            print(f"\nSTEP 5: FINAL RESULTS SUMMARY")
            print("=" * 60)
            
            final_report = {
                'dataset_info': {
                    'total_records': len(self.processed_df),
                    'features_used': len(X.columns),
                    'date_range': f"{self.processed_df['listing_date'].min()} to {self.processed_df['listing_date'].max()}" if 'listing_date' in self.processed_df.columns else "N/A"
                },
                'model_performance': {
                    'best_model': model_name,
                    'r2_score': metrics['R² Score (Test)'],
                    'rmse': metrics['RMSE (Test)'],
                    'mae': metrics['MAE (Test)'],
                    'prediction_error': f"±{metrics['RMSE (Test)']:.1f}%"
                },
                'key_insights': eda_report['insights'],
                'top_features': feature_importance.head(10)['feature'].tolist()
            }
            
            self._print_final_report(final_report)
            print("Pipeline completed successfully!")
            
            return final_report
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            return None
    
    def _print_final_report(self, report):
        """
        Print a formatted final report
        """
        print("\nFINAL MODEL PERFORMANCE")
        print("-" * 50)
        print(f"Best Model: {report['model_performance']['best_model']}")
        print(f"R² Score: {report['model_performance']['r2_score']:.4f}")
        print(f"RMSE: {report['model_performance']['rmse']:.2f}%")
        print(f"MAE: {report['model_performance']['mae']:.2f}%")
        print(f"Prediction Error: {report['model_performance']['prediction_error']}")
        
        print(f"\nDATASET INFORMATION")
        print("-" * 50)
        print(f"Total IPO Records: {report['dataset_info']['total_records']:,}")
        print(f"Features Used: {report['dataset_info']['features_used']}")
        print(f"Date Range: {report['dataset_info']['date_range']}")
        
        print(f"\nTOP 10 MOST IMPORTANT FEATURES")
        print("-" * 50)
        for i, feature in enumerate(report['top_features'], 1):
            print(f"{i:2d}. {feature}")
        
        print(f"\nKEY INSIGHTS")
        print("-" * 50)
        for insight in report['key_insights'][:5]:  # Show top 5 insights
            print(f"{insight}")
    
    def predict_new_ipo(self, ipo_data_dict):
        """
        Predict listing gains for a new IPO
        
        Parameters:
        -----------
        ipo_data_dict : dict
            Dictionary containing IPO information with keys matching feature names
        
        Returns:
        --------
        dict : Prediction results with confidence intervals
        """
        if self.model is None:
            raise ValueError("No model available. Please run the pipeline first.")
        
        # Convert dict to DataFrame
        ipo_df = pd.DataFrame([ipo_data_dict])
        
        # Process the data similar to training data
        # Note: This is a simplified version - in practice, you'd want to apply
        # the same preprocessing steps as used in training
        
        try:
            prediction = self.trainer.predict_new_ipo(ipo_df)
            
            # Calculate confidence intervals (approximate)
            if hasattr(self.trainer, 'results') and self.trainer.results:
                best_model_key = max(self.trainer.results.keys(), 
                                   key=lambda x: self.trainer.results[x]['test_r2'])
                rmse = self.trainer.results[best_model_key]['test_rmse']
                
                result = {
                    'predicted_gain': prediction[0],
                    'confidence_interval_68': (prediction[0] - rmse, prediction[0] + rmse),
                    'confidence_interval_95': (prediction[0] - 1.96*rmse, prediction[0] + 1.96*rmse),
                    'risk_category': self._categorize_prediction(prediction[0])
                }
            else:
                result = {
                    'predicted_gain': prediction[0],
                    'risk_category': self._categorize_prediction(prediction[0])
                }
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
    
    def _categorize_prediction(self, predicted_gain):
        """
        Categorize the predicted gain into risk categories
        """
        if predicted_gain < 0:
            return "High Risk (Negative gains expected)"
        elif predicted_gain < 10:
            return "Low Potential (Below 10% gains)"
        elif predicted_gain < 30:
            return "Moderate Potential (10-30% gains)"
        elif predicted_gain < 60:
            return "High Potential (30-60% gains)"
        else:
            return "Very High Potential (60%+ gains)"
    
    def generate_prediction_report(self, ipo_data_dict):
        """
        Generate a comprehensive prediction report for a new IPO
        """
        prediction_result = self.predict_new_ipo(ipo_data_dict)
        
        if prediction_result is None:
            return None
        
        print("\n" + "="*60)
        print("IPO LISTING GAINS PREDICTION REPORT")
        print("="*60)
        
        print(f"\nIPO Details:")
        print("-" * 30)
        for key, value in ipo_data_dict.items():
            print(f"{key}: {value}")
        
        print(f"\nPrediction Results:")
        print("-" * 30)
        print(f"Predicted Listing Gain: {prediction_result['predicted_gain']:.2f}%")
        print(f"Risk Category: {prediction_result['risk_category']}")
        
        if 'confidence_interval_68' in prediction_result:
            ci_68 = prediction_result['confidence_interval_68']
            ci_95 = prediction_result['confidence_interval_95']
            print(f"\nConfidence Intervals:")
            print("-" * 30)
            print(f"68% Confidence: {ci_68[0]:.2f}% to {ci_68[1]:.2f}%")
            print(f"95% Confidence: {ci_95[0]:.2f}% to {ci_95[1]:.2f}%")
        
        return prediction_result

def main():
    """
    Main function to demonstrate the complete pipeline
    """
    print("IPO Gains Prediction System")
    print("=" * 50)
    print("This system predicts IPO listing gains using machine learning")
    print("Based on historical Indian IPO data\n")
    
    # Initialize pipeline
    pipeline = IPOPredictionPipeline()
    
    # Example usage (you would replace with your actual Excel file path)
    excel_file = "your_ipo_data.xlsx"  # Replace with your file path
    
    print("To use this system:")
    print("1. Place your Excel file with IPO data in the same directory")
    print("2. Update the excel_file variable with your filename")
    print("3. Run: python main.py")
    print("\nExample usage:")
    print("pipeline = IPOPredictionPipeline()")
    print("report = pipeline.run_complete_pipeline('your_file.xlsx')")
    
    # Example prediction (uncomment and modify when you have trained model)
    """
    # Example new IPO data
    new_ipo = {
        'issue_size_cr': 500,  # Issue size in crores
        'issue_price': 100,    # Issue price
        'pe_ratio': 25,        # P/E ratio
        'total_subscription': 3.5,  # Total subscription times
        'qib_subscription': 2.1,    # QIB subscription
        'nii_subscription': 4.2,    # NII subscription  
        'rii_subscription': 2.8,    # RII subscription
        'sector_encoded': 1,        # Technology sector (encoded)
        'listing_year': 2024,       # Listing year
        'listing_month': 7          # Listing month
    }
    
    prediction = pipeline.generate_prediction_report(new_ipo)
    """

if __name__ == "__main__":
    main()
