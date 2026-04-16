import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentiment_engine import IPOSentimentEngine

app = Flask(__name__, static_folder='static', template_folder='templates')

def get_model_file():
    models = [f for f in os.listdir('.') if f.startswith('ipo_prediction_model') and f.endswith('.pkl')]
    if models:
        # Prefer Random Forest if it exists
        rf_models = [m for m in models if 'random_forest' in m.lower()]
        return rf_models[0] if rf_models else models[0]
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        company_name = data.get('company_name', '')
        issue_size_cr = float(data.get('size_of_ipo', 0))
        pe_ratio = float(data.get('pe_ratio', 0))
        total_subscription = float(data.get('subscription_status', 0))
        
        model_file = get_model_file()
        if not model_file:
            return jsonify({'error': 'Machine learning model not found. Please train the model first.'}), 500
            
        model_data = joblib.load(model_file)
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Process medians from processed CSV if available to fill missing features robustly
        fallback = {}
        processed_csv = 'IPOs_processed.csv'
        if os.path.exists(processed_csv):
            df_full = pd.read_csv(processed_csv)
            for col in feature_names:
                if col in df_full.columns:
                    fallback[col] = df_full[col].median()
        
        # Initialize input with median fallback values
        input_features = {}
        for f in feature_names:
            input_features[f] = fallback.get(f, 0.0)
            
        # Override with user inputs
        if 'issue_size_cr' in feature_names:
            input_features['issue_size_cr'] = issue_size_cr
        if 'issue_size_log' in feature_names:
            input_features['issue_size_log'] = np.log1p(issue_size_cr)
        if 'pe_ratio' in feature_names:
            input_features['pe_ratio'] = pe_ratio
        if 'total_subscription' in feature_names:
            input_features['total_subscription'] = total_subscription
            
        # Extract Sector from Company Name
        sector = 'Others'
        sector_keywords = {
            'Technology': ['tech', 'software', 'digital', 'cyber', 'data', 'it', 'systems', 'solutions', 'infotech', 'computers'],
            'Pharmaceuticals': ['pharma', 'drug', 'medicine', 'health', 'bio', 'medical', 'healthcare', 'therapeutics'],
            'Infrastructure': ['infra', 'construction', 'cement', 'steel', 'power', 'energy', 'engineering', 'projects'],
            'Finance': ['bank', 'finance', 'insurance', 'capital', 'fund', 'securities', 'financial', 'credit', 'loans'],
            'Manufacturing': ['manufacturing', 'industries', 'products', 'goods', 'factory', 'production'],
            'Textiles': ['textile', 'fabric', 'cotton', 'yarn', 'garments', 'apparel'],
            'Real_Estate': ['real estate', 'properties', 'housing', 'developers', 'builders'],
            'Food': ['food', 'beverages', 'restaurant', 'hotels', 'hospitality'],
            'Chemicals': ['chemical', 'fertilizer', 'pesticide', 'specialty chemicals'],
            'Automotive': ['auto', 'automobile', 'vehicles', 'motors', 'components']
        }
        name_lower = company_name.lower()
        for k, keywords in sector_keywords.items():
            if any(kw in name_lower for kw in keywords):
                sector = k
                break
        
        # Apply label encoding safely
        label_encoders = model_data.get('label_encoders', {})
        if 'sector' in label_encoders and 'sector_encoded' in feature_names:
            le_sector = label_encoders['sector']
            try:
                input_features['sector_encoded'] = le_sector.transform([sector])[0]
            except ValueError:
                input_features['sector_encoded'] = 0 # Default if unseen
                
        # Fix temporal features to current year/month
        from datetime import datetime
        now = datetime.now()
        if 'listing_year' in input_features:
            input_features['listing_year'] = now.year
        if 'listing_month' in input_features:
            input_features['listing_month'] = now.month
            
        # Create single row DataFrame
        X_new = pd.DataFrame([input_features])[feature_names]
        
        # Apply Model Scaler if it exists (for linear models) or directly predict
        scaler = model_data.get('scaler')
        # Actually random forest doesn't require scaling, but if model expects it, use it.
        # model_trainer.py uses scaled for Linear, not RF, but we verify type:
        if type(model).__name__ in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            X_new = scaler.transform(X_new)
            
        # Make the prediction
        prediction = model.predict(X_new)[0]
        
        # --- NEW: Sentiment & Market Pulse Integration ---
        sentiment_engine = IPOSentimentEngine()
        market_data = sentiment_engine.fetch_market_pulse(company_name)
        gmp_data = sentiment_engine.fetch_gmp_from_chittorgarh(company_name)
        
        # Merge GMP data if found
        if gmp_data:
            market_data['chittorgarh_gmp'] = gmp_data['gmp']
            
        adjustment_factor = sentiment_engine.calculate_adjustment_factor(market_data)
        adjusted_prediction = prediction * adjustment_factor
        
        # Risk Categorization logic from main_pipeline.py
        if adjusted_prediction < 0:
            risk_category = "High Risk (Negative expected)"
        elif adjusted_prediction < 10:
            risk_category = "Low Potential (0-10% gains)"
        elif adjusted_prediction < 30:
            risk_category = "Moderate Potential (10-30% gains)"
        elif adjusted_prediction < 60:
            risk_category = "High Potential (30-60% gains)"
        else:
            risk_category = "Very High Potential (60%+ gains)"
            
        # Look up standard RMSE if available
        results_metadata = model_data.get('results', {})
        best_model_name = max(results_metadata.keys(), key=lambda x: results_metadata[x].get('test_r2', 0)) if results_metadata else None
        rmse = results_metadata.get(best_model_name, {}).get('test_rmse', 6.5) if best_model_name else 6.5
            
        return jsonify({
            'success': True,
            'predicted_gain': round(adjusted_prediction, 2),
            'base_ml_prediction': round(prediction, 2),
            'risk_category': risk_category,
            'confidence_interval': [round(adjusted_prediction - rmse, 2), round(adjusted_prediction + rmse, 2)],
            'market_pulse': market_data
        })

    except ValueError as ve:
        return jsonify({'error': 'Invalid numerical inputs provided.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
