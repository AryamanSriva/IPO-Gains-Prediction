import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentiment_engine import IPOSentimentEngine
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- CACHE DEPLOYMENT DATA AT STARTUP ---
MODEL_DATA = None
FALLBACK_MEDIANS = {}
MARKET_AVG = 0
SECTOR_STATS = {}
PROCESSED_DF = None

def load_deployment_data():
    global MODEL_DATA, FALLBACK_MEDIANS, MARKET_AVG, SECTOR_STATS, PROCESSED_DF
    
    # 1. Load Model
    models = [f for f in os.listdir('.') if f.startswith('ipo_prediction_model') and f.endswith('.pkl')]
    if models:
        rf_models = [m for m in models if 'random_forest' in m.lower()]
        model_file = rf_models[0] if rf_models else models[0]
        MODEL_DATA = joblib.load(model_file)
        print(f"Loaded model from {model_file}")

    # 2. Load Processed Data for Benchmarks and Fallbacks
    processed_csv = 'IPOs_processed.csv'
    if os.path.exists(processed_csv):
        PROCESSED_DF = pd.read_csv(processed_csv)
        MARKET_AVG = PROCESSED_DF['listing_gains'].mean()
        SECTOR_STATS = PROCESSED_DF.groupby('sector')['listing_gains'].mean().to_dict()
        
        if MODEL_DATA:
            for col in MODEL_DATA['feature_names']:
                if col in PROCESSED_DF.columns:
                    FALLBACK_MEDIANS[col] = PROCESSED_DF[col].median()
        print("Cached market data and feature medians.")

# Initialize on startup
load_deployment_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not MODEL_DATA:
            return jsonify({'error': 'Machine learning model not loaded.'}), 500
            
        data = request.json
        company_name = data.get('company_name', '')
        issue_size_cr = float(data.get('size_of_ipo', 0))
        pe_ratio = float(data.get('pe_ratio', 0))
        total_subscription = float(data.get('subscription_status', 0))
        
        model = MODEL_DATA['model']
        feature_names = MODEL_DATA['feature_names']
        
        # Initialize input with median fallback values
        input_features = {}
        for f in feature_names:
            input_features[f] = FALLBACK_MEDIANS.get(f, 0.0)
            
        # Override with user inputs
        if 'issue_size_cr' in feature_names:
            input_features['issue_size_cr'] = issue_size_cr
        if 'issue_size_log' in feature_names:
            input_features['issue_size_log'] = np.log1p(issue_size_cr)
        if 'pe_ratio' in feature_names:
            input_features['pe_ratio'] = pe_ratio
        if 'total_subscription' in feature_names:
            input_features['total_subscription'] = total_subscription
            
        # Extract Sector logic
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
        
        # Label encoding
        label_encoders = MODEL_DATA.get('label_encoders', {})
        if 'sector' in label_encoders and 'sector_encoded' in feature_names:
            le_sector = label_encoders['sector']
            try:
                input_features['sector_encoded'] = le_sector.transform([sector])[0]
            except:
                input_features['sector_encoded'] = 0
                
        # Temporal targets
        now = datetime.now()
        if 'listing_year' in input_features: input_features['listing_year'] = now.year
        if 'listing_month' in input_features: input_features['listing_month'] = now.month
            
        X_new = pd.DataFrame([input_features])[feature_names]
        
        # Scaling
        scaler = MODEL_DATA.get('scaler')
        if scaler and type(model).__name__ in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            X_new = scaler.transform(X_new)
            
        prediction = model.predict(X_new)[0]
        
        # Sentiment adjustment
        sentiment_engine = IPOSentimentEngine()
        market_data = sentiment_engine.fetch_market_pulse(company_name)
        
        # New: Add direct scraping for Chittorgarh GMP
        gmp_data = sentiment_engine.fetch_gmp_from_chittorgarh(company_name)
        if gmp_data:
            market_data['chittorgarh_gmp'] = gmp_data['gmp']
            
        adjustment_factor = sentiment_engine.calculate_adjustment_factor(market_data)
        
        adjusted_prediction = prediction * adjustment_factor
        
        # Categories
        if adjusted_prediction < 0: risk_category = "High Risk (Negative expected)"
        elif adjusted_prediction < 10: risk_category = "Low Potential (0-10% gains)"
        elif adjusted_prediction < 30: risk_category = "Moderate Potential (10-30% gains)"
        elif adjusted_prediction < 60: risk_category = "High Potential (30-60% gains)"
        else: risk_category = "Very High Potential (60%+ gains)"
            
        rmse = 6.5 # Default
        results_metadata = MODEL_DATA.get('results', {})
        if results_metadata:
            best_model_name = max(results_metadata.keys(), key=lambda x: results_metadata[x].get('test_r2', 0))
            rmse = results_metadata.get(best_model_name, {}).get('test_rmse', 6.5)
            
        # Sector analytics
        sector_avg = SECTOR_STATS.get(sector, MARKET_AVG)
        perf = "Market Performer"
        if sector_avg > MARKET_AVG * 1.2: perf = "Outperformer"
        elif sector_avg < MARKET_AVG * 0.8: perf = "Underperformer"

        return jsonify({
            'success': True,
            'predicted_gain': round(adjusted_prediction, 2),
            'risk_category': risk_category,
            'confidence_interval': [round(adjusted_prediction - rmse, 2), round(adjusted_prediction + rmse, 2)],
            'market_pulse': market_data,
            'sector_analysis': {
                'sector_name': sector,
                'avg_gain': round(sector_avg, 2),
                'market_avg': round(MARKET_AVG, 2),
                'performance': perf
            }
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
