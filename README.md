# 🚀 AI-Powered IPO Gains Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange.svg)
![Deployment](https://img.shields.io/badge/Deployment-Render-black.svg)

A premium, full-stack predictive analytics dashboard that forecasts listing gains for Indian IPOs using Machine Learning and Real-time Market Sentiment.

[Live Demo](https://ipo-gains-prediction.onrender.com/)

---

## ✨ Key Features

### 🧠 Intelligent Prediction Engine
*   **Machine Learning Core**: Uses an optimized **Random Forest Regressor** trained on historical Indian IPO data.
*   **Multi-Factor Analysis**: Predicts gains based on Issue Size, P/E Ratio, Subscription Status, and Sector performance.
*   **Confidence Metrics**: Provides a 95% confidence range for every prediction (RMSE ~6.5%).

### 🌡️ Real-Time Market Pulse
*   **Live GMP Tracker**: Scrapes current Grey Market Premiums (GMP) directly from financial trackers.
*   **AI Sentiment Engine**: Uses VADER sentiment analysis on the latest Google News headlines to adjust predictions based on current market hype.
*   **Bullish/Bearish Signals**: Visual indicators of the current market mood for any specific IPO.

### 📊 Deep Sector Benchmarking
*   **Relative Performance**: Automatically tags IPOs as **Outperformers**, **Market Performers**, or **Underperformers** relative to the broader market index.
*   **Industry Insights**: Calculates historical average gains for specific sectors (Tech, Pharma, Infra, etc.) to grounded predictions in industry reality.

### 🎨 Premium Dashboard Experience
*   **Glassmorphism Design**: High-end dark mode UI with interactive analytical charts.
*   **Insights Gallery**: Visualizations of Feature Importance, Sector Success Rates, and Correlation Matrices.
*   **Mobile Responsive**: Optimized for tracking IPOs on the go.

---

## 🛠️ Technical Stack

*   **Backend**: Flask (Python)
*   **Machine Learning**: Scikit-Learn, Joblib
*   **Data Processing**: Pandas, NumPy, OpenPyXL
*   **Sentiment Analysis**: VADER Sentiment Engine
*   **Web Scraping**: Requests, ElementTree (Google News RSS)
*   **Frontend**: Vanilla HTML5, CSS3 (Modern Flexbox/Grid), JavaScript (ES6+)
*   **Production Server**: Gunicorn

---

## 🚀 Installation & Local Development

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/AryamanSriva/IPO-Gains-Prediction.git
    cd IPO-Gains-Prediction
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train/Re-train the Model** (Optional):
    ```bash
    python main_pipeline.py
    ```

4.  **Run the App**:
    ```bash
    python app.py
    ```
    Access the app at `http://localhost:5000`

---

## ☁️ Deployment (Render)

The project is fully optimized for **Render**.

1.  **Auto-Deploy**: Enabled via `render.yaml`. Any push to the `main` branch triggers an automatic build.
2.  **Blueprint Config**: The `render.yaml` file handles the entire infrastructure setup, environment variables, and build pipeline.

---

## 📂 Project Structure

```text
├── app.py                # Main Flask server
├── main_pipeline.py      # ML Training & Assessment pipeline
├── model_trainer.py      # ML model definitions and tuning
├── sentiment_engine.py   # Web scraping & AI Sentiment logic
├── data_processor.py     # Data cleaning & Engineering
├── eda_analyzer.py       # Analytical plotting & Sector analysis
├── static/               # CSS, JS, and Analytical Plots
├── templates/            # HTML Dashboards
└── IPOs_processed.csv    # Historical dataset context
```

---

## ⚠️ Disclaimer
*This tool is for educational and analytical purposes only. Grey Market Premium (GMP) data is speculative. Always consult with a certified financial advisor before making actual investments.*

---
**Developed by [AryamanSriva](https://github.com/AryamanSriva)**
