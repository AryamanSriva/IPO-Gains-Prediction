import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging
from xml.etree import ElementTree

# Set up logging for debugging
logging.basicConfig(filename='sentiment_debug.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class IPOSentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_gmp_from_chittorgarh(self, ipo_name):
        """
        Directly scrapes chittorgarh for current GMP from their tracker list.
        """
        url = 'https://www.chittorgarh.com/report/ipo-gmp-today/845/'
        try:
            logging.info(f"Scraping Chittorgarh for {ipo_name}")
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                # Use pandas to quickly grab matches
                tables = pd.read_html(response.text)
                if tables:
                    df = tables[0]
                    # Columns: IPO | Price | GMP | % | ...
                    # Try to find a row containing the name
                    match_row = df[df.iloc[:, 0].str.contains(ipo_name, case=False, na=False)]
                    if not match_row.empty:
                        gmp_val = str(match_row.iloc[0, 2]) # 3rd col is usually GMP
                        logging.info(f"Found GMP on Chittorgarh: {gmp_val}")
                        return {'gmp': gmp_val, 'source': 'Chittorgarh'}
        except Exception as e:
            logging.error(f"Chittorgarh scrape error: {e}")
        return None

    def fetch_market_pulse(self, ipo_name):
        """
        Uses Google News RSS to fetch data (extremely reliable and no API key needed).
        """
        # Clean name
        search_name = re.sub(r'\s+(Limited|Ltd|Pvt\.?|India|IPO)\.?$', '', ipo_name, flags=re.IGNORECASE).strip()
        logging.info(f"Fetching Google News Pulse for: {search_name}")
        
        query = f"{search_name} IPO news"
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        results = []
        sentiment_scores = []
        gmp_estimates = []

        try:
            response = requests.get(rss_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                root = ElementTree.fromstring(response.text)
                items = root.findall('.//item')
                logging.info(f"Google News RSS returned {len(items)} items")

                for item in items[:10]: # Process top 10
                    title = item.find('title').text
                    link = item.find('link').text
                    pub_date = item.find('pubDate').text
                    
                    # Sentiment Analysis on the Title (since RSS titles are descriptive)
                    vs = self.analyzer.polarity_scores(title)
                    sentiment_scores.append(vs['compound'])
                    
                    # Extract GMP from title
                    self._extract_gmp(title, gmp_estimates)
                    
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': f"Published: {pub_date}",
                        'sentiment': vs['compound']
                    })
        except Exception as e:
            logging.error(f"Google News RSS error: {e}")

        # Final aggregation
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        valid_gmps = [g for g in gmp_estimates if abs(g) > 0]
        gmp_consensus = sum(valid_gmps) / len(valid_gmps) if valid_gmps else None
        
        logging.info(f"Pulse Results: Sentiment={avg_sentiment}, GMP={gmp_consensus}, News={len(results)}")

        return {
            'ipo_name': ipo_name,
            'avg_sentiment': avg_sentiment,
            'sentiment_label': self._get_label(avg_sentiment),
            'gmp_estimate': gmp_consensus,
            'news': results
        }

    def _extract_gmp(self, text, estimates):
        patterns = [
            r'GMP\s*(?:is|of|at|₹|Rs\.?\s*)?\s*([-+]?\d+)',
            r'Premium(?:\s*is)?\s*(?:of|at|₹|Rs\.?\s*)?\s*([-+]?\d+)',
            r'Grey Market(?:\s*Premium)?\s*(?:is|at|₹|Rs\.?\s*)?\s*([-+]?\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    gval = int(match.group(1))
                    if 0 < abs(gval) < 5000: 
                        estimates.append(gval)
                        return True
                except: continue
        return False

    def _get_label(self, score):
        if score > 0.15:
            return "Very Bullish" if score > 0.5 else "Bullish"
        elif score < -0.15:
            return "Very Bearish" if score < -0.5 else "Bearish"
        return "Neutral / Mixed"

    def calculate_adjustment_factor(self, sentiment_data):
        score = sentiment_data['avg_sentiment']
        multiplier = 1.0 + (score * 0.2)
        return multiplier

# Test standalone
if __name__ == "__main__":
    engine = IPOSentimentEngine()
    print(engine.fetch_market_pulse("Paytm"))
