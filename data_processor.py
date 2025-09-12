import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class IPODataProcessor:
    """
    Class to handle IPO data loading, cleaning, and preprocessing
    """
    
    def __init__(self):
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """
        Load IPO data from Excel file with proper column mapping
        """
        print("Loading IPO data from Excel...")
        
        try:
            # Load data from Excel
            df = pd.read_excel(path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            print(f"Original dataset shape: {df.shape}")
            print(f"Original columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def map_columns(self, df):
        """
        Map column names to standard format based on your data structure
        """
        # Column mapping based on your data structure
        column_mapping = {
            'Issuer Company': 'company_name',
            'Listing Date': 'listing_date',
            'Issue Price': 'issue_price',
            'Lot Size': 'lot_size',
            'P/E Ratio': 'pe_ratio',
            'QIB': 'qib_subscription',
            'NII': 'nii_subscription',
            'RII': 'rii_subscription',
            'EMP': 'emp_subscription',
            'TOTAL': 'total_subscription',
            'Open Price': 'open_price',
            'Low Price': 'low_price',
            'High Price': 'high_price',
            'Close Price': 'close_price',
            '% Change': 'listing_gains',
            '(Rs Cr)': 'issue_size_cr'
        }
        
        # Apply column mapping for existing columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                print(f"Mapped: '{old_col}' -> '{new_col}'")
        
        # Handle case where % Change might be the last column without a proper name
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols and 'listing_gains' not in df.columns:
            # Check if the unnamed column contains percentage-like values
            for col in unnamed_cols:
                if df[col].dtype in ['float64', 'int64'] and abs(df[col].mean()) < 200:
                    df = df.rename(columns={col: 'listing_gains'})
                    print(f"Mapped unnamed column '{col}' to 'listing_gains'")
                    break
        
        return df
    
    def clean_data(self, df):
        """
        Clean and convert data types
        """
        print("\nCleaning data...")
        
        # Calculate listing gains if not present
        if 'listing_gains' not in df.columns and 'close_price' in df.columns and 'issue_price' in df.columns:
            df['listing_gains'] = ((df['close_price'] - df['issue_price']) / df['issue_price']) * 100
            print("Calculated listing_gains from close_price and issue_price")
        
        # Define numeric columns for cleaning
        numeric_columns = [
            'issue_price', 'lot_size', 'pe_ratio', 
            'qib_subscription', 'nii_subscription', 'rii_subscription', 
            'emp_subscription', 'total_subscription',
            'open_price', 'low_price', 'high_price', 'close_price',
            'listing_gains', 'issue_size_cr'
        ]
        
        # Clean numeric columns
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, handling any non-numeric characters
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle listing date
        if 'listing_date' in df.columns:
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
            print(f"Converted listing_date to datetime")
        
        # Remove rows where target variable is missing
        initial_rows = len(df)
        df = df.dropna(subset=['listing_gains'])
        final_rows = len(df)
        print(f"Removed {initial_rows - final_rows} rows with missing listing_gains")
        
        return df
    
    def feature_engineering(self, df):
        """
        Create new features from existing data
        """
        print("\nEngineering features...")
        
        # Issue size features
        if 'issue_size_cr' in df.columns:
            df['issue_size_log'] = np.log1p(df['issue_size_cr'])
            print("Created: issue_size_log")
        
        # Date-based features
        if 'listing_date' in df.columns:
            df['listing_year'] = df['listing_date'].dt.year
            df['listing_month'] = df['listing_date'].dt.month
            df['listing_quarter'] = df['listing_date'].dt.quarter
            df['listing_day_of_week'] = df['listing_date'].dt.dayofweek
            print("Created: listing_year, listing_month, listing_quarter, listing_day_of_week")
        
        # Subscription-related features
        if 'total_subscription' in df.columns:
            # Subscription categories
            df['subscription_category'] = pd.cut(
                df['total_subscription'], 
                bins=[0, 1, 5, 10, 50, float('inf')], 
                labels=['Under_subscribed', 'Low', 'Medium', 'High', 'Very_High']
            )
            print("Created: subscription_category")
            
            # Over-subscription flag
            df['is_oversubscribed'] = (df['total_subscription'] > 1).astype(int)
            print("Created: is_oversubscribed")
        
        # Price-related features
        if all(col in df.columns for col in ['high_price', 'low_price', 'open_price']):
            # Listing day volatility
            df['listing_volatility'] = ((df['high_price'] - df['low_price']) / df['open_price']) * 100
            print("Created: listing_volatility")
        
        if all(col in df.columns for col in ['open_price', 'issue_price']):
            # Opening premium
            df['opening_premium'] = ((df['open_price'] - df['issue_price']) / df['issue_price']) * 100
            print("Created: opening_premium")
        
        # Subscription ratio features
        subscription_cols = ['qib_subscription', 'nii_subscription', 'rii_subscription']
        available_subs = [col for col in subscription_cols if col in df.columns]
        
        if len(available_subs) >= 2:
            # QIB to RII ratio
            if 'qib_subscription' in df.columns and 'rii_subscription' in df.columns:
                df['qib_rii_ratio'] = df['qib_subscription'] / (df['rii_subscription'] + 0.1)  # Adding small value to avoid division by zero
                print("Created: qib_rii_ratio")
            
            # Institutional vs Retail ratio
            if 'qib_subscription' in df.columns and 'nii_subscription' in df.columns and 'rii_subscription' in df.columns:
                df['institutional_ratio'] = (df['qib_subscription'] + df['nii_subscription']) / (df['rii_subscription'] + 0.1)
                print("Created: institutional_ratio")
        
        # Price efficiency features
        if 'pe_ratio' in df.columns and 'issue_price' in df.columns:
            df['price_pe_ratio'] = df['issue_price'] / (df['pe_ratio'] + 0.1)
            print("Created: price_pe_ratio")
        
        # Market timing features (basic market conditions)
        if 'listing_year' in df.columns:
            # Create market period categories (you can adjust these based on Indian market cycles)
            market_periods = {
                2020: 'COVID_Recovery',
                2021: 'Bull_Market',
                2022: 'Volatile_Market',
                2023: 'Recovery_Market',
                2024: 'Current_Market'
            }
            df['market_period'] = df['listing_year'].map(market_periods).fillna('Other')
            print("Created: market_period")
        
        return df
    
    def create_sector_classification(self, df):
        """
        Create sector classification based on company names and keywords
        """
        if 'company_name' not in df.columns:
            return df
        
        print("\nClassifying sectors based on company names...")
        
        # Initialize sector column
        df['sector'] = 'Others'
        
        # Define sector keywords
        sector_keywords = {
            'Technology': ['tech', 'software', 'digital', 'cyber', 'data', 'IT', 'systems', 'solutions', 'infotech', 'computers'],
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
        
        # Classify sectors
        for sector, keywords in sector_keywords.items():
            mask = df['company_name'].str.lower().str.contains('|'.join(keywords), na=False)
            df.loc[mask, 'sector'] = sector
        
        # Print sector distribution
        sector_counts = df['sector'].value_counts()
        print("Sector distribution:")
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count}")
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features for machine learning
        """
        print("\nEncoding categorical features...")
        
        categorical_features = ['sector', 'subscription_category', 'market_period']
        
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded: {col} -> {col}_encoded")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        """
        print("\nHandling missing values...")
        
        # Check missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            print("Missing values per column:")
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count} ({percentage:.1f}%)")
            
            # Fill missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Fill numeric columns with median
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
            
            # Fill categorical columns with mode
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def process_data(self, file_path):
        """
        Complete data processing pipeline
        """
        print("="*60)
        print("IPO DATA PROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Process data step by step
        df = self.map_columns(df)
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        df = self.create_sector_classification(df)
        df = self.encode_categorical_features(df)
        df = self.handle_missing_values(df)
        
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Final columns: {len(df.columns)}")
        
        # Save processed data
        output_file = file_path.replace('.xlsx', '_processed.csv').replace('.xls', '_processed.csv')
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")
        
        return df

# Example usage
if __name__ == "__main__":
    processor = IPODataProcessor()
    
    # Process your IPO data
    # df = processor.process_data('your_ipo_data.xlsx')
    
    print("Data processor ready to use!")
    print("Usage: processor.process_data('your_excel_file.xlsx')")
