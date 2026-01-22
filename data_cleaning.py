"""
Data Cleaning Module for Used Car Price Prediction.

This script loads the raw Kaggle dataset, performs feature selection,
imputes missing values using IterativeImputer, removes outliers via IQR,
and saves the processed dataset to a CSV file.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn import preprocessing

# Suppress chained assignment warnings for cleaner output
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# Constants
RAW_DATA_PATH = 'vehicles.csv'
PROCESSED_DATA_PATH = 'processed_dataset.csv'

IRRELEVANT_COLUMNS = [
    'url', 'region_url', 'VIN', 'image_url', 'description', 
    'county', 'state', 'posting_date', 'lat', 'long'
]
NUMERIC_COLS = ['year', 'odometer']
CATEGORICAL_COLS = [
    'region', 'manufacturer', 'model', 'condition', 'cylinders', 
    'fuel', 'title_status', 'transmission', 'drive', 'size', 
    'type', 'paint_color'
]

def load_and_trim_data(filepath: str) -> pd.DataFrame:
    """Loads CSV and drops irrelevant columns."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.drop(columns=IRRELEVANT_COLUMNS, errors='ignore')
    return df

def impute_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing numeric values using ExtraTreesRegressor."""
    print("Imputing numeric data (this may take a few minutes)...")
    imputer = IterativeImputer(
        ExtraTreesRegressor(n_estimators=10, random_state=0)
    )
    imputed_data = imputer.fit_transform(df[NUMERIC_COLS])
    df[NUMERIC_COLS] = imputed_data
    return df

def impute_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes categorical data by encoding, imputing with BayesianRidge, 
    and decoding back to strings.
    """
    print("Imputing categorical data...")
    char_encoder = preprocessing.LabelEncoder()
    
    # Process each column individually
    for col in CATEGORICAL_COLS:
        # Encode non-nulls
        series = df[col]
        non_nulls = np.array(series.dropna())
        impute_ord = char_encoder.fit_transform(non_nulls.reshape(-1, 1))
        
        # Temporarily fill non-nulls with encoded values
        series.loc[series.notnull()] = np.squeeze(impute_ord)
        
        # Impute missing values
        imputer = IterativeImputer(BayesianRidge())
        imputed_col = imputer.fit_transform(series.values.reshape(-1, 1))
        
        # Decode back to original string labels
        imputed_col = imputed_col.astype('int64')
        decoded_col = char_encoder.inverse_transform(imputed_col.ravel())
        
        df[col] = decoded_col
        
    return df

def remove_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Removes outliers based on 1.5 * IQR."""
    print("Removing outliers...")
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iq_range = q3 - q1
        lower_bound = q1 - 1.5 * iq_range
        upper_bound = q3 + 1.5 * iq_range
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Ensure no zero-price cars
    df = df[df['price'] != 0]
    return df

def main():
    try:
        data = load_and_trim_data(RAW_DATA_PATH)
        
        # 1. Impute Numeric
        data = impute_numeric_data(data)
        
        # 2. Impute Categorical
        data = impute_categorical_data(data)
        
        # 3. Remove Outliers
        data = remove_outliers(data, ['odometer', 'year'])
        
        print(f"Final dataset shape: {data.shape}")
        data.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Successfully saved processed data to {PROCESSED_DATA_PATH}")

    except FileNotFoundError:
        print(f"Error: Could not find {RAW_DATA_PATH}. Please ensure it exists.")

if __name__ == "__main__":
    main()