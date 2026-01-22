"""
Visualization Module.

Generates pair plots for the processed dataset to analyze distributions
and correlations before and after log transformation of the price.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = 'processed_dataset.csv'
IMG_DIR = 'FinalReportImages'

def create_output_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_visualizations(df: pd.DataFrame):
    """Generates and saves pair plots."""
    
    # Take a sample for visualization to avoid performance issues
    df_sample = df.sample(100, random_state=42)
    
    print("Generating Pair Plot (Original Scale)...")
    sns.pairplot(df_sample)
    plt.savefig(os.path.join(IMG_DIR, 'pair_plot.png'))
    plt.close() # Close plot to free memory
    
    print("Generating Pair Plot (Log Price Scale)...")
    df_sample['price'] = np.log(df_sample['price'])
    sns.pairplot(df_sample)
    plt.savefig(os.path.join(IMG_DIR, 'log_price_pair_plot.png'))
    plt.close()
    
    print(f"Images saved to {IMG_DIR}/")

def main():
    try:
        create_output_dir(IMG_DIR)
        
        print("Loading data for visualization...")
        df = pd.read_csv(DATA_PATH)
        
        # Basic type correction
        df['year'] = df['year'].astype('int64')
        if 'id' in df.columns:
            df.drop('id', axis=1, inplace=True)
            
        generate_visualizations(df)
        
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Run data_cleaning.py first.")

if __name__ == "__main__":
    main()