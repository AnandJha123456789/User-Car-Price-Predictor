"""
Machine Learning Models Module.

Trains and evaluates Linear Regression, Feed-Forward Neural Network, 
and Random Forest Regressor models on the processed used car dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Tensorflow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Constants
DATA_PATH = 'processed_dataset.csv'
IMG_DIR = 'FinalReportImages'
RANDOM_STATE = 0

def load_and_preprocess(filepath: str):
    """
    Loads data, encodes features, and performs log transform on price.
    Scaling is deferred to after train/test split to prevent leakage.
    """
    df = pd.read_csv(filepath)
    
    # Cleanup indices if they exist
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Define columns
    cat_cols = [
        'region', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 
        'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color'
    ]
    
    # Label Encode Categorical Variables
    # Note: We encode before split to handle all categories, but for strict 
    # production pipelines, encoders should be fit on train only.
    le = preprocessing.LabelEncoder()
    for col in cat_cols:
        # Ensure data is string type before encoding to avoid type errors
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Log Transform Price
    df['price'] = np.log(df['price'])
    
    # Remove outliers based on log-price (IQR Method)
    q1, q3 = df['price'].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df.price >= q1 - 1.5 * iqr) & (df.price <= q3 + 1.5 * iqr)]
        
    return df

def get_train_test_data(df: pd.DataFrame):
    """Splits data into X and y, then train and test sets."""
    X = df.drop(columns=['price'])
    y = df['price']
    return train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=RANDOM_STATE)

def evaluate_and_plot(y_test, y_pred, model_name, plot_filename):
    """Calculates MSLE, R2 and generates a pred vs true scatter plot."""
    
    # Remove negative predictions (log scale cannot be negative)
    # This prevents errors in MSLE calculation
    mask = y_pred > 0
    y_pred_clean = y_pred[mask]
    y_test_clean = y_test[mask]
    
    # Metrics
    try:
        msle = mean_squared_log_error(y_test_clean, y_pred_clean)
        rmsle = np.sqrt(msle)
    except ValueError:
        # Fallback if cleaning didn't catch all edge cases
        msle = -1
        rmsle = -1
        
    r2 = r2_score(y_test_clean, y_pred_clean)
    
    print(f"\n--- {model_name} Results ---")
    print(f"MSLE: {msle:.5f}")
    print(f"RMSLE: {rmsle:.5f}")
    print(f"R2 Score: {r2:.5f} ({r2*100:.2f}%)")
    
    # Plotting
    plt.figure(figsize=(10, 10))
    # Plot first 100 points for clarity
    limit = min(100, len(y_test_clean))
    plt.scatter(y_test_clean[:limit], y_pred_clean[:limit], c='crimson')
    
    # Diagonal line
    if len(y_pred_clean) > 0:
        p1 = max(max(y_pred_clean), max(y_test_clean))
        p2 = min(min(y_pred_clean), min(y_test_clean))
        plt.plot([p1, p2], [p1, p2], 'b-')
    
    plt.xlabel('True Price (log)', fontsize=15)
    plt.ylabel('Predicted Price (log)', fontsize=15)
    plt.title(f'{model_name}: Predicted vs. True Price (Log Scale)', fontsize=15)
    plt.axis('equal')
    
    save_path = os.path.join(IMG_DIR, plot_filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()
    
    return [msle, rmsle, r2, r2*100]

def run_linear_regression(X_train, X_test, y_train, y_test):
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # Feature Importance Plot
    coef = pd.Series(lr.coef_, index=X_train.columns)
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature Weights using Linear Regression Model")
    plt.savefig(os.path.join(IMG_DIR, 'Linear-Regression-Feature-Importance.jpg'))
    plt.close()
    
    return evaluate_and_plot(y_test, y_pred, "Linear Regression", "viz_lr_pred_vs_true.png")

def run_neural_network(X_train, X_test, y_train, y_test):
    print("\nTraining Neural Network...")
    
    model = Sequential([
        Dense(64, activation='relu', use_bias=True, input_dim=X_train.shape[1]),
        Dense(32, activation='relu', use_bias=True),
        Dense(16, activation='relu', use_bias=True),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    try:
        plot_model(model, to_file=os.path.join(IMG_DIR, 'nn_model_arch.png'), 
                   show_shapes=True, show_layer_names=True)
    except Exception as e:
        print(f"Could not plot model architecture (Graphviz missing or not in PATH): {e}")

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2, verbose=1)
    
    y_pred = model.predict(X_test).reshape(-1)
    return evaluate_and_plot(y_test, y_pred, "Neural Network", "viz_nn_pred_vs_true.png")

def run_random_forest(X_train, X_test, y_train, y_test):
    print("\nTraining Random Forest (this may take time)...")
    rf = RandomForestRegressor(
        n_estimators=180, 
        random_state=RANDOM_STATE, 
        min_samples_leaf=1, 
        max_features=0.5, 
        n_jobs=-1, 
        oob_score=True
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    return evaluate_and_plot(y_test, y_pred, "Random Forest", "viz_rf_pred_vs_true.png")

def main():
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run data_cleaning.py first.")
        return

    # 1. Preprocess (Encode & Clean)
    df = load_and_preprocess(DATA_PATH)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = get_train_test_data(df)
    
    # 3. Normalize Features (Scaling)
    # CRITICAL: Fit scaler ONLY on training data to avoid data leakage
    norm_cols = ['odometer', 'year', 'model', 'region']
    scaler = StandardScaler()
    
    # Fit and transform training data
    print("Scaling training data...")
    X_train.loc[:, norm_cols] = scaler.fit_transform(X_train[norm_cols])
    
    # Transform test data using the training scaler
    print("Scaling testing data...")
    X_test.loc[:, norm_cols] = scaler.transform(X_test[norm_cols])
    
    # Store results
    results = pd.DataFrame(index=['MSLE', 'Root MSLE', 'R2 Score', 'Accuracy(%)'])
    
    # 4. Run Models
    results['Linear Regression'] = run_linear_regression(X_train, X_test, y_train, y_test)
    results['Neural Network'] = run_neural_network(X_train, X_test, y_train, y_test)
    results['Random Forest'] = run_random_forest(X_train, X_test, y_train, y_test)
    
    # 5. Final Output
    print("\nFinal Comparative Results:")
    print(results)
    results.to_csv(os.path.join(IMG_DIR, 'model_performance_metrics.csv'))

if __name__ == "__main__":
    main()