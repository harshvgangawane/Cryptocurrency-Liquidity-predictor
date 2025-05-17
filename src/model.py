import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.feature import engineer_features

def train_and_save_model(data_path, model_dir='models'):
    # Ensure the correct file path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', data_path)
    df = pd.read_csv(data_path)

    # Apply feature engineering
    df = engineer_features(df)

    FEATURES = [
        'price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
        'log_mkt_cap', 'log_24h_volume'
    ]

    X = df[FEATURES]
    y = df['Liquidity']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Random Forest model
    rf_model = RandomForestRegressor(random_state=42)

    # Perform GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    gridsearch_cv = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
    gridsearch_cv.fit(X_train_scaled, y_train)

    # Best model
    best_model = gridsearch_cv.best_estimator_
    print(f"Best Parameters: {gridsearch_cv.best_params_}")
    print(f"Test R^2 score: {best_model.score(X_test_scaled, y_test):.4f}")

    # Save model and scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

if __name__ == "__main__":
    train_and_save_model("notebook/data/coin_gecko_2022-03-17_cleaned.csv")

    # Re-save the model and scaler explicitly
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    # Load the saved model and scaler
    model_path = os.path.join('models', 'model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Re-save using the current version of joblib
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)