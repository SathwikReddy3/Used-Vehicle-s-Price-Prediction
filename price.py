import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['make', 'model', 'fuel_type', 'condition'], drop_first=True)
    
    # Normalize numeric features
    numeric_cols = ['year', 'mileage']
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

# EDA
def perform_eda(df):
    print("Dataset Info:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlations")
    plt.show()

# Model Training
def train_models(X_train, y_train):
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return r2, mae, predictions

# Visualization
def visualize_results(y_test, predictions, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs Predicted Prices ({model_name})")
    plt.show()

# Main Function
def main():
    # Load and preprocess data
    file_path = "used_vehicles.csv"  # Replace with your dataset path
    df = load_data(file_path)
    df = preprocess_data(df)
    
    # Define features and target
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Perform EDA
    perform_eda(df)
    
    # Train models
    lr_model, rf_model = train_models(X_train, y_train)
    
    # Evaluate models
    for model, name in zip([lr_model, rf_model], ["Linear Regression", "Random Forest"]):
        r2, mae, predictions = evaluate_model(model, X_test, y_test)
        print(f"\n{name} Performance:")
        print(f"R² Score: {r2:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        
        # Visualize results
        visualize_results(y_test, predictions, name)

if __name__ == "__main__":
    main()
