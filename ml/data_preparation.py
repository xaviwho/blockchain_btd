"""
data_preparation.py

This module handles data preparation and feature engineering for the nerve agent detection ML models.
It processes the time-series acetylcholinesterase-based sensor data for training and inference.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os


def load_sensor_data(data_path):
    """
    Load sensor data from a JSON file
    
    Args:
        data_path (str): Path to the sensor data JSON file
        
    Returns:
        list: Loaded sensor data
    """
    try:
        with open(data_path, 'r') as f:
            sensor_data = json.load(f)
        print(f"Loaded {len(sensor_data)} sensor readings from {data_path}")
        return sensor_data
    except Exception as e:
        print(f"Error loading sensor data: {e}")
        return []


def extract_features(df):
    """
    Extract and engineer features from the sensor data
    
    Args:
        df (DataFrame): DataFrame containing sensor readings
        
    Returns:
        DataFrame: DataFrame with engineered features
    """
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create additional features based on biochemical knowledge
    # Use reaction_rate directly as it's already in the data
    df['reaction_rate_normalized'] = df['reaction_rate'] / df['reaction_rate'].max()
    
    # Create a binary label for threat detection (high concentration = threat)
    threshold = df['concentration'].quantile(0.7)  # Top 30% are threats
    df['is_threat'] = (df['concentration'] > threshold).astype(int)
    
    print(f"Using concentration threshold of {threshold:.4f} for threat classification")
    print(f"Number of threat samples: {df['is_threat'].sum()} out of {len(df)}")
    
    return df


def create_train_test_data(df, test_size=0.2):
    """
    Create training and test data
    
    Args:
        df (DataFrame): DataFrame with features
        test_size (float): Fraction of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Use concentration and reaction_rate as features
    X = df[['concentration', 'reaction_rate', 'reaction_rate_normalized']].values
    y = df['is_threat'].values
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    print(f"Created training set with {len(X_train)} samples and test set with {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test=None):
    """
    Normalize numerical features to [0,1] range
    
    Args:
        X_train: Training data
        X_test: Test data (optional)
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    # Create and fit scaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Scale test data if provided
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_sensor_data(data_path, test_size=0.2):
    """
    Prepare sensor data for ML model training
    
    Args:
        data_path (str): Path to the sensor data JSON file
        test_size (float): Fraction of data to use for validation
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler)
    """
    # Load data
    sensor_data = load_sensor_data(data_path)
    
    # Convert to DataFrame
    df = pd.DataFrame(sensor_data)
    
    # Ensure required columns exist
    required_columns = ['sensor_id', 'concentration', 'timestamp', 'reaction_rate']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract features
    df = extract_features(df)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_data(df, test_size)
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_scaler(scaler, output_path="models/concentration_scaler.pkl"):
    """
    Save the fitted scaler for later use in inference
    
    Args:
        scaler: Fitted MinMaxScaler
        output_path (str): Path to save the scaler
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {output_path}")


def load_scaler(file_path="models/concentration_scaler.pkl"):
    """
    Load a previously fitted scaler
    
    Args:
        file_path (str): Path to the saved scaler
        
    Returns:
        MinMaxScaler: Loaded scaler
    """
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def preprocess_inference_data(reading, scaler):
    """
    Preprocess a single sensor reading for inference
    
    Args:
        reading (dict): Sensor reading
        scaler (MinMaxScaler): Fitted scaler
        
    Returns:
        np.array: Processed data ready for inference
    """
    # Extract features
    features = np.array([[
        reading['concentration'],
        reading['reaction_rate'],
        reading['reaction_rate'] / 1.0  # Normalize assuming max is 1.0
    ]])
    
    # Normalize
    features_scaled = scaler.transform(features)
    
    return features_scaled


if __name__ == "__main__":
    # Example usage
    data_path = "/home/vboxuser/blockchain_btd/data/sensor_data.json"
    X_train, X_val, y_train, y_val, scaler = prepare_sensor_data(data_path)
    
    # Save the scaler for later use
    save_scaler(scaler)
    
    print("Data preparation complete!")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")