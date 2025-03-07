"""
models_sklearn.py

This module defines scikit-learn based models for nerve agent detection.
These models are more compatible with limited CPU environments.
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_random_forest_model():
    """Create a Random Forest model for sequence classification"""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

def create_gradient_boosting_model():
    """Create a Gradient Boosting model for sequence classification"""
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

def create_svm_model():
    """Create an SVM model for sequence classification"""
    return SVC(
        probability=True,
        kernel='rbf',
        random_state=42
    )

def create_knn_model():
    """Create a K-Nearest Neighbors model for sequence classification"""
    return KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    )

def create_mlp_model():
    """Create a simple neural network using MLP"""
    return MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=500,
        random_state=42
    )

def save_model(model, model_name, models_dir="models"):
    """Save a scikit-learn model to file"""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Saved model to {model_path}")
    
    # Get model size
    model_size = os.path.getsize(model_path) / 1024  # KB
    print(f"Model size: {model_size:.2f} KB")
    
    return model_size

def load_model(model_name, models_dir="models"):
    """Load a scikit-learn model from file"""
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded model from {model_path}")
    return model

def get_model_by_name(model_name):
    """
    Get a model by name
    
    Args:
        model_name (str): Name of the model ('rf', 'gb', 'svm', 'knn', 'mlp')
        
    Returns:
        model: The requested scikit-learn model
    """
    models = {
        'rf': create_random_forest_model,
        'gb': create_gradient_boosting_model,
        'svm': create_svm_model,
        'knn': create_knn_model,
        'mlp': create_mlp_model
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name]()

if __name__ == "__main__":
    # Example: Create and print each model
    for model_name in ['rf', 'gb', 'svm', 'knn', 'mlp']:
        model = get_model_by_name(model_name)
        print(f"\n{model_name.upper()} Model:")
        print(model)