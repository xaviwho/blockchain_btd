"""
train_sklearn.py

This script trains and evaluates scikit-learn models for nerve agent detection.
"""

import os
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
import pandas as pd

# Import from other modules
from data_preparation import prepare_sensor_data, save_scaler
from models_sklearn import get_model_by_name, save_model

# Constants
SEED = 42
MODEL_TYPES = ['rf', 'gb', 'svm', 'knn', 'mlp']
DATA_PATH = "/home/vboxuser/blockchain_btd/data/sensor_data.json"
MODELS_DIR = "models"
RESULTS_DIR = "results"

def setup_directories():
    """Create necessary directories for saving models and results"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

def train_model(model, X_train, y_train, model_name):
    """
    Train a scikit-learn model
    
    Args:
        model: The model to train
        X_train, y_train: Training data
        model_name: Name of the model
        
    Returns:
        trained_model: The trained model
        elapsed_time: Training time in seconds
    """
    print(f"\nTraining {model_name} model...")
    
    # Record start time
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print(f"{model_name} training completed in {elapsed_time:.2f} seconds")
    
    return model, elapsed_time

def evaluate_model(model, X_val, y_val, model_name):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_val, y_val: Validation data
        model_name: Name of the model
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Predict on validation data
    y_pred = model.predict(X_val)
    
    # For ROC curve
    try:
        y_scores = model.predict_proba(X_val)[:, 1]
    except:
        # Some models don't support predict_proba
        y_scores = model.decision_function(X_val) if hasattr(model, 'decision_function') else y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # Print results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    
    classes = ['No Threat', 'Threat']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_val, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{model_name}_roc_curve.png"))
    plt.close()
    
    # Collect metrics
    metrics = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'roc_auc': roc_auc
    }
    
    return metrics

def evaluate_models(models, X_val, y_val):
    """
    Evaluate all models and save results
    
    Args:
        models: Dictionary of trained models
        X_val, y_val: Validation data
        
    Returns:
        results: Dictionary of evaluation results
    """
    print("\nEvaluating models...")
    
    results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val, model_name)
        results[model_name] = metrics
    
    # Save results to file
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_model_comparison(results):
    """
    Create comparison plot of the models
    
    Args:
        results: Dictionary of evaluation results
    """
    # Extract data
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    precisions = [results[m]['precision'] for m in model_names]
    recalls = [results[m]['recall'] for m in model_names]
    f1_scores = [results[m]['f1'] for m in model_names]
    roc_aucs = [results[m].get('roc_auc', 0) for m in model_names]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]  # Descending
    sorted_models = [model_names[i] for i in sorted_indices]
    
    # Create bar chart for accuracy
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(sorted_models))
    width = 0.15
    
    plt.bar(x - width*2, [accuracies[sorted_indices.tolist().index(model_names.index(m))] for m in sorted_models], 
            width, label='Accuracy', color='skyblue')
    plt.bar(x - width, [precisions[sorted_indices.tolist().index(model_names.index(m))] for m in sorted_models], 
            width, label='Precision', color='lightgreen')
    plt.bar(x, [recalls[sorted_indices.tolist().index(model_names.index(m))] for m in sorted_models], 
            width, label='Recall', color='salmon')
    plt.bar(x + width, [f1_scores[sorted_indices.tolist().index(model_names.index(m))] for m in sorted_models], 
            width, label='F1', color='purple')
    plt.bar(x + width*2, [roc_aucs[sorted_indices.tolist().index(model_names.index(m))] for m in sorted_models], 
            width, label='ROC AUC', color='gold')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, sorted_models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "model_comparison.png"))
    plt.close()

def main():
    """Main training pipeline"""
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    # Create directories
    setup_directories()
    
    print("=== Nerve Agent Detection Model Training (Scikit-Learn) ===")
    print(f"Using data from: {DATA_PATH}")
    
    # Prepare data
    X_train, X_val, y_train, y_val, scaler = prepare_sensor_data(
        DATA_PATH
    )
    
    # Save scaler for inference
    save_scaler(scaler, os.path.join(MODELS_DIR, "concentration_scaler.pkl"))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Train each model type
    models = {}
    training_times = {}
    
    for model_type in MODEL_TYPES:
        # Create model
        model = get_model_by_name(model_type)
        
        # Train model
        trained_model, elapsed_time = train_model(model, X_train, y_train, model_type)
        
        # Save model
        save_model(trained_model, model_type)
        
        # Store model and metrics
        models[model_type] = trained_model
        training_times[model_type] = elapsed_time
    
    # Evaluate all models
    results = evaluate_models(models, X_val, y_val)
    
    # Plot model comparison
    plot_model_comparison(results)
    
    print("\nTraining complete! Models saved to:", MODELS_DIR)
    print("Results saved to:", RESULTS_DIR)
    
    # Print training times
    print("\nTraining times:")
    for model_type, elapsed_time in training_times.items():
        print(f"  {model_type}: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()