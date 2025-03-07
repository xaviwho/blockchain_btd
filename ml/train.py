"""
train.py

This script trains and evaluates different models for nerve agent detection.
It loads sensor data, trains multiple model architectures, and evaluates their performance.
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Import from other modules
from data_preparation import prepare_sensor_data, save_scaler
from models import get_model_by_name, convert_to_tflite

# Constants
SEED = 42
SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 50
MODEL_TYPES = ['cnn', 'lstm', 'hybrid', 'tiny']  # 'bilstm' is excluded for resource conservation
DATA_PATH = "./data/sensor_data.json"
MODELS_DIR = "models"
RESULTS_DIR = "results"


def setup_directories():
    """Create necessary directories for saving models and results"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=EPOCHS):
    """
    Train a model with early stopping and learning rate reduction
    
    Args:
        model: The model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_name: Name of the model (for saving)
        epochs: Maximum number of epochs
        
    Returns:
        history: Training history
        elapsed_time: Training time in seconds
    """
    print(f"\nTraining {model_name} model...")
    
    # Define callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # Reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001
        ),
        # Model checkpoint to save the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Record start time
    start_time = time.time()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print training results
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"{model_name} training completed in {elapsed_time:.2f} seconds")
    print(f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")
    
    return history, elapsed_time


def save_model_and_tflite(model, model_name):
    """
    Save the model in both Keras and TFLite formats
    
    Args:
        model: The trained model
        model_name: Name of the model
    """
    # Save Keras model
    keras_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")
    
    # Convert and save TFLite model
    tflite_path = os.path.join(MODELS_DIR, f"{model_name}.tflite")
    convert_to_tflite(model, quantize=True, output_path=tflite_path)
    
    # Get model size
    keras_size = os.path.getsize(keras_path) / 1024  # KB
    tflite_size = os.path.getsize(tflite_path) / 1024  # KB
    
    print(f"Model sizes - Keras: {keras_size:.2f} KB, TFLite: {tflite_size:.2f} KB")
    
    return keras_size, tflite_size


def plot_training_history(history, model_name):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history
        model_name: Name of the model
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{model_name}_history.png"))
    plt.close()


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
        start_time = time.time()
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        inference_time = time.time() - start_time
        
        results[model_name] = {
            'accuracy': accuracy,
            'loss': loss,
            'inference_time': inference_time
        }
        
        print(f"{model_name}: Accuracy={accuracy:.4f}, Inference time={inference_time:.4f}s")
    
    # Save results to file
    with open(os.path.join(RESULTS_DIR, "evaluation_results.txt"), "w") as f:
        f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name.upper()} Model:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Inference time: {metrics['inference_time']:.4f} seconds\n\n")
    
    return results


def plot_model_comparison(results, model_sizes):
    """
    Create comparison plots of the models
    
    Args:
        results: Dictionary of evaluation results
        model_sizes: Dictionary of model sizes
    """
    # Extract data
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    inference_times = [results[m]['inference_time'] for m in model_names]
    tflite_sizes = [model_sizes[m]['tflite'] for m in model_names]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy vs size
    ax1.scatter(tflite_sizes, accuracies, s=100, alpha=0.7)
    for i, model in enumerate(model_names):
        ax1.annotate(model, (tflite_sizes[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax1.set_title('Model Accuracy vs Size')
    ax1.set_xlabel('Model Size (KB)')
    ax1.set_ylabel('Validation Accuracy')
    ax1.grid(True)
    
    # Plot accuracy vs inference time
    ax2.scatter(inference_times, accuracies, s=100, alpha=0.7)
    for i, model in enumerate(model_names):
        ax2.annotate(model, (inference_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_title('Model Accuracy vs Inference Time')
    ax2.set_xlabel('Inference Time (seconds)')
    ax2.set_ylabel('Validation Accuracy')
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "model_comparison.png"))
    plt.close()


def main():
    """Main training pipeline"""
    # Set random seeds for reproducibility
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Create directories
    setup_directories()
    
    print("=== Nerve Agent Detection Model Training ===")
    print(f"Using data from: {DATA_PATH}")
    
    # Prepare data
    X_train, X_val, y_train, y_val, scaler = prepare_sensor_data(
        DATA_PATH, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Save scaler for inference
    save_scaler(scaler, os.path.join(MODELS_DIR, "concentration_scaler.pkl"))
    
    # Get input shape from data
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")
    
    # Train each model type
    models = {}
    histories = {}
    training_times = {}
    model_sizes = {}
    
    for model_type in MODEL_TYPES:
        # Create model
        model = get_model_by_name(model_type, input_shape)
        
        # Train model
        history, elapsed_time = train_model(model, X_train, y_train, X_val, y_val, model_type)
        
        # Save model
        keras_size, tflite_size = save_model_and_tflite(model, model_type)
        
        # Plot training history
        plot_training_history(history, model_type)
        
        # Store model and metrics
        models[model_type] = model
        histories[model_type] = history
        training_times[model_type] = elapsed_time
        model_sizes[model_type] = {'keras': keras_size, 'tflite': tflite_size}
    
    # Evaluate all models
    results = evaluate_models(models, X_val, y_val)
    
    # Plot model comparison
    plot_model_comparison(results, model_sizes)
    
    print("\nTraining complete! Models saved to:", MODELS_DIR)
    print("Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()