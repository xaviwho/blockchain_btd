"""
evaluation.py

This module evaluates nerve agent detection models under various network conditions.
It simulates disruptions like packet loss and evaluates model resilience.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import tensorflow as tf
import pickle
import time

# Import from other modules
from data_preparation import prepare_sensor_data, preprocess_node_data, load_scaler
from models import get_model_by_name, convert_to_tflite

# Constants
MODELS_DIR = "models"
RESULTS_DIR = "results"
DATA_PATH = "./data/sensor_data.json"
SEQUENCE_LENGTH = 10
MODEL_TYPES = ['cnn', 'lstm', 'hybrid', 'tiny']
PACKET_LOSS_RATES = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
NODE_FAILURE_RATES = [0, 0.1, 0.2, 0.3]


def setup_directories():
    """Create necessary directories for results"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "disruption"), exist_ok=True)


def load_models():
    """
    Load all models for evaluation
    
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    
    for model_type in MODEL_TYPES:
        try:
            # Load Keras model
            model_path = os.path.join(MODELS_DIR, f"{model_type}.h5")
            if os.path.exists(model_path):
                models[model_type] = tf.keras.models.load_model(model_path)
                print(f"Loaded {model_type} model from {model_path}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    return models


def simulate_packet_loss(X, loss_rate):
    """
    Simulate packet loss by randomly setting features to zero
    
    Args:
        X: Input data
        loss_rate: Probability of losing a packet
        
    Returns:
        X_disrupted: Data with simulated packet loss
    """
    if loss_rate == 0:
        return X
    
    X_disrupted = X.copy()
    
    # Generate random mask (1 = keep, 0 = lost)
    mask = np.random.random(X_disrupted.shape) >= loss_rate
    
    # Apply mask
    X_disrupted = X_disrupted * mask
    
    return X_disrupted


def simulate_node_failure(X, failure_rate):
    """
    Simulate node failure by setting entire sequences to zero
    
    Args:
        X: Input data
        failure_rate: Probability of node failure
        
    Returns:
        X_disrupted: Data with simulated node failures
    """
    if failure_rate == 0:
        return X
    
    X_disrupted = X.copy()
    
    # Randomly select sequences to zero out
    num_sequences = X_disrupted.shape[0]
    num_failures = int(num_sequences * failure_rate)
    
    if num_failures > 0:
        # Get random indices
        failure_indices = np.random.choice(num_sequences, num_failures, replace=False)
        
        # Set sequences to zero
        X_disrupted[failure_indices] = 0
    
    return X_disrupted


def evaluate_under_disruption(models, X_val, y_val):
    """
    Evaluate model performance under various disruption conditions
    
    Args:
        models: Dictionary of models to evaluate
        X_val, y_val: Validation data
        
    Returns:
        dict: Results for all models under different conditions
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} model under disruption...")
        model_results = {
            'packet_loss': {},
            'node_failure': {},
            'combined': {}
        }
        
        # Evaluate under packet loss
        for loss_rate in PACKET_LOSS_RATES:
            X_disrupted = simulate_packet_loss(X_val, loss_rate)
            loss, accuracy = model.evaluate(X_disrupted, y_val, verbose=0)
            
            model_results['packet_loss'][loss_rate] = {
                'loss': float(loss),
                'accuracy': float(accuracy)
            }
            
            print(f"  Packet loss {loss_rate*100:.0f}%: Accuracy {accuracy:.4f}")
        
        # Evaluate under node failure
        for failure_rate in NODE_FAILURE_RATES:
            X_disrupted = simulate_node_failure(X_val, failure_rate)
            loss, accuracy = model.evaluate(X_disrupted, y_val, verbose=0)
            
            model_results['node_failure'][failure_rate] = {
                'loss': float(loss),
                'accuracy': float(accuracy)
            }
            
            print(f"  Node failure {failure_rate*100:.0f}%: Accuracy {accuracy:.4f}")
        
        # Evaluate under combined disruption
        for loss_rate in [0, 0.2, 0.4]:
            for failure_rate in [0, 0.2]:
                X_disrupted = X_val.copy()
                X_disrupted = simulate_packet_loss(X_disrupted, loss_rate)
                X_disrupted = simulate_node_failure(X_disrupted, failure_rate)
                
                loss, accuracy = model.evaluate(X_disrupted, y_val, verbose=0)
                
                key = f"PL{loss_rate:.1f}_NF{failure_rate:.1f}"
                model_results['combined'][key] = {
                    'loss': float(loss),
                    'accuracy': float(accuracy),
                    'packet_loss_rate': loss_rate,
                    'node_failure_rate': failure_rate
                }
                
                print(f"  Combined PL {loss_rate*100:.0f}%, NF {failure_rate*100:.0f}%: Accuracy {accuracy:.4f}")
        
        results[model_name] = model_results
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "disruption", "disruption_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_inference_time(models, X_val, batch_sizes=[1, 10, 50]):
    """
    Evaluate model inference time for different batch sizes
    
    Args:
        models: Dictionary of models
        X_val: Validation data
        batch_sizes: List of batch sizes to test
        
    Returns:
        dict: Inference time results
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating inference time for {model_name} model...")
        model_results = {}
        
        for batch_size in batch_sizes:
            # Create batch
            if batch_size <= len(X_val):
                batch = X_val[:batch_size]
                
                # Warm-up run
                _ = model.predict(batch, verbose=0)
                
                # Timed run
                start_time = time.time()
                _ = model.predict(batch, verbose=0)
                end_time = time.time()
                
                inference_time = end_time - start_time
                time_per_sample = inference_time / batch_size
                
                model_results[batch_size] = {
                    'total_time': float(inference_time),
                    'time_per_sample': float(time_per_sample)
                }
                
                print(f"  Batch size {batch_size}: {inference_time:.4f}s total, {time_per_sample*1000:.2f}ms per sample")
        
        results[model_name] = model_results
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "disruption", "inference_time_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_tflite_models(X_val, y_val, batch_size=10):
    """
    Evaluate TFLite models
    
    Args:
        X_val, y_val: Validation data
        batch_size: Number of samples to evaluate
        
    Returns:
        dict: Results for TFLite models
    """
    results = {}
    
    for model_type in MODEL_TYPES:
        tflite_path = os.path.join(MODELS_DIR, f"{model_type}.tflite")
        
        if not os.path.exists(tflite_path):
            print(f"TFLite model for {model_type} not found at {tflite_path}")
            continue
        
        print(f"\nEvaluating TFLite model for {model_type}...")
        
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare test data
            test_data = X_val[:batch_size]
            true_labels = y_val[:batch_size]
            
            # Run inference
            predictions = []
            inference_times = []
            
            for i in range(len(test_data)):
                input_data = np.expand_dims(test_data[i], axis=0).astype(np.float32)
                
                # Run inference
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                end_time = time.time()
                
                predictions.append(output[0][0])
                inference_times.append(end_time - start_time)
            
            # Convert predictions to binary classes
            binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
            
            # Calculate accuracy
            correct = sum([1 for i in range(len(binary_preds)) if binary_preds[i] == true_labels[i]])
            accuracy = correct / len(binary_preds)
            
            # Calculate average inference time
            avg_inference_time = sum(inference_times) / len(inference_times)
            
            results[model_type] = {
                'accuracy': float(accuracy),
                'avg_inference_time': float(avg_inference_time),
                'inference_times': [float(t) for t in inference_times]
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Average inference time: {avg_inference_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"Error evaluating TFLite model for {model_type}: {e}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "disruption", "tflite_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def plot_disruption_results(results):
    """
    Plot accuracy under different disruption conditions
    
    Args:
        results: Results from evaluate_under_disruption
    """
    # Plot packet loss results
    plt.figure(figsize=(12, 8))
    
    for model_name, model_results in results.items():
        loss_rates = [float(rate) for rate in model_results['packet_loss'].keys()]
        accuracies = [model_results['packet_loss'][rate]['accuracy'] for rate in model_results['packet_loss']]
        
        plt.plot(loss_rates, accuracies, marker='o', label=model_name)
    
    plt.title('Model Accuracy Under Packet Loss')
    plt.xlabel('Packet Loss Rate')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "packet_loss_comparison.png"))
    plt.close()
    
    # Plot node failure results
    plt.figure(figsize=(12, 8))
    
    for model_name, model_results in results.items():
        failure_rates = [float(rate) for rate in model_results['node_failure'].keys()]
        accuracies = [model_results['node_failure'][rate]['accuracy'] for rate in model_results['node_failure']]
        
        plt.plot(failure_rates, accuracies, marker='o', label=model_name)
    
    plt.title('Model Accuracy Under Node Failure')
    plt.xlabel('Node Failure Rate')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "node_failure_comparison.png"))
    plt.close()
    
    # Plot combined disruption results as heatmap for each model
    for model_name, model_results in results.items():
        combined = model_results['combined']
        
        # Extract unique values
        packet_loss_rates = sorted(set([combined[key]['packet_loss_rate'] for key in combined]))
        node_failure_rates = sorted(set([combined[key]['node_failure_rate'] for key in combined]))
        
        # Create data grid
        data = np.zeros((len(node_failure_rates), len(packet_loss_rates)))
        
        for i, nf in enumerate(node_failure_rates):
            for j, pl in enumerate(packet_loss_rates):
                key = f"PL{pl:.1f}_NF{nf:.1f}"
                if key in combined:
                    data[i, j] = combined[key]['accuracy']
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(data, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='Accuracy')
        
        # Add labels
        plt.xticks(range(len(packet_loss_rates)), [f"{pl*100:.0f}%" for pl in packet_loss_rates])
        plt.yticks(range(len(node_failure_rates)), [f"{nf*100:.0f}%" for nf in node_failure_rates])
        
        plt.xlabel('Packet Loss Rate')
        plt.ylabel('Node Failure Rate')
        plt.title(f'{model_name} Model: Accuracy Under Combined Disruption')
        
        # Add text annotations
        for i in range(len(node_failure_rates)):
            for j in range(len(packet_loss_rates)):
                plt.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="w")
        
        plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{model_name}_combined_disruption.png"))
        plt.close()


def plot_inference_time_results(inference_results):
    """
    Plot inference time results
    
    Args:
        inference_results: Results from evaluate_inference_time
    """
    plt.figure(figsize=(12, 8))
    
    # Extract batch sizes
    batch_sizes = list(inference_results[list(inference_results.keys())[0]].keys())
    batch_sizes = [int(bs) for bs in batch_sizes]
    
    for model_name, results in inference_results.items():
        times = [results[bs]['time_per_sample'] * 1000 for bs in results]  # Convert to ms
        plt.plot(batch_sizes, times, marker='o', label=model_name)
    
    plt.title('Inference Time Per Sample')
    plt.xlabel('Batch Size')
    plt.ylabel('Time per Sample (ms)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "inference_time_comparison.png"))
    plt.close()


def main():
    """Main evaluation pipeline"""
    # Set up directories
    setup_directories()
    
    print("=== Nerve Agent Detection Model Evaluation ===")
    
    # Load data
    X_train, X_val, y_train, y_val, scaler = prepare_sensor_data(
        DATA_PATH, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Load models
    models = load_models()
    if not models:
        print("No models found. Please train models first.")
        return
    
    # Evaluate models under disruption
    print("\nEvaluating models under network disruption...")
    disruption_results = evaluate_under_disruption(models, X_val, y_val)
    
    # Evaluate inference time
    print("\nEvaluating inference time...")
    inference_results = evaluate_inference_time(models, X_val)
    
    # Evaluate TFLite models
    print("\nEvaluating TFLite models...")
    tflite_results = evaluate_tflite_models(X_val, y_val)
    
    # Plot results
    print("\nGenerating plots...")
    plot_disruption_results(disruption_results)
    plot_inference_time_results(inference_results)
    
    print("\nEvaluation complete! Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()