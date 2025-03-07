"""
models.py

This module defines the model architectures for nerve agent detection.
It includes multiple model options of varying complexity, optimized for resource-constrained environments.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, LSTM, Bidirectional, BatchNormalization


def create_1d_cnn_model(input_shape, num_classes=1):
    """
    Create a simple 1D CNN model for sequence classification
    Optimized for very resource-constrained environments
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        num_classes (int): Number of output classes (1 for binary classification)
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = Sequential([
        # Convolutional layer for feature extraction
        Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        
        # Second convolutional layer
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(input_shape, num_classes=1):
    """
    Create an LSTM model for sequence classification
    Balanced for performance and resource usage
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        num_classes (int): Number of output classes (1 for binary classification)
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = Sequential([
        # LSTM layer
        LSTM(32, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_bidirectional_lstm_model(input_shape, num_classes=1):
    """
    Create a Bidirectional LSTM model for sequence classification
    Higher accuracy but more computationally intensive
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        num_classes (int): Number of output classes (1 for binary classification)
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = Sequential([
        # Bidirectional LSTM layer
        Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        
        # Second Bidirectional LSTM layer
        Bidirectional(LSTM(16, return_sequences=False)),
        Dropout(0.2),
        
        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_hybrid_model(input_shape, num_classes=1):
    """
    Create a hybrid CNN-LSTM model for sequence classification
    Good balance of feature extraction and temporal pattern recognition
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        num_classes (int): Number of output classes (1 for binary classification)
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = Sequential([
        # CNN for feature extraction
        Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        
        # LSTM for temporal patterns
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_tiny_model(input_shape, num_classes=1):
    """
    Create an extremely lightweight model for very constrained devices
    Sacrifices some accuracy for minimal resource usage
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        num_classes (int): Number of output classes (1 for binary classification)
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = Sequential([
        # Simple feature extraction
        Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        
        # Flatten and minimal dense layer
        Flatten(),
        Dense(8, activation='relu'),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    # Compile model with simpler optimizer
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def convert_to_tflite(model, quantize=True, output_path=None):
    """
    Convert a Keras model to TensorFlow Lite format
    
    Args:
        model (tf.keras.Model): The model to convert
        quantize (bool): Whether to apply quantization to reduce model size
        output_path (str, optional): Path to save the TFLite model
        
    Returns:
        bytes: The TFLite model as a byte array
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    tflite_model = converter.convert()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {output_path}")
    
    return tflite_model


def get_model_by_name(model_name, input_shape, num_classes=1):
    """
    Get a model by name
    
    Args:
        model_name (str): Name of the model ('cnn', 'lstm', 'bilstm', 'hybrid', 'tiny')
        input_shape (tuple): Shape of input data
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: The requested model
    """
    models = {
        'cnn': create_1d_cnn_model,
        'lstm': create_lstm_model,
        'bilstm': create_bidirectional_lstm_model,
        'hybrid': create_hybrid_model,
        'tiny': create_tiny_model
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](input_shape, num_classes)


if __name__ == "__main__":
    # Example usage - create and summarize each model
    input_shape = (10, 4)  # 10 time steps, 4 features
    
    for model_name in ['cnn', 'lstm', 'bilstm', 'hybrid', 'tiny']:
        print(f"\n{model_name.upper()} Model:")
        model = get_model_by_name(model_name, input_shape)
        model.summary()
        
        # Convert to TFLite to check size
        tflite_model = convert_to_tflite(model)
        print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")