"""
Model Utility Functions for Crop Leaf Disease Classification
Handles model building, training, and evaluation
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (confusion_matrix, classification_report, 
                            precision_score, recall_score, f1_score, accuracy_score)


def _build_optimizer(optimizer, learning_rate: float):
    """Create an optimizer from a name or return an optimizer instance unchanged."""
    if isinstance(optimizer, keras.optimizers.Optimizer):
        return optimizer

    optimizer_name = str(optimizer).lower()

    if optimizer_name == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    if optimizer_name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)

    return keras.optimizers.Adam(learning_rate=learning_rate)


def build_cnn_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer='adam',
    activation: str = 'leaky_relu',
    filters: Tuple[int, int, int, int] = (32, 64, 128, 256),
    dense_units: int = 128,
    kernel_size: Tuple[int, int] = (3, 3),
    dropout_rates: Tuple[float, float, float, float, float] = (0.20, 0.25, 0.30, 0.35, 0.50)
) -> keras.Model:
    """
    Build a compact custom CNN model for image classification.

    The dataset is small and imbalanced, so this baseline intentionally uses
    fewer filters, global pooling, and smaller dense layers to reduce
    overfitting risk compared with a large Flatten-heavy CNN.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        learning_rate: Adam optimizer learning rate
        weight_decay: L2 regularization strength for trainable layers
        optimizer: Optimizer name or instance
        activation: Activation function for convolution layers
        filters: Number of filters for each convolution block
        dense_units: Number of units in the dense head
        kernel_size: Convolution kernel size
        dropout_rates: Dropout rates for each block and dense head
        
    Returns:
        Compiled Keras model
    """
    regularizer = keras.regularizers.l2(weight_decay)

    use_leaky_relu = activation.lower() == 'leaky_relu'
    optimizer_instance = _build_optimizer(optimizer, learning_rate)

    def conv_block(num_filters: int, dropout_rate: float):
        block = [
            layers.Conv2D(num_filters, kernel_size, activation=None if use_leaky_relu else activation,
                          padding='same', kernel_regularizer=regularizer)
        ]
        if use_leaky_relu:
            block.append(layers.LeakyReLU(alpha=0.1))
        block.extend([
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate)
        ])
        return block

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Block 1
        *conv_block(filters[0], dropout_rates[0]),
        
        # Block 2
        *conv_block(filters[1], dropout_rates[1]),
        
        # Block 3
        *conv_block(filters[2], dropout_rates[2]),
        
        # Block 4
        *conv_block(filters[3], dropout_rates[3]),
        
        # Pooling and Dense
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rates[4]),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizer_instance,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_mobilenet_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    trainable_layers: int = 50,
    learning_rate: float = 1e-4,
    optimizer='adam',
    dense_units: int = 256,
    dropout_rate: float = 0.35
) -> keras.Model:
    """
    Build MobileNetV2 transfer learning model.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        trainable_layers: Number of layers to unfreeze for fine-tuning
        learning_rate: Learning rate used while the base model is frozen
        optimizer: Optimizer name or instance
        dense_units: Units in the classification head
        dropout_rate: Dropout rate for the classification head
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False

    optimizer_instance = _build_optimizer(optimizer, learning_rate)
    
    # Add custom top layers
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(2.0, offset=-1.0, name='mobilenetv2_preprocessing'),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with frozen base
    model.compile(
        optimizer=optimizer_instance,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_resnet50_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    optimizer='adam',
    dense_units: int = 256,
    dropout_rate: float = 0.35
) -> keras.Model:
    """
    Build ResNet50 transfer learning model with improved regularization.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        learning_rate: Learning rate for the initial training
        optimizer: Optimizer name or instance
        dense_units: Units in the classification head
        dropout_rate: Dropout rate for the classification head
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50
    base_model = keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially (Phase 1: feature extraction)
    base_model.trainable = False

    optimizer_instance = _build_optimizer(optimizer, learning_rate)
    
    # Add custom classification head with improved regularization
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Convert [0, 1] range to [-1, 1] range expected by ResNet50
        # Formula: x * 2.0 - 1.0
        layers.Rescaling(2.0, offset=-1.0, name='resnet50_preprocessing'),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(
            dense_units, 
            activation='relu', 
            kernel_regularizer=keras.regularizers.l2(1e-4)  # L2 regularization
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with frozen base
    model.compile(
        optimizer=optimizer_instance,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_augmentation() -> ImageDataGenerator:
    """
    Create image data augmentation generator.
    
    Returns:
        ImageDataGenerator object
    """
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )


def create_validation_generator() -> ImageDataGenerator:
    """
    Create validation data generator (normalization only).
    
    Returns:
        ImageDataGenerator object
    """
    return ImageDataGenerator()


def compute_class_weights_from_labels(
    y_train: np.ndarray,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Compute balanced class weights from training labels.

    This uses label frequencies from the training split and is preferred over
    SMOTE for raw image data because it does not synthesize artificial pixels.

    Args:
        y_train: Training labels as integer labels or one-hot encoded labels
        class_names: Optional class-name list aligned to class indices
        verbose: Print a readable summary when True

    Returns:
        Dictionary mapping class index to class weight
    """
    if y_train.ndim > 1:
        y_indices = np.argmax(y_train, axis=1)
    else:
        y_indices = y_train.astype(int).ravel()

    classes = np.unique(y_indices)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_indices)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    if verbose:
        print("\nClass weights computed from y_train")
        print("(Using class weights instead of SMOTE keeps the original image distribution intact.)")
        print("-" * 70)
        for cls in classes:
            cls_index = int(cls)
            cls_name = class_names[cls_index] if class_names and cls_index < len(class_names) else f"Class {cls_index}"
            sample_count = int(np.sum(y_indices == cls_index))
            print(f"  {cls_index:2d} | {cls_name:35s} | samples: {sample_count:5d} | weight: {class_weight[cls_index]:.4f}")
        print("-" * 70)

    return class_weight


def _sample_weights_from_labels(
    y_train: np.ndarray,
    class_weight: Optional[Dict[int, float]]
) -> Optional[np.ndarray]:
    """
    Convert class weights into sample weights for generator-based training.
    """
    if class_weight is None:
        return None

    if y_train.ndim > 1:
        y_indices = np.argmax(y_train, axis=1)
    else:
        y_indices = y_train.astype(int).ravel()

    return np.array([class_weight.get(int(label), 1.0) for label in y_indices], dtype=np.float32)


def unfreeze_model_layers(model: keras.Model, num_layers: int, learning_rate: float = 1e-5, optimizer='adam') -> keras.Model:
    """
    Unfreeze the last num_layers for fine-tuning.
    
    Args:
        model: Keras model
        num_layers: Number of layers to unfreeze from the end
        learning_rate: Learning rate used for fine-tuning
        optimizer: Optimizer name or instance
        
    Returns:
        Model with unfrozen layers
    """
    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=_build_optimizer(optimizer, learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model: keras.Model, train_data, val_data, epochs: int = 50,
                batch_size: int = 32, callbacks: List = None,
                class_weight: Optional[Dict[int, float]] = None,
                augment: bool = True) -> Dict:
    """
    Train the model.
    
    Args:
        model: Keras model
        train_data: Training data generator or arrays (X, y)
        val_data: Validation data (X, y)
        epochs: Number of training epochs
        batch_size: Batch size
        callbacks: List of Keras callbacks
        class_weight: Optional dictionary mapping integer class index to weight
        augment: Apply on-the-fly augmentation when train_data is (X_train, y_train)
        
    Returns:
        Training history dictionary
    """
    if callbacks is None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    if augment and isinstance(train_data, tuple) and len(train_data) == 2:
        X_train, y_train = train_data
        sample_weight = _sample_weights_from_labels(y_train, class_weight)
        generator_kwargs = {
            'batch_size': batch_size,
            'shuffle': True
        }
        if sample_weight is not None:
            generator_kwargs['sample_weight'] = sample_weight

        train_generator = create_data_augmentation().flow(
            X_train,
            y_train,
            **generator_kwargs
        )
        steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

        history = model.fit(
            train_generator,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
    
    return history.history


def evaluate_model(model: keras.Model, test_data_x: np.ndarray, test_data_y: np.ndarray,
                   class_names: List[str]) -> Dict:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained Keras model
        test_data_x: Test images
        test_data_y: Test labels (one-hot encoded)
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    predictions = model.predict(test_data_x, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_data_y, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Classification report
    # Guard against mismatch when some classes are not present in the test set.
    unique_labels = np.unique(np.concatenate([true_labels, pred_labels]))
    # If class_names covers more classes than present in true/pred, select names for present labels
    try:
        target_names = [class_names[int(i)] for i in unique_labels]
    except Exception:
        target_names = None

    report = classification_report(
        true_labels,
        pred_labels,
        labels=unique_labels.tolist(),
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'predicted_labels': pred_labels.tolist()
    }


def save_model(model: keras.Model, model_name: str, output_dir: str = 'models') -> str:
    """
    Save trained model.
    
    Args:
        model: Keras model
        model_name: Name for the model
        output_dir: Output directory
        
    Returns:
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_name}.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


def load_model(model_path: str) -> keras.Model:
    """
    Load a saved model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    return model


def convert_to_tflite(model: keras.Model, output_path: str = 'models/tflite/model.tflite') -> str:
    """
    Convert Keras model to TensorFlow Lite.
    
    Args:
        model: Keras model to convert
        output_path: Path for output TFLite model
        
    Returns:
        Path to converted model
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    return output_path


def get_model_summary(model: keras.Model) -> str:
    """
    Get model architecture summary.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary as string
    """
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    return f.getvalue()


def save_metrics(metrics: Dict, filename: str, output_dir: str = 'results/reports') -> str:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        filename: Output filename
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {filepath}")
    return filepath
