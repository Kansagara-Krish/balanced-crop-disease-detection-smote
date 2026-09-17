"""
Transfer Learning Utility Functions for Crop Leaf Disease Classification
Handles building and training models with pretrained architectures (MobileNetV2, EfficientNetB0, ResNet50)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
from sklearn.metrics import (confusion_matrix, classification_report, 
                            precision_score, recall_score, f1_score, accuracy_score)
from model_utils import compute_class_weights_from_labels


# ============================================================================
# MODEL BUILDERS - Transfer Learning Models
# ============================================================================

def build_mobilenetv2_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 0.0001
) -> keras.Model:
    """
    Build MobileNetV2 transfer learning model.
    
    Architecture:
    - Base model: MobileNetV2 (ImageNet pretrained)
    - Classification head: GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.3) -> Dense(num_classes, softmax)
    - Initial state: Base model frozen
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 with ImageNet weights
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model with classification head
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', name='dense_head_1'),
        layers.Dropout(0.3, name='dropout_head_1'),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='MobileNetV2_Transfer_Learning')
    
    # Compile with specified learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_efficientnetb0_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 0.0001,
    dense_units: int = 256,
    dropout_rate: float = 0.4,
    label_smoothing: float = 0.05
) -> keras.Model:
    """
    Build EfficientNetB0 transfer learning model.
    
    Architecture:
    - Base model: EfficientNetB0 (ImageNet pretrained)
        - Classification head: GlobalAveragePooling2D -> BatchNormalization -> Dense(dense_units, relu)
            -> Dropout(dropout_rate) -> Dense(dense_units // 2, relu) -> Dropout(dropout_rate / 2)
            -> Dense(num_classes, softmax)
    - Initial state: Base model frozen
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNetB0 with ImageNet weights
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model with classification head
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(name='head_batch_norm'),
        layers.Dense(dense_units, activation='relu', name='dense_head_1'),
        layers.Dropout(dropout_rate, name='dropout_head_1'),
        layers.Dense(dense_units // 2, activation='relu', name='dense_head_2'),
        layers.Dropout(dropout_rate / 2, name='dropout_head_2'),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='EfficientNetB0_Transfer_Learning')
    
    # Compile with specified learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy']
    )
    
    return model


def build_resnet50_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 0.0001
) -> keras.Model:
    """
    Build ResNet50 transfer learning model.
    
    Architecture:
    - Base model: ResNet50 (ImageNet pretrained)
    - Classification head: GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.3) -> Dense(num_classes, softmax)
    - Initial state: Base model frozen
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50 with ImageNet weights
    base_model = keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model with classification head
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', name='dense_head_1'),
        layers.Dropout(0.3, name='dropout_head_1'),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='ResNet50_Transfer_Learning')
    
    # Compile with specified learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# PREPROCESSING FUNCTIONS - Model-Specific Input Preprocessing
# ============================================================================

def get_mobilenetv2_preprocessor():
    """
    Get MobileNetV2 preprocessing function.
    
    Returns:
        Preprocessing function from keras.applications.mobilenet_v2
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input


def get_efficientnetb0_preprocessor():
    """
    Get EfficientNetB0 preprocessing function.
    
    Returns:
        Preprocessing function from keras.applications.efficientnet
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input


def get_resnet50_preprocessor():
    """
    Get ResNet50 preprocessing function.
    
    Returns:
        Preprocessing function from keras.applications.resnet50
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input


def preprocess_images(images: np.ndarray, model_type: str) -> np.ndarray:
    """
    Apply model-specific preprocessing to images.
    
    Args:
        images: Input images array (numpy array)
        model_type: Type of model ('mobilenetv2', 'efficientnetb0', 'resnet50')
        
    Returns:
        Preprocessed images array
    """
    if model_type.lower() == 'mobilenetv2':
        preprocessor = get_mobilenetv2_preprocessor()
    elif model_type.lower() == 'efficientnetb0':
        preprocessor = get_efficientnetb0_preprocessor()
    elif model_type.lower() == 'resnet50':
        preprocessor = get_resnet50_preprocessor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return preprocessor(images)


def create_data_augmentation_generator() -> ImageDataGenerator:
    """
    Create a shared image augmentation generator for training.
    
    Returns:
        ImageDataGenerator object with consistent augmentation parameters
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def create_validation_generator() -> ImageDataGenerator:
    """
    Create a validation data generator with no augmentation.
    
    Returns:
        ImageDataGenerator object for validation/test data
    """
    return ImageDataGenerator()


# ============================================================================
# FINE-TUNING FUNCTIONS
# ============================================================================

def unfreeze_base_model(
    model: keras.Model,
    num_layers: int = 50,
    learning_rate: float = 0.0001
) -> keras.Model:
    """
    Unfreeze the last num_layers of the base model for fine-tuning.
    
    Args:
        model: Keras model with frozen base
        num_layers: Number of layers to unfreeze from the end
        learning_rate: Learning rate for fine-tuning phase
        
    Returns:
        Model with unfrozen layers and recompiled with lower learning rate
    """
    # Locate the nested pretrained backbone inside the wrapper model.
    base_model = next(
        (layer for layer in model.layers if isinstance(layer, keras.Model)),
        None
    )
    if base_model is None:
        raise ValueError("Could not locate a nested base model to unfreeze.")
    
    # Unfreeze all layers
    base_model.trainable = True
    
    # Unfreeze only the last num_layers
    freeze_until = max(0, len(base_model.layers) - num_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Keep batch-normalization layers frozen during fine-tuning for stability
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    
    return model


def get_model_info(model: keras.Model) -> Dict:
    """
    Get information about a transfer learning model.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    info = {
        'name': model.name,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
    }
    return info


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_training_callbacks(
    model_name: str,
    output_dir: str = 'models',
    patience_early_stop: int = 10,
    patience_reduce_lr: int = 5
) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks for transfer learning models.
    
    Callbacks included:
    - EarlyStopping: Stop training if validation loss doesn't improve
    - ReduceLROnPlateau: Reduce learning rate if validation loss plateaus
    - ModelCheckpoint: Save best model
    
    Args:
        model_name: Name of the model (for checkpoint naming)
        output_dir: Directory to save models
        patience_early_stop: Patience for early stopping
        patience_reduce_lr: Patience for reduce learning rate
        
    Returns:
        List of Keras callbacks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-8,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f'best_{model_name}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0,
            mode='max'
        )
    ]
    
    return callbacks


def train_transfer_learning_model(
    model: keras.Model,
    X_train,
    y_train=None,
    X_val=None,
    y_val=None,
    epochs: int = 30,
    batch_size: int = 32,
    callbacks: List[keras.callbacks.Callback] = None,
    class_weight: Optional[Dict[int, float]] = None,
    verbose: int = 1
) -> Dict:
    """
    Train a transfer learning model.
    
    Args:
        model: Compiled Keras model
        X_train: Training images array or generator
        y_train: Training labels (one-hot encoded) if X_train is array; ignored for generators
        X_val: Validation images array or generator
        y_val: Validation labels (one-hot encoded) if X_val is array; ignored for generators
        epochs: Number of training epochs
        batch_size: Batch size for training
        callbacks: List of callbacks (if None, default callbacks are used)
        verbose: Verbosity level (0, 1, or 2)
        
    Returns:
        Dictionary with training history
    """
    if callbacks is None:
        callbacks = []

    if y_train is None:
        history = model.fit(
            X_train,
            validation_data=X_val,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose
        )

    return history.history


def train_two_phase_transfer_learning(
    model: keras.Model,
    train_data,
    val_data,
    epochs_phase1: int = 20,
    epochs_phase2: int = 10,
    batch_size: int = 32,
    num_unfreeze_layers: int = 50,
    model_name: str = 'transfer_model',
    output_dir: str = 'models',
    class_weight: Optional[Dict[int, float]] = None,
    verbose: int = 1
) -> Dict:
    """
    Train a transfer learning model in two phases:
    Phase 1: Train with frozen base model
    Phase 2: Unfreeze and fine-tune

    Args:
        model: Compiled Keras model with frozen base
        train_data: Training images or generator
        val_data: Validation images or generator
        epochs_phase1: Number of epochs for phase 1 (frozen base)
        epochs_phase2: Number of epochs for phase 2 (fine-tuning)
        batch_size: Batch size for training
        num_unfreeze_layers: Number of base model layers to unfreeze in phase 2
        model_name: Name of the model for checkpoint saving
        output_dir: Directory to save models
        verbose: Verbosity level

    Returns:
        Dictionary with combined training history from both phases
    """
    # ======== PHASE 1: Train with Frozen Base ========
    print("\n" + "="*70)
    print(f"PHASE 1: Training {model_name} with Frozen Base Layers")
    print("="*70)
    
    callbacks_phase1 = create_training_callbacks(
        model_name=f'{model_name}_phase1',
        output_dir=output_dir,
        patience_early_stop=8,
        patience_reduce_lr=4
    )
    
    history_phase1 = train_transfer_learning_model(
        model=model,
        X_train=train_data,
        X_val=val_data,
        epochs=epochs_phase1,
        batch_size=batch_size,
        callbacks=callbacks_phase1,
        class_weight=class_weight,
        verbose=verbose
    )
    
    print(f"\n✓ Phase 1 completed!")
    
    # ======== PHASE 2: Fine-tune with Unfrozen Layers ========
    print("\n" + "="*70)
    print(f"PHASE 2: Fine-tuning {model_name} with Unfrozen Layers")
    print("="*70)
    
    # Unfreeze base model layers
    model = unfreeze_base_model(
        model=model,
        num_layers=num_unfreeze_layers,
        learning_rate=0.0001
    )
    
    print(f"✓ Unfroze last {num_unfreeze_layers} layers")
    print(f"✓ Learning rate set to 0.0001\n")
    
    callbacks_phase2 = create_training_callbacks(
        model_name=f'{model_name}_phase2',
        output_dir=output_dir,
        patience_early_stop=10,
        patience_reduce_lr=5
    )
    
    history_phase2 = train_transfer_learning_model(
        model=model,
        X_train=train_data,
        X_val=val_data,
        epochs=epochs_phase2,
        batch_size=batch_size,
        callbacks=callbacks_phase2,
        class_weight=class_weight,
        verbose=verbose
    )
    
    print(f"\n✓ Phase 2 completed!")
    
    # Combine histories
    combined_history = {
        'phase1': history_phase1,
        'phase2': history_phase2,
        'total_epochs': epochs_phase1 + epochs_phase2
    }
    
    return combined_history


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_transfer_learning_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Evaluate a trained transfer learning model.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
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


# ============================================================================
# MODEL SAVING AND LOADING
# ============================================================================

def save_transfer_learning_model(
    model: keras.Model,
    model_name: str,
    output_dir: str = 'models'
) -> str:
    """
    Save a transfer learning model.
    
    Args:
        model: Keras model to save
        model_name: Name for the model file (without extension)
        output_dir: Output directory
        
    Returns:
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_name}.h5')
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    return model_path


def load_transfer_learning_model(model_path: str) -> keras.Model:
    """
    Load a saved transfer learning model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    return model
