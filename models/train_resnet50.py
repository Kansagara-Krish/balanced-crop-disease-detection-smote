"""
ResNet50 Transfer Learning Training Script
Trains ResNet50 model with two-phase approach (frozen base -> fine-tuning)
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent / 'utils'
sys.path.insert(0, str(utils_path))

from transfer_learning_utils import (
    build_resnet50_model,
    preprocess_images,
    create_data_augmentation_generator,
    create_validation_generator,
    compute_class_weights_from_labels,
    train_two_phase_transfer_learning,
    evaluate_transfer_learning_model,
    save_transfer_learning_model,
    get_model_info
)


def main():
    """
    Main training function for ResNet50 model.
    """
    # ============================================================================
    # 1. CONFIGURATION
    # ============================================================================
    print("\n" + "="*70)
    print("ResNet50 Transfer Learning Training")
    print("="*70)
    
    MODEL_NAME = 'resnet50'
    INPUT_SHAPE = (224, 224, 3)
    LEARNING_RATE = 0.0001
    
    # Phase 1: Frozen base
    EPOCHS_PHASE1 = 20
    
    # Phase 2: Fine-tuning
    EPOCHS_PHASE2 = 15
    
    BATCH_SIZE = 32
    NUM_UNFREEZE_LAYERS = 50
    
    # Paths
    DATA_DIR = Path(__file__).parent.parent / 'results' / 'preprocessed_data'
    MODEL_OUTPUT_DIR = Path(__file__).parent / MODEL_NAME
    REPORT_OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'reports'
    
    # Create output directories
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: ResNet50")
    print(f"  Input Shape: {INPUT_SHAPE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Phase 1 Epochs (frozen): {EPOCHS_PHASE1}")
    print(f"  Phase 2 Epochs (fine-tuning): {EPOCHS_PHASE2}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Unfreeze Layers (Phase 2): {NUM_UNFREEZE_LAYERS}")
    print(f"  Output Directory: {MODEL_OUTPUT_DIR}")
    
    # ============================================================================
    # 2. LOAD PREPROCESSED DATA
    # ============================================================================
    print("\n" + "-"*70)
    print("Loading preprocessed data...")
    print("-"*70)
    
    try:
        X_train = np.load(DATA_DIR / 'X_train.npy')
        y_train = np.load(DATA_DIR / 'y_train.npy')
        X_val = np.load(DATA_DIR / 'X_val.npy')
        y_val = np.load(DATA_DIR / 'y_val.npy')
        X_test = np.load(DATA_DIR / 'X_test.npy')
        y_test = np.load(DATA_DIR / 'y_test.npy')
        
        with open(DATA_DIR / 'label_encoding.json', 'r') as f:
            label_encoding = json.load(f)
        
        num_classes = len(label_encoding)
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Label encoding: {label_encoding}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Preprocessed data not found!")
        print(f"  Please run 03_data_preprocessing.ipynb first")
        print(f"  Error details: {e}")
        sys.exit(1)
    
    # ============================================================================
    # 3. APPLY MODEL-SPECIFIC PREPROCESSING
    # ============================================================================
    print("\n" + "-"*70)
    print("Applying ResNet50-specific preprocessing...")
    print("-"*70)
    
    X_train_processed = preprocess_images(X_train, model_type='resnet50')
    X_val_processed = preprocess_images(X_val, model_type='resnet50')
    X_test_processed = preprocess_images(X_test, model_type='resnet50')
    
    print("✓ Preprocessing completed!")
    print(f"  Value range after preprocessing: [{X_train_processed.min():.3f}, {X_train_processed.max():.3f}]")

    class_names = sorted(label_encoding.keys(), key=lambda x: label_encoding[x])
    class_weight = compute_class_weights_from_labels(y_train, class_names=class_names, verbose=True)
    
    # Create generators with shared augmentation pipeline
    train_generator = create_data_augmentation_generator().flow(
        X_train_processed,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_generator = create_validation_generator().flow(
        X_val_processed,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # ============================================================================
    # 4. BUILD MODEL
    # ============================================================================
    print("\n" + "-"*70)
    print("Building ResNet50 model...")
    print("-"*70)
    
    model = build_resnet50_model(
        num_classes=num_classes,
        input_shape=INPUT_SHAPE,
        learning_rate=LEARNING_RATE
    )
    
    model_info = get_model_info(model)
    print(f"\n✓ Model built successfully!")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    print(f"  Non-trainable parameters: {model_info['non_trainable_params']:,}")
    
    # ============================================================================
    # 5. TRAIN MODEL (TWO-PHASE)
    # ============================================================================
    history = train_two_phase_transfer_learning(
        model=model,
        train_data=train_generator,
        val_data=val_generator,
        epochs_phase1=EPOCHS_PHASE1,
        epochs_phase2=EPOCHS_PHASE2,
        batch_size=BATCH_SIZE,
        num_unfreeze_layers=NUM_UNFREEZE_LAYERS,
        model_name=MODEL_NAME,
        output_dir=str(MODEL_OUTPUT_DIR),
        class_weight=class_weight,
        verbose=1
    )
    
    # ============================================================================
    # 6. EVALUATE ON TEST SET
    # ============================================================================
    print("\n" + "-"*70)
    print("Evaluating on test set...")
    print("-"*70)
    
    eval_metrics = evaluate_transfer_learning_model(
        model=model,
        X_test=X_test_processed,
        y_test=y_test,
        class_names=class_names
    )
    
    print(f"\n✓ Evaluation completed!")
    print(f"  Accuracy:  {eval_metrics['accuracy']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall:    {eval_metrics['recall']:.4f}")
    print(f"  F1-Score:  {eval_metrics['f1_score']:.4f}")
    
    # ============================================================================
    # 7. SAVE MODEL AND METRICS
    # ============================================================================
    print("\n" + "-"*70)
    print("Saving model and metrics...")
    print("-"*70)
    
    # Save trained model
    model_path = save_transfer_learning_model(
        model=model,
        model_name=f'{MODEL_NAME}_model',
        output_dir=str(MODEL_OUTPUT_DIR)
    )
    
    # Save metrics
    metrics_file = REPORT_OUTPUT_DIR / f'{MODEL_NAME}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save training configuration
    config_file = REPORT_OUTPUT_DIR / f'{MODEL_NAME}_config.json'
    config = {
        'model': MODEL_NAME,
        'input_shape': INPUT_SHAPE,
        'learning_rate': LEARNING_RATE,
        'epochs_phase1': EPOCHS_PHASE1,
        'epochs_phase2': EPOCHS_PHASE2,
        'batch_size': BATCH_SIZE,
        'num_unfreeze_layers': NUM_UNFREEZE_LAYERS,
        'num_classes': num_classes,
        'class_names': class_names
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Config saved to: {config_file}")
    
    # ============================================================================
    # 8. SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"Model Name: {MODEL_NAME}")
    print(f"Model Path: {model_path}")
    print(f"Metrics Path: {metrics_file}")
    print(f"Config Path: {config_file}")
    print(f"\nFinal Metrics:")
    print(f"  Test Accuracy:  {eval_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {eval_metrics['precision']:.4f}")
    print(f"  Test Recall:    {eval_metrics['recall']:.4f}")
    print(f"  Test F1-Score:  {eval_metrics['f1_score']:.4f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
