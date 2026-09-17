r"""
Memory-Efficient Data Loader for Training with Integrated SMOTE Support

Loads preprocessed images from disk in batches instead of keeping
entire dataset in RAM. This avoids MemoryError when dataset is large.
Supports seamless SMOTE oversampling for class balancing across all models.

Usage:
    from efficient_data_loader import create_train_val_test_datasets
    
    # Standard loading:
    train_ds, val_ds, test_ds, metadata = create_train_val_test_datasets(
        base_path=r"D:\cnn_model\results\preprocessed_data_efficient",
        batch_size=16,
        use_smote=False
    )
    
    # SMOTE-balanced loading:
    train_ds, val_ds, test_ds, metadata = create_train_val_test_datasets(
        base_path=r"D:\cnn_model\results\preprocessed_data_efficient",
        batch_size=16,
        use_smote=True
    )
"""

import numpy as np
import tensorflow as tf
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional


def load_image_from_npz(file_path: str) -> np.ndarray:
    """Load preprocessed image from compressed NPZ file."""
    try:
        data = np.load(file_path)
        return data['image'].astype('float32')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros((224, 224, 3), dtype='float32')


def create_train_val_test_datasets(
    base_path: str = r"D:\cnn_model\results\preprocessed_data_efficient",
    batch_size: int = 16,
    use_smote: bool = False,
    smote_base_path: Optional[str] = None,
    k_neighbors: int = 3,
    random_state: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create tf.data.Dataset objects from saved preprocessed images with optional SMOTE balancing.
    
    Args:
        base_path: Path to preprocessed_data_efficient directory
        batch_size: Batch size for datasets
        use_smote: If True, uses/generates SMOTE balanced training dataset
        smote_base_path: Custom directory for SMOTE data (defaults to results/preprocessed_data_smote)
        k_neighbors: Number of nearest neighbors for SMOTE interpolation
        random_state: Random seed for reproducibility
        
    Returns:
        train_ds, val_ds, test_ds, metadata_dict
    """
    if smote_base_path is None:
        parent_dir = os.path.dirname(os.path.abspath(base_path))
        smote_base_path = os.path.join(parent_dir, "preprocessed_data_smote")

    # If SMOTE is requested, ensure SMOTE dataset exists or generate it
    if use_smote:
        train_img_dir = os.path.join(smote_base_path, 'train_images')
        meta_file = os.path.join(smote_base_path, 'split_indices.json')
        if not (os.path.exists(train_img_dir) and os.path.exists(meta_file)):
            print(f"[SMOTE] Generating memory-efficient SMOTE dataset into: {smote_base_path}")
            from smote_utils import generate_smote_dataset_to_disk
            generate_smote_dataset_to_disk(
                base_path=base_path,
                output_dir=smote_base_path,
                k_neighbors=k_neighbors,
                random_state=random_state,
                verbose=True
            )
        
        # Load from SMOTE directory
        active_train_dir = train_img_dir
        with open(os.path.join(smote_base_path, 'split_indices.json'), 'r') as f:
            split_info = json.load(f)
        with open(os.path.join(smote_base_path, 'image_metadata.json'), 'r') as f:
            metadata = json.load(f)
        with open(os.path.join(smote_base_path, 'label_encoding.json'), 'r') as f:
            label_encoding = json.load(f)
        with open(os.path.join(smote_base_path, 'preprocessing_info.json'), 'r') as f:
            preprocess_info = json.load(f)

        num_classes = preprocess_info['num_classes']
        train_labels = np.array(metadata['train_labels'])
        val_labels = np.array(metadata['val_labels'])
        test_labels = np.array(metadata['test_labels'])
        
        val_source_dir = split_info.get('val_source_dir', os.path.join(base_path, 'val_images'))
        test_source_dir = split_info.get('test_source_dir', os.path.join(base_path, 'test_images'))
        train_count = split_info['train_count']
        val_count = split_info['val_count']
        test_count = split_info['test_count']

    else:
        # Standard non-SMOTE loading
        with open(os.path.join(base_path, 'split_indices.json'), 'r') as f:
            split_info = json.load(f)
        with open(os.path.join(base_path, 'image_metadata.json'), 'r') as f:
            metadata = json.load(f)
        with open(os.path.join(base_path, 'label_encoding.json'), 'r') as f:
            label_encoding = json.load(f)
        with open(os.path.join(base_path, 'preprocessing_info.json'), 'r') as f:
            preprocess_info = json.load(f)

        labels_all = np.array(metadata['labels'])
        num_classes = preprocess_info['num_classes']
        train_labels = labels_all[split_info['train_indices']]
        val_labels = labels_all[split_info['val_indices']]
        test_labels = labels_all[split_info['test_indices']]

        active_train_dir = os.path.join(base_path, 'train_images')
        val_source_dir = os.path.join(base_path, 'val_images')
        test_source_dir = os.path.join(base_path, 'test_images')
        train_count = split_info['train_count']
        val_count = split_info['val_count']
        test_count = split_info['test_count']

    # Convert labels to one-hot
    y_train = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

    # Streaming generators (<10 MB RAM per batch)
    def train_generator():
        indices = np.arange(train_count)
        np.random.shuffle(indices)
        for i in range(0, train_count, batch_size):
            batch_indices = indices[i:min(i + batch_size, train_count)]
            batch_images = [load_image_from_npz(os.path.join(active_train_dir, f'train_{idx:05d}.npz')) for idx in batch_indices]
            batch_labels = [y_train[idx] for idx in batch_indices]
            yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)

    def val_generator():
        for i in range(0, val_count, batch_size):
            batch_indices = np.arange(i, min(i + batch_size, val_count))
            batch_images = [load_image_from_npz(os.path.join(val_source_dir, f'val_{idx:05d}.npz')) for idx in batch_indices]
            batch_labels = [y_val[idx] for idx in batch_indices]
            yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)

    def test_generator():
        for i in range(0, test_count, batch_size):
            batch_indices = np.arange(i, min(i + batch_size, test_count))
            batch_images = [load_image_from_npz(os.path.join(test_source_dir, f'test_{idx:05d}.npz')) for idx in batch_indices]
            batch_labels = [y_test[idx] for idx in batch_indices]
            yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)

    # Create tf.data Datasets
    train_ds = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
        test_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    metadata_dict = {
        'num_classes': num_classes,
        'num_train': train_count,
        'num_val': val_count,
        'num_test': test_count,
        'label_encoding': label_encoding,
        'batches_per_epoch': (train_count + batch_size - 1) // batch_size,
        'smote_enabled': bool(use_smote),
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels
    }

    return train_ds, val_ds, test_ds, metadata_dict


def get_class_weights(base_path: str, use_smote: bool = False) -> Dict[int, float]:
    """
    Load precomputed class weights. If SMOTE is used, returns uniform weights (1.0).
    """
    if use_smote:
        # For perfectly balanced SMOTE datasets, all classes have uniform weight
        with open(os.path.join(base_path, 'label_encoding.json'), 'r') as f:
            label_encoding = json.load(f)
        return {i: 1.0 for i in range(len(label_encoding))}

    class_weights_file = os.path.join(base_path, 'class_weights.json')
    if os.path.exists(class_weights_file):
        with open(class_weights_file, 'r') as f:
            class_weights_dict = json.load(f)
        return {int(k): float(v) for k, v in class_weights_dict.items()}
    
    # Fallback to uniform if file not found
    with open(os.path.join(base_path, 'label_encoding.json'), 'r') as f:
        label_encoding = json.load(f)
    return {i: 1.0 for i in range(len(label_encoding))}


def get_label_encoding(base_path: str) -> Dict[str, int]:
    """Load label encoding mapping."""
    with open(os.path.join(base_path, 'label_encoding.json'), 'r') as f:
        return json.load(f)
