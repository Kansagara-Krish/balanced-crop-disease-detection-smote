"""
Helper utility to load and consolidate individual NPZ files into numpy arrays.
Designed for memory-efficient training while providing numpy array interface.
"""

import numpy as np
import os
from pathlib import Path


def load_consolidated_arrays(base_path, split_type='train'):
    """
    Load and consolidate individual NPZ files for a split into a single numpy array.
    
    Args:
        base_path (str): Path to preprocessed_data_efficient directory
        split_type (str): 'train', 'val', or 'test'
        
    Returns:
        np.ndarray: Consolidated image array of shape (N, 224, 224, 3)
    """
    image_dir = os.path.join(base_path, f'{split_type}_images')
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Get list of all NPZ files
    npz_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npz')])
    
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in: {image_dir}")
    
    print(f"Loading {len(npz_files)} images from {split_type} set...")
    
    # Load all images
    images = []
    for i, npz_file in enumerate(npz_files):
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(npz_files)} images...")
        
        npz_path = os.path.join(image_dir, npz_file)
        data = np.load(npz_path)
        images.append(data['image'].astype('float32'))
    
    # Concatenate into single array
    consolidated = np.stack(images, axis=0)
    print(f"✓ Consolidated {split_type} set: {consolidated.shape}, dtype: {consolidated.dtype}\n")
    
    return consolidated


def load_all_splits(base_path):
    """
    Load and consolidate all three splits (train, val, test).
    
    Args:
        base_path (str): Path to preprocessed_data_efficient directory
        
    Returns:
        tuple: (X_train, X_val, X_test)
    """
    X_train = load_consolidated_arrays(base_path, 'train')
    X_val = load_consolidated_arrays(base_path, 'val')
    X_test = load_consolidated_arrays(base_path, 'test')
    
    return X_train, X_val, X_test


if __name__ == "__main__":
    # Example usage
    base_path = r'D:\cnn_model\results\preprocessed_data_efficient'
    
    print("Loading consolidated arrays...")
    X_train, X_val, X_test = load_all_splits(base_path)
    
    print("Summary:")
    print(f"  X_train: {X_train.shape}, range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  X_val:   {X_val.shape}, range: [{X_val.min():.3f}, {X_val.max():.3f}]")
    print(f"  X_test:  {X_test.shape}, range: [{X_test.min():.3f}, {X_test.max():.3f}]")
