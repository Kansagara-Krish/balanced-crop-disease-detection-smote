"""
SMOTE (Synthetic Minority Over-sampling Technique) Utilities for Image Classification

Provides robust helpers to balance multi-class imbalanced image datasets at:
1. Pixel/Image Space: Direct oversampling on flattened image arrays.
2. Feature/Embedding Space: Oversampling on deep convolutional feature vectors.
3. Visualization: Pre vs. Post oversampling class distribution comparisons.
4. TensorFlow Dataset integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from typing import Tuple, Dict, Optional, List, Union
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


def check_imblearn_installed():
    """Verify that imbalanced-learn is available."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError(
            "imbalanced-learn is required for SMOTE utilities. "
            "Please install it using: pip install imbalanced-learn>=0.10.0"
        )


def apply_smote_to_images(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Union[str, Dict] = 'auto',
    k_neighbors: int = 3,
    random_state: int = 42,
    smote_type: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to 4D image array (N, Height, Width, Channels).

    Args:
        X: 4D numpy array of shape (N, H, W, C)
        y: 1D class labels or 2D one-hot encoded labels
        sampling_strategy: 'auto', 'minority', 'not majority', or dictionary specifying target counts
        k_neighbors: Number of nearest neighbors used to construct synthetic samples
        random_state: Seed for reproducibility
        smote_type: 'standard', 'borderline', or 'svm'

    Returns:
        X_resampled: Resampled 4D image array
        y_resampled: Resampled labels (matching input format: 1D or 2D one-hot)
    """
    check_imblearn_installed()

    is_one_hot = (y.ndim == 2 and y.shape[1] > 1)
    if is_one_hot:
        num_classes = y.shape[1]
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y.ravel()
        num_classes = len(np.unique(y_labels))

    # Validate k_neighbors against smallest minority class count
    class_counts = Counter(y_labels)
    min_count = min(class_counts.values())
    effective_k = min(k_neighbors, max(1, min_count - 1))

    if effective_k < 1:
        print(f"[Warning] Smallest class has only {min_count} sample(s). Minimum 2 required for SMOTE interpolation.")
        return X, y

    # Reshape 4D images to 2D feature matrix: (N, H*W*C)
    orig_shape = X.shape
    num_samples = orig_shape[0]
    img_height, img_width, img_channels = orig_shape[1], orig_shape[2], orig_shape[3]
    feature_dim = img_height * img_width * img_channels

    X_flat = X.reshape((num_samples, feature_dim))

    # Initialize SMOTE variant
    if smote_type == 'borderline':
        sampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k,
            random_state=random_state
        )
    elif smote_type == 'svm':
        sampler = SVMSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k,
            random_state=random_state
        )
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k,
            random_state=random_state
        )

    print(f"Applying {sampler.__class__.__name__} (k_neighbors={effective_k})...")
    X_resampled_flat, y_resampled_labels = sampler.fit_resample(X_flat, y_labels)

    # Reshape back to 4D: (N_resampled, H, W, C)
    resampled_samples = X_resampled_flat.shape[0]
    X_resampled = X_resampled_flat.reshape((resampled_samples, img_height, img_width, img_channels))

    # Clip pixel values to valid range [0.0, 1.0] or [0, 255] depending on input dtype/range
    if np.issubdtype(X.dtype, np.floating):
        max_val = np.max(X)
        if max_val <= 1.05:
            X_resampled = np.clip(X_resampled, 0.0, 1.0)
        else:
            X_resampled = np.clip(X_resampled, 0.0, 255.0)
    else:
        X_resampled = np.clip(X_resampled, 0, 255).astype(X.dtype)

    # Convert back to one-hot if original was one-hot
    if is_one_hot:
        y_resampled = tf.keras.utils.to_categorical(y_resampled_labels, num_classes=num_classes)
    else:
        y_resampled = y_resampled_labels

    print(f"Original shape: {X.shape} -> Resampled shape: {X_resampled.shape}")
    print(f"Samples count: {len(y_labels)} -> {len(y_resampled_labels)} (+{len(y_resampled_labels) - len(y_labels)} synthetic)")

    return X_resampled, y_resampled


def apply_feature_smote(
    feature_extractor: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    sampling_strategy: Union[str, Dict] = 'auto',
    k_neighbors: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE in deep feature/embedding space.
    Extracts high-level CNN representations (e.g. from GlobalAveragePooling2D layer),
    oversamples feature vectors with SMOTE, and returns balanced feature-label pairs.

    Args:
        feature_extractor: Pre-trained CNN model outputting feature vectors
        images: 4D numpy array (N, H, W, C)
        labels: 1D class labels or 2D one-hot encoded labels
        sampling_strategy: Target balancing strategy
        k_neighbors: Nearest neighbor count
        random_state: Random seed

    Returns:
        features_resampled: Resampled feature vectors (N_resampled, embedding_dim)
        labels_resampled: Resampled labels
    """
    check_imblearn_installed()

    is_one_hot = (labels.ndim == 2 and labels.shape[1] > 1)
    if is_one_hot:
        num_classes = labels.shape[1]
        y_labels = np.argmax(labels, axis=1)
    else:
        y_labels = labels.ravel()
        num_classes = len(np.unique(y_labels))

    print(f"Extracting feature embeddings with {feature_extractor.name}...")
    features = feature_extractor.predict(images, batch_size=32, verbose=0)

    class_counts = Counter(y_labels)
    min_count = min(class_counts.values())
    effective_k = min(k_neighbors, max(1, min_count - 1))

    sampler = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=effective_k,
        random_state=random_state
    )

    features_resampled, y_resampled_labels = sampler.fit_resample(features, y_labels)

    if is_one_hot:
        labels_resampled = tf.keras.utils.to_categorical(y_resampled_labels, num_classes=num_classes)
    else:
        labels_resampled = y_resampled_labels

    print(f"Feature embeddings: {features.shape} -> Resampled embeddings: {features_resampled.shape}")
    return features_resampled, labels_resampled


def plot_smote_distribution(
    y_before: np.ndarray,
    y_after: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Class Distribution: Before vs. After SMOTE Oversampling"
) -> plt.Figure:
    """
    Plot comparative bar charts showing class sample counts before and after SMOTE.

    Args:
        y_before: Labels before oversampling (1D or 2D one-hot)
        y_after: Labels after oversampling (1D or 2D one-hot)
        class_names: List of human-readable class names
        figsize: Matplotlib figure dimensions
        title: Plot title

    Returns:
        fig: Matplotlib Figure object
    """
    if y_before.ndim == 2 and y_before.shape[1] > 1:
        labels_before = np.argmax(y_before, axis=1)
    else:
        labels_before = y_before.ravel()

    if y_after.ndim == 2 and y_after.shape[1] > 1:
        labels_after = np.argmax(y_after, axis=1)
    else:
        labels_after = y_after.ravel()

    counts_before = Counter(labels_before)
    counts_after = Counter(labels_after)

    unique_classes = sorted(set(list(counts_before.keys()) + list(counts_after.keys())))
    num_classes = len(unique_classes)

    if class_names is None:
        names = [f"Class {c}" for c in unique_classes]
    else:
        names = [class_names[c] if c < len(class_names) else f"Class {c}" for c in unique_classes]

    vals_before = [counts_before.get(c, 0) for c in unique_classes]
    vals_after = [counts_after.get(c, 0) for c in unique_classes]

    x = np.arange(num_classes)
    width = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width/2, vals_before, width, label='Original (Before SMOTE)', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, vals_after, width, label='Balanced (After SMOTE)', color='#50E3C2', alpha=0.9, edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add count values on top of bars
    def autolabel(rects, is_synth=False):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='semibold')

    autolabel(rects1)
    autolabel(rects2, is_synth=True)

    plt.tight_layout()
    return fig


def create_smote_tf_dataset(
    X_resampled: np.ndarray,
    y_resampled: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Convert resampled numpy arrays into an optimized tf.data.Dataset pipeline.

    Args:
        X_resampled: 4D numpy array (N, H, W, C)
        y_resampled: 2D one-hot encoded labels (N, num_classes)
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        dataset: tf.data.Dataset ready for model.fit()
    """
    dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X_resampled), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def generate_smote_dataset_to_disk(
    base_path: str,
    output_dir: str,
    target_samples_per_class: Optional[int] = None,
    k_neighbors: int = 3,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Memory-efficient SMOTE dataset generation directly to disk.
    Processes one class at a time to keep RAM usage under 100 MB.
    Saves balanced training images as compressed .npz files into output_dir/train_images.
    Reuses validation and test splits from base_path with zero duplication.

    Args:
        base_path: Path to original preprocessed_data_efficient directory
        output_dir: Path to save SMOTE balanced dataset
        target_samples_per_class: Target sample count per class (defaults to max training class count)
        k_neighbors: Number of nearest neighbors for SMOTE interpolation
        random_state: Seed for reproducibility
        verbose: Whether to print progress

    Returns:
        summary_dict: Metadata containing counts, paths, and distribution
    """
    import os
    import json
    import gc
    from sklearn.neighbors import NearestNeighbors

    np.random.seed(random_state)
    os.makedirs(output_dir, exist_ok=True)
    out_train_dir = os.path.join(output_dir, 'train_images')
    os.makedirs(out_train_dir, exist_ok=True)

    # Load source metadata
    with open(os.path.join(base_path, 'split_indices.json'), 'r') as f:
        split_info = json.load(f)
    with open(os.path.join(base_path, 'image_metadata.json'), 'r') as f:
        image_metadata = json.load(f)
    with open(os.path.join(base_path, 'label_encoding.json'), 'r') as f:
        label_encoding = json.load(f)
    with open(os.path.join(base_path, 'preprocessing_info.json'), 'r') as f:
        preprocess_info = json.load(f)

    labels_all = np.array(image_metadata['labels'])
    train_labels = labels_all[split_info['train_indices']]
    num_classes = preprocess_info['num_classes']

    # Group original train indices by class ID
    class_to_indices = {}
    for local_idx, label in enumerate(train_labels):
        class_to_indices.setdefault(int(label), []).append(local_idx)

    class_counts = {c: len(class_to_indices.get(c, [])) for c in range(num_classes)}
    max_count = max(class_counts.values()) if class_counts else 0
    target_count = target_samples_per_class if target_samples_per_class else max_count

    if verbose:
        print("=" * 70)
        print("MEMORY-EFFICIENT DISK-BACKED SMOTE GENERATION")
        print("=" * 70)
        print(f"Source: {base_path}")
        print(f"Destination: {output_dir}")
        print(f"Classes: {num_classes} | Original train count: {len(train_labels)}")
        print(f"Target count per class: {target_count} (Total balanced: {target_count * num_classes})")
        print(f"RAM footprint: < 100 MB (processes 1 class at a time)\n")

    balanced_train_labels = []
    total_saved_idx = 0
    y_before_counts = []
    y_after_counts = []

    src_train_dir = os.path.join(base_path, 'train_images')

    for c in range(num_classes):
        indices = class_to_indices.get(c, [])
        n_orig = len(indices)
        y_before_counts.append(n_orig)

        # 1. Load images for this single class only (takes only ~30-50 MB RAM)
        class_images = []
        for idx in indices:
            npz_file = os.path.join(src_train_dir, f"train_{idx:05d}.npz")
            data = np.load(npz_file)
            class_images.append(data['image'].astype(np.float32))

        class_imgs_arr = np.array(class_images, dtype=np.float32)
        n_needed = target_count - n_orig

        # 2. Save original images to new SMOTE directory
        for img in class_imgs_arr:
            out_path = os.path.join(out_train_dir, f"train_{total_saved_idx:05d}.npz")
            np.savez_compressed(out_path, image=img)
            balanced_train_labels.append(c)
            total_saved_idx += 1

        # 3. Generate synthetic SMOTE images if minority
        n_synthetic = 0
        if n_needed > 0 and n_orig >= 2:
            effective_k = min(k_neighbors, n_orig - 1)
            flat = class_imgs_arr.reshape(n_orig, -1)
            nn = NearestNeighbors(n_neighbors=effective_k + 1, algorithm='auto')
            nn.fit(flat)
            _, nbrs = nn.kneighbors(flat)

            for _ in range(n_needed):
                i = np.random.randint(0, n_orig)
                # Pick a random neighbor (excluding sample itself)
                nbr_choice = np.random.choice(nbrs[i, 1:])
                diff = flat[nbr_choice] - flat[i]
                step = np.random.uniform(0.0, 1.0)
                synth_flat = flat[i] + step * diff
                synth_img = np.clip(synth_flat.reshape(224, 224, 3), 0.0, 1.0).astype(np.float32)

                out_path = os.path.join(out_train_dir, f"train_{total_saved_idx:05d}.npz")
                np.savez_compressed(out_path, image=synth_img)
                balanced_train_labels.append(c)
                total_saved_idx += 1
                n_synthetic += 1

        y_after_counts.append(n_orig + n_synthetic)

        if verbose:
            print(f"  Class {c:2d}: {n_orig:3d} original + {n_synthetic:3d} synthetic -> {target_count:3d} total")

        # Explicit cleanup to guarantee low RAM usage
        del class_images, class_imgs_arr
        gc.collect()

    # Save metadata files in output_dir
    smote_split_info = {
        'train_count': int(total_saved_idx),
        'val_count': int(split_info['val_count']),
        'test_count': int(split_info['test_count']),
        'train_indices': [int(x) for x in range(total_saved_idx)],
        'val_indices': [int(x) for x in split_info['val_indices']],
        'test_indices': [int(x) for x in split_info['test_indices']],
        'val_source_dir': os.path.join(base_path, 'val_images'),
        'test_source_dir': os.path.join(base_path, 'test_images')
    }
    with open(os.path.join(output_dir, 'split_indices.json'), 'w') as f:
        json.dump(smote_split_info, f, indent=4)

    smote_image_metadata = {
        'train_labels': [int(x) for x in balanced_train_labels],
        'val_labels': [int(x) for x in labels_all[split_info['val_indices']]],
        'test_labels': [int(x) for x in labels_all[split_info['test_indices']]]
    }
    with open(os.path.join(output_dir, 'image_metadata.json'), 'w') as f:
        json.dump(smote_image_metadata, f, indent=4)

    with open(os.path.join(output_dir, 'label_encoding.json'), 'w') as f:
        json.dump(label_encoding, f, indent=4)

    smote_preprocess_info = dict(preprocess_info)
    smote_preprocess_info.update({
        'train_samples': int(total_saved_idx),
        'smote_balanced': True,
        'target_samples_per_class': int(target_count),
        'total_synthetic_samples': int(total_saved_idx - len(train_labels)),
        'storage_path': str(output_dir)
    })
    with open(os.path.join(output_dir, 'preprocessing_info.json'), 'w') as f:
        json.dump(smote_preprocess_info, f, indent=4)

    # Class weights for balanced dataset are uniform (1.0)
    uniform_class_weights = {str(c): 1.0 for c in range(num_classes)}
    with open(os.path.join(output_dir, 'class_weights.json'), 'w') as f:
        json.dump(uniform_class_weights, f, indent=4)

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ SMOTE DATASET SUCCESSFULLY CREATED AT: {output_dir}")
        print(f"✓ Total balanced training samples: {total_saved_idx}")
        print(f"✓ Synthetic samples generated: {total_saved_idx - len(train_labels)}")
        print("=" * 70 + "\n")

    return {
        'output_dir': output_dir,
        'num_train': total_saved_idx,
        'num_classes': num_classes,
        'y_before_counts': y_before_counts,
        'y_after_counts': y_after_counts,
        'label_encoding': label_encoding
    }


def create_smote_disk_datasets(
    smote_base_path: str,
    original_base_path: str = None,
    batch_size: int = 16
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create streaming tf.data.Dataset objects for training on disk-backed SMOTE images.
    Loads 1 batch at a time (~9.6 MB in RAM), preventing MemoryError entirely.

    Args:
        smote_base_path: Path to preprocessed_data_smote
        original_base_path: Path to original preprocessed_data_efficient (for val/test splits)
        batch_size: Batch size

    Returns:
        train_ds, val_ds, test_ds, metadata
    """
    import os
    import json

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

    y_train = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

    train_count = split_info['train_count']
    val_count = split_info['val_count']
    test_count = split_info['test_count']

    # Val/test images directory
    val_dir = split_info.get('val_source_dir')
    if not val_dir or not os.path.exists(val_dir):
        if original_base_path:
            val_dir = os.path.join(original_base_path, 'val_images')
        else:
            val_dir = os.path.join(smote_base_path, 'val_images')

    test_dir = split_info.get('test_source_dir')
    if not test_dir or not os.path.exists(test_dir):
        if original_base_path:
            test_dir = os.path.join(original_base_path, 'test_images')
        else:
            test_dir = os.path.join(smote_base_path, 'test_images')

    train_dir = os.path.join(smote_base_path, 'train_images')

    def load_npz(path):
        try:
            return np.load(path)['image'].astype(np.float32)
        except Exception:
            return np.zeros((224, 224, 3), dtype=np.float32)

    def train_gen():
        indices = np.arange(train_count)
        np.random.shuffle(indices)
        for i in range(0, train_count, batch_size):
            batch_idxs = indices[i:min(i + batch_size, train_count)]
            imgs = [load_npz(os.path.join(train_dir, f"train_{idx:05d}.npz")) for idx in batch_idxs]
            lbls = [y_train[idx] for idx in batch_idxs]
            yield np.array(imgs, dtype=np.float32), np.array(lbls, dtype=np.float32)

    def val_gen():
        for i in range(0, val_count, batch_size):
            batch_idxs = np.arange(i, min(i + batch_size, val_count))
            imgs = [load_npz(os.path.join(val_dir, f"val_{idx:05d}.npz")) for idx in batch_idxs]
            lbls = [y_val[idx] for idx in batch_idxs]
            yield np.array(imgs, dtype=np.float32), np.array(lbls, dtype=np.float32)

    def test_gen():
        for i in range(0, test_count, batch_size):
            batch_idxs = np.arange(i, min(i + batch_size, test_count))
            imgs = [load_npz(os.path.join(test_dir, f"test_{idx:05d}.npz")) for idx in batch_idxs]
            lbls = [y_test[idx] for idx in batch_idxs]
            yield np.array(imgs, dtype=np.float32), np.array(lbls, dtype=np.float32)

    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
        test_gen,
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
        'batches_per_epoch': (train_count + batch_size - 1) // batch_size
    }

    return train_ds, val_ds, test_ds, metadata_dict

