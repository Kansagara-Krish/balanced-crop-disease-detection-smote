"""
Visualization Utility Functions for Crop Leaf Disease Classification
Handles plotting, confusion matrices, and exploratory analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


def plot_sample_images(image_arrays: np.ndarray, labels: List[str], 
                       title: str = "Sample Images", figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot a grid of sample images.
    
    Args:
        image_arrays: Array of images
        labels: Labels for each image
        title: Title for the plot
        figsize: Figure size (width, height)
    """
    n_images = len(image_arrays)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, (img, label) in enumerate(zip(image_arrays, labels)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Normalize if needed
        if img.max() > 1:
            img = img / 255.0
        
        ax.imshow(img)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_counts: Dict[str, int], figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot class distribution.
    
    Args:
        class_counts: Dictionary of class names to counts
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_crop_disease_breakdown(stats_dict: Dict, figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Plot disease breakdown for each crop.
    
    Args:
        stats_dict: Dictionary with crop statistics
        figsize: Figure size
    """
    n_crops = len(stats_dict['crops'])
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (crop, crop_data) in enumerate(stats_dict['crops'].items()):
        ax = axes[idx]
        
        # Prepare data
        labels = ['Healthy'] + list(crop_data['unhealthy'].keys())
        counts = [crop_data['healthy']] + list(crop_data['unhealthy'].values())
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(labels)))
        wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        
        ax.set_title(f'{crop}\n(Total: {sum(counts)})', fontweight='bold', fontsize=11)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
    
    # Hide extra subplots
    for idx in range(n_crops, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Disease Distribution by Crop', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_training_history(history_dict: Dict, figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Plot training history (accuracy and loss).
    
    Args:
        history_dict: History dictionary from model training
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy plot
    if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
        axes[0].plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontweight='bold')
        axes[0].set_title('Model Accuracy', fontweight='bold', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    # Loss plot
    if 'loss' in history_dict and 'val_loss' in history_dict:
        axes[1].plot(history_dict['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Loss', fontweight='bold')
        axes[1].set_title('Model Loss', fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         figsize: Tuple[int, int] = (12, 10),
                         normalize: bool = False) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize: Figure size
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, float], figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot comparison of different metrics.
    
    Args:
        metrics_dict: Dictionary of metric names to values
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Model Metrics', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {width:.4f}',
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_predictions_with_confidence(predictions: np.ndarray, images: np.ndarray,
                                     true_labels: List[str], pred_labels: List[str],
                                     class_names: List[str],
                                     figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot predictions with confidence scores.
    
    Args:
        predictions: Prediction probability arrays
        images: Image arrays
        true_labels: True class labels
        pred_labels: Predicted class labels
        class_names: All class names
        figsize: Figure size
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for idx in range(n_images):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        
        # Plot image
        img = images[idx]
        if img.max() > 1:
            img = img / 255.0
        ax.imshow(img)
        
        # Get confidence
        confidence = np.max(predictions[idx])
        correct = true_labels[idx] == pred_labels[idx]
        color = 'green' if correct else 'red'
        
        title = f'True: {true_labels[idx]}\nPred: {pred_labels[idx]}\nConf: {confidence:.2f}'
        ax.set_title(title, fontweight='bold', color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Predictions with Confidence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def save_plot(filename: str, output_dir: str = 'results/plots') -> None:
    """
    Save current plot to file.
    
    Args:
        filename: Output filename
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filepath}")
