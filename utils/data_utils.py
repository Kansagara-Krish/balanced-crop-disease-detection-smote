"""
Data Utility Functions for Crop Leaf Disease Classification
Handles dataset structure validation, image loading, and preprocessing
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional


def verify_dataset_structure(base_path: str) -> Dict:
    """
    Verify the dataset structure and check for issues.
    
    Args:
        base_path: Root path containing crop folders
        
    Returns:
        Dictionary containing dataset statistics and issues
    """
    stats = {
        'total_images': 0,
        'crops': {},
        'issues': [],
        'empty_classes': [],
        'corrupted_images': [],
        'image_formats': {}
    }
    
    crops = os.listdir(base_path)
    crops = [c for c in crops if os.path.isdir(os.path.join(base_path, c))]
    
    for crop in crops:
        crop_path = os.path.join(base_path, crop)
        crop_stats = {'healthy': 0, 'unhealthy': {}}
        
        # Check Healthy folder
        healthy_path = os.path.join(crop_path, 'Healthy_' + crop.lower())
        if not os.path.exists(healthy_path):
            # Try alternative naming
            for item in os.listdir(crop_path):
                if item.lower().startswith('healthy') and os.path.isdir(os.path.join(crop_path, item)):
                    healthy_path = os.path.join(crop_path, item)
                    break
        
        if os.path.exists(healthy_path):
            healthy_images = get_image_files(healthy_path)
            crop_stats['healthy'] = len(healthy_images)
            stats['total_images'] += len(healthy_images)
        else:
            stats['issues'].append(f"Missing healthy folder for {crop}")
            stats['empty_classes'].append(f"{crop}/Healthy")
        
        # Check Unhealthy folders (diseases)
        unhealthy_path = os.path.join(crop_path, 'Unhealthy_' + crop.lower())
        if not os.path.exists(unhealthy_path):
            for item in os.listdir(crop_path):
                if item.lower().startswith('unhealthy') and os.path.isdir(os.path.join(crop_path, item)):
                    unhealthy_path = os.path.join(crop_path, item)
                    break
        
        if os.path.exists(unhealthy_path):
            diseases = [d for d in os.listdir(unhealthy_path) if os.path.isdir(os.path.join(unhealthy_path, d))]
            for disease in diseases:
                disease_path = os.path.join(unhealthy_path, disease)
                disease_images = get_image_files(disease_path)
                crop_stats['unhealthy'][disease] = len(disease_images)
                stats['total_images'] += len(disease_images)
        
        stats['crops'][crop] = crop_stats
    
    return stats


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Path to directory
        extensions: Allowed file extensions (default: ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'])
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    if not os.path.exists(directory):
        return []
    
    images = []
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in extensions):
            images.append(os.path.join(directory, file))
    
    return images


def check_image_integrity(image_path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if an image file is not corrupted.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True, None
    except Exception as e:
        return False, str(e)


def find_corrupted_images(base_path: str) -> List[Dict]:
    """
    Find all corrupted images in the dataset.
    
    Args:
        base_path: Root path containing crop folders
        
    Returns:
        List of dictionaries with corrupted image info
    """
    corrupted = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                is_valid, error = check_image_integrity(image_path)
                if not is_valid:
                    corrupted.append({
                        'path': image_path,
                        'error': error,
                        'relative_path': os.path.relpath(image_path, base_path)
                    })
    
    return corrupted


def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Load and resize an image.
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
        
    Returns:
        Resized image array or None if image is corrupted
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, (target_size[1], target_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def create_image_label_mapping(base_path: str) -> Tuple[Dict, Dict]:
    """
    Create mapping between image paths and their labels.
    
    Args:
        base_path: Root path containing crop folders
        
    Returns:
        Tuple of (image_label_dict, label_encoding_dict)
    """
    image_labels = {}
    label_encoding = {}
    label_counter = 0
    
    for crop in os.listdir(base_path):
        crop_path = os.path.join(base_path, crop)
        if not os.path.isdir(crop_path):
            continue
        
        # Process healthy images
        for item in os.listdir(crop_path):
            item_path = os.path.join(crop_path, item)
            if not os.path.isdir(item_path):
                continue
                
            if item.lower().startswith('healthy'):
                label = f"{crop}_Healthy"
                if label not in label_encoding:
                    label_encoding[label] = label_counter
                    label_counter += 1
                
                images = get_image_files(item_path)
                for img in images:
                    image_labels[img] = label
            
            elif item.lower().startswith('unhealthy'):
                # Process diseases
                for disease in os.listdir(item_path):
                    disease_path = os.path.join(item_path, disease)
                    if os.path.isdir(disease_path):
                        label = f"{crop}_{disease}"
                        if label not in label_encoding:
                            label_encoding[label] = label_counter
                            label_counter += 1
                        
                        images = get_image_files(disease_path)
                        for img in images:
                            image_labels[img] = label
    
    return image_labels, label_encoding


def get_dataset_summary(base_path: str) -> str:
    """
    Generate a text summary of the dataset.
    
    Args:
        base_path: Root path containing crop folders
        
    Returns:
        Summary string
    """
    stats = verify_dataset_structure(base_path)
    
    summary = "\n" + "="*60 + "\n"
    summary += "DATASET SUMMARY\n"
    summary += "="*60 + "\n"
    summary += f"Total Images: {stats['total_images']}\n"
    summary += f"Total Crops: {len(stats['crops'])}\n\n"
    
    summary += "CROP-WISE BREAKDOWN:\n"
    summary += "-"*60 + "\n"
    for crop, crop_data in stats['crops'].items():
        summary += f"\n{crop}:\n"
        summary += f"  Healthy: {crop_data['healthy']}\n"
        summary += f"  Diseases: {len(crop_data['unhealthy'])}\n"
        for disease, count in crop_data['unhealthy'].items():
            summary += f"    - {disease}: {count}\n"
    
    if stats['issues']:
        summary += "\n" + "="*60 + "\n"
        summary += "ISSUES DETECTED:\n"
        summary += "-"*60 + "\n"
        for issue in stats['issues']:
            summary += f"⚠️  {issue}\n"
    
    if stats['empty_classes']:
        summary += "\nEMPTY CLASSES:\n"
        for empty in stats['empty_classes']:
            summary += f"  - {empty}\n"
    
    return summary


# ============================================================================
# Advanced Image Preprocessing Functions
# ============================================================================

def denoise_image(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Denoise an image using various methods.
    
    Args:
        image: Input image (RGB, values 0-255)
        method: 'bilateral' (edge-preserving), 'gaussian' (strong blur), or 'morphological'
        
    Returns:
        Denoised image
    """
    if method == 'bilateral':
        # Bilateral filter: preserves edges while removing noise
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'gaussian':
        # Gaussian blur: simple but may blur edges
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'morphological':
        # Morphological opening: remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    else:
        denoised = image
    
    return denoised


def enhance_image_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
    
    Args:
        image: Input image (RGB, values 0-255, uint8)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for adaptive histogram
        
    Returns:
        Enhanced image (uint8, values 0-255)
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply CLAHE only to the V (value/brightness) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    v_enhanced = clahe.apply(v)
    
    # Merge back
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    
    return enhanced


def enhance_image_sharpening(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Apply unsharp masking to enhance details and sharpness.
    
    Args:
        image: Input image (RGB, values 0-255)
        strength: Strength of sharpening (1.0 = no change)
        
    Returns:
        Sharpened image
    """
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Apply Gaussian blur for unsharp mask
    blurred = cv2.GaussianBlur(img_float, (0, 0), 1.0)
    
    # Unsharp mask: enhanced = original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img_float, 1.0 + strength, blurred, -strength, 0)
    sharpened = np.clip(sharpened, 0, 1)
    
    # Convert back to uint8
    sharpened = (sharpened * 255).astype(np.uint8)
    
    return sharpened


def check_image_quality(image: np.ndarray, min_contrast: float = 10.0, min_sharpness: float = 50.0) -> Tuple[bool, Dict]:
    """
    Check image quality metrics: contrast, sharpness, brightness.
    
    Args:
        image: Input image (RGB, values 0-255)
        min_contrast: Minimum acceptable standard deviation (contrast)
        min_sharpness: Minimum acceptable Laplacian variance (sharpness)
        
    Returns:
        Tuple of (is_good_quality, quality_dict)
    """
    quality = {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Contrast (standard deviation of pixel values)
    contrast = np.std(gray)
    quality['contrast'] = float(contrast)
    
    # Sharpness (Laplacian variance - measure of detail)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    quality['sharpness'] = float(sharpness)
    
    # Brightness (mean pixel value)
    brightness = np.mean(gray)
    quality['brightness'] = float(brightness)
    
    # Check if image quality is acceptable
    is_good = contrast >= min_contrast and sharpness >= min_sharpness and brightness > 20 and brightness < 230
    quality['is_acceptable'] = is_good
    
    return is_good, quality


def preprocess_image_complete(image: np.ndarray, 
                             denoise: bool = True,
                             enhance: bool = True,
                             sharpen: bool = True,
                             check_quality: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Complete preprocessing pipeline: denoise → enhance → sharpen → quality check.
    
    Args:
        image: Input image (RGB, values 0-1 float or 0-255 uint8)
        denoise: Apply bilateral denoising
        enhance: Apply CLAHE contrast enhancement
        sharpen: Apply sharpening
        check_quality: Perform quality checks
        
    Returns:
        Tuple of (preprocessed_image, quality_dict) - returned as float [0, 1] if input was float
    """
    quality_dict = {}
    
    # Store original input type to determine output format
    input_is_float = (image.dtype == np.float32 or image.dtype == np.float64)
    
    # Step 0: Ensure input is uint8 [0, 255] for OpenCV functions
    if input_is_float:
        # Convert [0, 1] float to [0, 255] uint8
        if image.max() <= 1.0:
            processed = (image * 255.0).astype(np.uint8)
        else:
            processed = np.clip(image, 0, 255).astype(np.uint8)
    else:
        processed = image.astype(np.uint8)
    
    # Step 1: Denoise
    if denoise:
        processed = denoise_image(processed, method='bilateral')
    
    # Step 2: Enhance contrast
    if enhance:
        processed = enhance_image_clahe(processed, clip_limit=2.0, tile_size=8)
    
    # Step 3: Sharpen
    if sharpen:
        processed = enhance_image_sharpening(processed, strength=1.5)
    
    # Step 4: Quality check (on uint8 image)
    if check_quality:
        is_good, quality_dict = check_image_quality(processed)
    
    # Step 5: Convert back to original format
    if input_is_float:
        # Return as float [0, 1] to match normalized training pipeline
        processed = processed.astype(np.float32) / 255.0
    else:
        # Ensure uint8 in range [0, 255]
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    return processed, quality_dict
