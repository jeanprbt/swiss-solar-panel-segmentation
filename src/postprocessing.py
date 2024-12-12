import torch
import cv2
import numpy as np
from skimage.measure import label, regionprops
    

def post_process(pred: np.ndarray, roof_mask: np.ndarray, min_area: int = 150, kernel_size: int = 7):
    """
    Denoise the predicted segmentation by filtering out small connected components, smoothing edges
    and masking out non-roof areas.

    Args:
        preds (np.ndarray): binary predicted mask (shape: H x W)
        roof_mask (np.ndarray): mask of roof areas (shape: H x W)
        min_area (int): minimum area to keep connected components
        kernel_size (int): kernel size for median filtering

    Returns:
        np.ndarray: post-processed predicted mask
    """
    cleaned_mask = remove_small_components(pred, min_area)
    smoothed_mask = smooth_edges(cleaned_mask, kernel_size)
    return np.multiply(smoothed_mask, roof_mask.reshape(pred.shape))


def remove_small_components(pred: np.ndarray, min_area: int = 150):
    """
    Remove small connected components from the predicted segmentation.
    
    Args:
        pred (np.ndarray): binary predicted mask (shape: H x W)
        min_area (int): minimum area to keep connected components
        
    Returns:
        np.ndarray: post-processed predicted mask
    """
    labels = label(pred)
    cleaned_mask = np.zeros_like(pred)
    for region in regionprops(labels):
        if region.area >= min_area:
            cleaned_mask[labels == region.label] = 1
    return cleaned_mask


def smooth_edges(pred: np.ndarray, kernel_size: int = 7):
    """
    Remove small noise around the edges of the predicted segmentation using morphological operations.
    
    Args:
        pred (np.ndarray): binary predicted mask (shape: H x W)
        kernel_size (int): kernel size for morphological operations
        
    Returns:
        np.ndarray: post-processed predicted mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

