import cv2
import numpy as np
import scipy.ndimage as ndimage


def postprocess(pred: np.ndarray, min_area: int = 150, kernel_size: int = 7) -> np.ndarray:
    """
    Remove small connected components from the predicted segmentation, as well as smaall noise
    around the edges using morphological operations.
    
    Args:
        pred (np.ndarray): binary predicted mask (shape: H x W)
        min_area (int): minimum area to keep connected components
        kernel_size (int): kernel size for morphological operations
        
    Returns:
        np.ndarray: post-processed predicted mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    filtered_mask = np.zeros_like(pred)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1    
            
    median_filtered_mask = ndimage.median_filter(filtered_mask, size=kernel_size)
    res =  ndimage.binary_fill_holes(median_filtered_mask).astype(np.uint8)
    return res