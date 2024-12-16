import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from postprocessing import postprocess


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    idx: int,
    threshold: float = 0.2,
    post_process: bool = False,
    roof_masks_dir: str = "",
    device: torch.device = torch.device("cpu"),
    deeplab: bool = False,
) -> Image.Image | np.ndarray | np.ndarray:
    """
    Evaluate a segmentation model on a single image and return the predicted mask.
    
    Args:
        model (torch.nn.Module): Trained model for segmentation.
        test_loader (torch.utils.data.DataLoader): Data loader for the test dataset.
        idx (int): Index of the image to evaluate.
        threshold (float): Threshold to turn predictions binary.
        post_process (bool): Whether to apply post-processing to the predicted mask.
        roof_masks_dir (str): Path of directory containing masks of roof areas (only used if post_process is True).
        device (torch.device): Device to perform computations on (e.g., "cuda" or "cpu").
        deeplab (bool): Whether the model is a DeepLab model.
        
    Returns:
        PIL.Image.Image: Real image.
        np.ndarray: Ground truth mask for the image.
        np.ndarray: Predicted mask for the image.
    """
    model.eval()
    with torch.no_grad():
        image, label, image_name = test_loader.dataset[idx]
        
        # Run predictions
        input_image, label_image = image.to(device), label.to(device)
        predicted_output = (
            model(input_image if not deeplab else input_image.unsqueeze(0)).cpu().squeeze() > threshold
        ).numpy().astype(np.uint8) * 255
        
        # Retrieve real image and ground truth mask
        real_image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        ground_truth_mask = label_image.cpu().squeeze().numpy().astype(np.uint8) * 255
        
        # Post processing
        if post_process:
            roof_mask = np.array(
                Image.open(os.path.join(roof_masks_dir, image_name + ".png")).convert("L")
            ).astype(np.uint32)
            predicted_output = np.multiply(
                postprocess(predicted_output), roof_mask.reshape(predicted_output.shape)
            )   
    return real_image, ground_truth_mask, predicted_output
   

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: int = 0.2,
    post_process: bool = False,
    roof_masks_dir: str = "",
    device: torch.device = torch.device("cpu"),
    deeplab: bool = False,
):
    """
    Evaluate a segmentation model on a test split and compute overall metrics.

    Args:
        model (torch.nn.Module): Trained model for segmentation.
        test_loader (torch.utils.data.DataLoader): Data loader for the test dataset.
        threshold (float): Threshold to turn predictions binary.
        post_process (bool): Whether to apply post-processing to the predicted masks.
        roof_masks_dir (str): Directory containing roof masks.
        device (torch.device): Device to perform computations on (e.g., "cuda" or "cpu").
        deeplab (bool): Whether the model is a DeepLab model

    Returns:
        accuracy (float): Accuracy of the model.
        f1 (float): F1 score of the model.
        iou (float): Intersection over Union of the model.
    """
    model.eval()
    model = model.to(device)
    tn, fp, fn, tp = [0] * 4
    with torch.no_grad():
        for i in tqdm(range(len(test_loader.dataset)), desc="Evaluating on test set"):
            _, ground_truth, preds = predict(
                model=model, 
                test_loader=test_loader, 
                idx=i, 
                threshold=threshold,
                post_process=post_process, 
                roof_masks_dir=roof_masks_dir, 
                device=device,
                deeplab=deeplab,
            )  
            preds = preds.flatten()  
            ground_truth = ground_truth.flatten()
            conf = confusion_matrix(ground_truth, preds)
            if len(conf[0]) == 2:
                fp += conf[0, 1]
                fn += conf[1, 0]
                tp += conf[1, 1]
            tn += conf[0, 0]

    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn != 0 else 0
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return accuracy, f1, iou