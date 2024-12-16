import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from postprocessing import postprocess


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device = torch.device("cpu"),
) -> list[float] | list[float]:
    """
    Train a model with early stopping, computing training and validation losses.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function to use.
        lr (float): Learning rate for the optimizer.
        epochs (int): Maximum number of epochs to train (default: 50).
        patience (int): Number of epochs to wait for improvement before early stopping.
        device (torch.device): Device to run the training on (e.g., 'cuda', 'mps', or 'cpu').

    Returns:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for image, label, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for image, label, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                image, label = image.float().to(device), label.float().to(device)
                outputs = model(image)
                loss = criterion(outputs, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")
    return train_losses, val_losses


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    idx: int,
    threshold: float = 0.2,
    post_process: bool = False,
    roof_masks_dir: str = "",
    device: torch.device = torch.device("cpu"),
) -> Image.Image | np.ndarray | np.ndarray:
    """
    Evaluate the model on a single image and return the predicted mask.
    
    Args:
        model (torch.nn.Module): Trained model for segmentation.
        test_loader (torch.utils.data.DataLoader): Data loader for the test dataset.
        idx (int): Index of the image to evaluate.
        threshold (float): Threshold to turn predictions binary.
        post_process (bool): Whether to apply post-processing to the predicted mask.
        roof_masks_dir (str): Path of directory containing masks of roof areas (only used if post_process is True).
        device (torch.device): Device to perform computations on (e.g., "cuda" or "cpu").
        
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
        predicted_output = (model(input_image).cpu().squeeze() > threshold).numpy().astype(np.uint8) * 255
        
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
):
    """
    Evaluate the model on the dataset and compute overall metrics.

    Args:
        model (torch.nn.Module): Trained model for segmentation.
        test_loader (torch.utils.data.DataLoader): Data loader for the test dataset.
        threshold (float): Threshold to turn predictions binary.
        post_process (bool): Whether to apply post-processing to the predicted masks.
        roof_masks_dir (str): Directory containing roof masks.
        device (torch.device): Device to perform computations on (e.g., "cuda" or "cpu").

    Returns:
        accuracy (float): Accuracy of the model.
        f1 (float): F1 score of the model.
        iou (float): Intersection over Union of the model.
    """
    model.eval()
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
                device=device
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