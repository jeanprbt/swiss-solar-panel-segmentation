import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader


def plot_mask_on_image(
    image_path: str,
    mask: dict,
    title: str,
    image_size: tuple[int, int] = (1000, 1000),
    alpha: float = 0.5,
):
    """
    Plot a segmentation mask on top of an image.

    Args:
        image_path (str): Path to the original image file.
        mask (np.ndarray): 1D numpy array (flattened) representing the segmentation mask.
        title (str): Title of the plot.
        image_size (tuple): Dimensions of the image (width, height).
        alpha (float): Transparency level of the mask overlay (0 to 1).
    """
    mask_2d = mask.reshape(image_size)
    image = Image.open(image_path).resize(image_size)

    plt.imshow(image, cmap="gray")
    plt.imshow(mask_2d, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title(title)


def visualize_results(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    idx: int,
    threshold: float = 0.3,
):
    """
    Visualise the results of the model on the test dataset

    Args:
        model (nn.Module): trained model
        test_loader (DataLoader): test data loader
        device (torch.device): device to perform computations on
        idx (int): index of the image to visualise
        threshold (float): threshold for binary predictions
    """
    model.eval()
    test_images, test_masks, *_ = test_loader.dataset[idx]
    sample_image = test_images.clone().detach().float().to(device)
    sample_mask = test_masks.clone().detach().float().to(device)

    with torch.no_grad():
        predicted_output = model(sample_image)

    input_image = sample_image.cpu().squeeze().permute(1, 2, 0) / 255
    ground_truth_mask = sample_mask.cpu().squeeze() / 255

    predicted_mask = predicted_output.cpu().squeeze() / 255
    binary_predicted_mask = predicted_mask > 0.3

    _, axes = plt.subplots(1, 4, figsize=(15, 5))
    # Input image
    axes[0].imshow(input_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Ground truth mask
    axes[1].imshow(ground_truth_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Predicted mask
    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    # Thresholded predicted mask
    axes[3].imshow(binary_predicted_mask, cmap="gray")
    axes[3].set_title("Thresholded Predicted Mask")
    axes[3].axis("off")


def plot_losses(train_losses: list[float], val_losses: list[float]):
    """
    plots the training and validation losses over epochs.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()


def compare_model_metrics(
    models: list[str], iou_scores: list[float], f1_scores: list[float]
):
    """
    Plots bar plots comparing IOU and F1 scores for multiple models.

    Parameters:
    - models (list[str]): List of model names.
    - iou_scores (list[float]): List of IOU scores corresponding to each model.
    - f1_scores (list[float]): List of F1 scores corresponding to each model.
    """
    if len(models) != len(iou_scores) or len(models) != len(f1_scores):
        raise ValueError(
            "The lengths of models, iou_scores, and f1_scores must be equal."
        )

    bar_width = 0.35
    indices = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        indices - bar_width / 2, iou_scores, bar_width, label="IOU", color="skyblue"
    )
    bars2 = ax.bar(
        indices + bar_width / 2, f1_scores, bar_width, label="F1 Score", color="salmon"
    )

    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Scores", fontsize=12)
    ax.set_title("Comparison of IOU and F1 Scores for Different Models", fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # Offset for text
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()
