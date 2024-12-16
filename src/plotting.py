import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

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


def plot_prediction(
    real_image: Image,
    ground_truth: np.ndarray,
    predicted_mask: np.ndarray,
):
    """
    Visualize the output of a segmentation model.

    Args:
        model (nn.Module): trained model
        test_loader (DataLoader): test data loader
        idx (int): index of the image to visualize
        threshold (float): threshold for binary predictions
        post_process (bool): whether to apply post-processing to the predicted mask
        roof_masks_dir (str): directory containing roof masks
        device (torch.device): device to perform computations on
    """
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(real_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Predicted Segmentation")
    axes[2].axis("off")


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


def plot_model_metrics(
    models: list[str], iou_scores: list[float], f1_scores: list[float]
):
    """
    Create bar plots comparing IOU and F1 scores for multiple models.

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
    _, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        indices - bar_width / 2, iou_scores, bar_width, label="IOU", color="skyblue"
    )
    bars2 = ax.bar(
        indices + bar_width / 2, f1_scores, bar_width, label="F1 Score", color="salmon"
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Scores", fontsize=12)
    ax.set_title("Comparison of IOU and F1 Scores for Different Models", fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10
            )
            
