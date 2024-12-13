import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
    
    
def visualize_results(model: nn.Module, test_loader: DataLoader, device: torch.device , idx: int , threshold: float = 0.3):
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
    test_images, test_masks, *_  = test_loader.dataset[idx]
    sample_image = test_images.clone().detach().float().to(device)
    sample_mask = test_masks.clone().detach().float().to(device)


    with torch.no_grad():
        predicted_output = model(sample_image)

    input_image = sample_image.cpu().squeeze().permute(1, 2, 0) / 255
    ground_truth_mask = sample_mask.cpu().squeeze() / 255

    predicted_mask = predicted_output.cpu().squeeze() / 255
    binary_predicted_mask = (predicted_mask > 0.3)

    _, axes = plt.subplots(1, 4, figsize=(15, 5))
    # Input image
    axes[0].imshow(input_image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Ground truth mask
    axes[1].imshow(ground_truth_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    # Predicted mask
    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    # Thresholded predicted mask
    axes[3].imshow(binary_predicted_mask, cmap="gray")
    axes[3].set_title("Thresholded Predicted Mask")
    axes[3].axis('off');
    
    
def plot_losses(train_losses, val_losses):
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
