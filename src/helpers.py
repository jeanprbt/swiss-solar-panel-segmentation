import torch
import matplotlib.pyplot as plt
import albumentations as A
import torch.nn as nn

from tqdm import tqdm
from segmentation_dataset import SegmentationDataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from sklearn.metrics import confusion_matrix

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    lr,
    epochs=50,
    patience=3
):
    """
    Trains a UNet model with early stopping and returns training and validation losses.

    Args:
        model (torch.nn.Module): The UNet model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        criterion (torch.nn.Module): Loss function to use.
        lr (float): Learning rate for the optimizer.
        epochs (int): Maximum number of epochs to train (default: 50).
        patience (int):Number of epochs to wait for improvement before early stopping.

    Returns:
        train_losses (list): of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for image, mask, *_ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            image, mask = image.to(device), mask.to(device)
            outputs = model(image)
            loss = criterion(outputs, mask)

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
            for *_, image, mask in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                image, mask = image.float().to(device), mask.float().to(device)
                outputs = model(image)
                loss = criterion(outputs, mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Validation loss improved, saving model...")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")
    return train_losses, val_losses


def get_loaders(image_names, original_image_dir, masks_dir, batch_size=2, transform: Optional[A.Compose] = None):
    """
    Returns DataLoader objects for the training, validation, and test sets.

    Args:
        image_names (list): List of image names to include in the dataset.
        original_image_dir (str): Path to the directory containing the original images.
        masks_dir (str): Path to the directory containing the masks.
        batch_size (int): Batch size for the DataLoader objects.
        transform (Optional[A.Compose]): Transform to apply to the images and masks.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """    

    dataset = SegmentationDataset(original_image_dir, masks_dir, image_names, transform)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1]) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader



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



def metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: int = 0.2,
):
    """
    Evaluate the model on the dataset and calculate overall metrics.

    Args:
        model (torch.nn.Module): Trained model for segmentation.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform computations on (e.g., "cuda" or "cpu").
        threshold (float): Threshold to turn predictions binary.

    Returns:
        accuracy (float): Accuracy of the model.
        f1 (float): F1 score of the model.
        precision (float): Precision of the model.
        recall (float): Recall of the model.
        iou (float): Intersection over Union of the model.
    """
    model.eval()
    tn = 0
    fp = 0
    fn = 0
    tp = 0

    with torch.no_grad():
        for image, label, *_ in test_loader.dataset:
            image = torch.tensor(image).float().to(device)
            label = torch.tensor(label).float().to(device)

            predicted_output = model(image)
            predicted_mask = predicted_output.cpu().squeeze()
            binary_predicted = (predicted_mask > threshold).int()
            
            binary_predicted = binary_predicted.cpu().numpy().flatten()

            ground_truth_mask = label.cpu().squeeze()
            ground_truth = ground_truth_mask.cpu().numpy().flatten()

            conf = confusion_matrix(ground_truth, binary_predicted)

            if len(conf[0]) == 2:
                fp += conf[0, 1]
                fn += conf[1, 0]
                tp += conf[1, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    iou = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, f1, precision, recall, iou




def visualise_results(model: nn.Module, test_loader: DataLoader, device: torch.device , idx: int , threshold: float = 0.3):
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

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
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