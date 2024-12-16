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
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    lr: float,
    epochs: int = 50,
    patience: int = 3,
) -> list[float] | list[float]:
    """
    Train a model with early stopping, computing training and validation losses.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        criterion (torch.nn.Module): Loss function to use.
        lr (float): Learning rate for the optimizer.
        epochs (int): Maximum number of epochs to train (default: 50).
        patience (int): Number of epochs to wait for improvement before early stopping.

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
            print(
                f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")
    return train_losses, val_losses


def get_loaders(
    image_names: list[str],
    images_dir: str,
    masks_dir: str,
    batch_size: int = 2,
    transform: Optional[A.Compose] = None,
    seed: int = 42,
) -> DataLoader | DataLoader | DataLoader:
    """
    Create DataLoader objects for the training, validation, and test sets.

    Args:
        image_names (list): List of image names to include in the dataset.
        original_image_dir (str): Path to the directory containing the original images.
        masks_dir (str): Path to the directory containing the masks.
        batch_size (int): Batch size for the DataLoader objects.
        transform (Optional[A.Compose]): Transform to apply to the images and masks.
        seed (int): Seed for the random number generator.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = SegmentationDataset(images_dir, masks_dir, image_names, transform)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1], 
        generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


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
            image = image.clone().detach().float().to(device)
            label = label.clone().detach().float().to(device)

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

    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn != 0 else 0
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return accuracy, f1, precision, recall, iou
