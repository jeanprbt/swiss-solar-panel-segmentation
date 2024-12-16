import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader


def train_unet(
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
    Train U-Net with early stopping, computing training and validation losses.

    Args:
        model (torch.nn.Module): U-Net model to be trained.
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
