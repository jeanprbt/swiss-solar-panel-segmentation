import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deeplab.utils import IoU


def train_deeplab(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    lr: float,
    epochs: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """
    Train DeepLab model, compute IoU and loss, and perform validation with checkpoint saving.

    Args:
        model (torch.nn.Module): DeepLab model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function to use.
        lr (float): Learning rate for the optimizer.
        device (torch.device): The device to run the training on.
        pochs (int): Number of epochs to train for.
        
    Returns:
        list[float]: training losses
        list[float]: training IoUs
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    iou_fn = IoU()
    scaler = torch.amp.GradScaler(str(device))
    
    train_iou = []
    train_loss = []
    for epoch in range(epochs):
        model.train()
        
        iterations = 0
        iter_loss = 0.0
        iter_iou = 0.0
        
        # Training loop
        batch_loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for image, mask, _ in batch_loop:
            image, mask = image.to(device), mask.to(device)
            with torch.amp.autocast(device_type=str(device)):
                predictions = model(image)
                loss = criterion(predictions, mask)
                iou = iou_fn(predictions, mask)
                iter_loss += loss.item()
                iter_iou += iou.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iterations += 1
            batch_loop.set_postfix(diceloss=loss.item(), iou=iou.item())

        train_loss.append(iter_loss / iterations)
        train_iou.append(iter_iou / iterations)

        print(f"Epoch: {epoch + 1}/{epochs}, Training loss: {round(train_loss[-1], 3)}")

        # Validation loop
        num_correct = 0
        num_pixels = 0
        dice_score = 0.0
        model.eval()

        with torch.no_grad():
            for image, mask, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                x, y = image.float().to(device), mask.float().to(device)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum().item()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}%")
        print(f"Dice score: {dice_score / len(val_loader):.4f}")

    print("Training complete.")
    return train_loss, train_iou
