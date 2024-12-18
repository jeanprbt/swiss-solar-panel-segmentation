import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker

from models.deeplab.utils import IOU, DiceBCELoss


def train_deeplab(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """
    Train DeepLab model, compute IoU and loss, and perform validation with checkpoint saving.

    Args:
        model (torch.nn.Module): DeepLab model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train for.
        patience (int): Number of epochs to wait for improvement before early stopping.
        device (torch.device): The device to run the training on.
        
    Returns:
        list[float]: training losses
        list[float]: training IoUs
    """
    tracker = EmissionsTracker(log_level="critical", save_to_file=False)
    tracker.start()
    
    model = model.to(device)
    loss_fn = DiceBCELoss()
    iou_fn = IOU()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(str(device))
    
    train_iou = []
    train_loss = []
    
    best_dice = 0
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch+1}/{epochs}")

        iterations = 0
        iter_loss = 0.0
        iter_iou = 0.0

        batch_loop = tqdm(train_loader, desc=f"Training Epoch: {epoch+1}/{epochs}")
        for _, (data, targets, _) in enumerate(batch_loop):

            data = data.to(device=device)
            targets = targets.float().unsqueeze(1).to(device=device)

            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions , targets)
                iou = iou_fn(predictions , targets)
                iter_loss += loss.item()
                iter_iou += iou.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iterations += 1 
            batch_loop.set_postfix(diceloss = loss.item(), iou = iou.item())


        train_loss.append(iter_loss / iterations)
        train_iou.append(iter_iou/iterations)
        print(f"Training loss: {round(train_loss[-1] , 3)}")

        num_correct = 0
        num_pixels = 0
        dice_score = 0
        model.eval()

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x = x.to(device)
                y = y.float().unsqueeze(1).to(device)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
                )
        
        dice = dice_score / len(val_loader)
        print(f"Dice score: {dice}")
        
        if dice > best_dice:
            best_dice = dice
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt > patience:
            print("Early stopping triggered")
            break

    emissions = tracker.stop()
    print(f"Training complete, emitted {emissions} kgCO2")
    return train_loss, train_iou
