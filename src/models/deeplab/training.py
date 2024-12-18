import torch
import torch.nn as nn
import torch.optim as optim

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
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """
    Train DeepLab model, compute IoU and loss, and perform validation with checkpoint saving.

    Args:
        model (torch.nn.Module): DeepLab model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        lr (float): Learning rate for the optimizer.
        device (torch.device): The device to run the training on.
        pochs (int): Number of epochs to train for.
        
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

    for epoch in range(epochs):
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

        all_data = []
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
                )
                all_data.append(x)
                all_targets.append(y)
                all_preds.append(preds)

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(val_loader)}")

        all_data = torch.cat(all_data, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_preds = torch.cat(all_preds, dim=0)

        emissions = tracker.stop()
        print(f"Training complete, emitted {emissions} kgCO2")
        return train_loss, train_iou
