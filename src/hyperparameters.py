import optuna
import torch.nn as nn
import torch.optim as optim
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)


# ------------------------------------- Segmentation Threshold -----------------------------------------

def optimize_threshold(
    model: nn.Module, test_loader: DataLoader, n_trials: int = 10
) -> float:
    """
    Optimize the threshold of a semantic segmentation model using Optuna.

    Args:
        model (nn.Module): A PyTorch semantic segmentation model accepting images as input and outputting masks.
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for validation data.
        n_trials (int): Number of trials to run.

    Returns:
        float: Best threshold found by Optuna.
    """
    study = optuna.create_study(direction=["maximize", "maximize"])
    study.optimize(lambda trial: threshold_objective(trial, model, test_loader))
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    return study.best_params["threshold"]


def threshold_objective(
    trial: optuna.trial.Trial, model: nn.Module, test_loader: DataLoader
) -> float | float:
    """
    Objective function for Optuna to optimize the threshold of a semantic segmentation model.

    Args:
        trial (optuna.trial.Trial): Optuna's trial object.
        model (nn.Module): A PyTorch semantic segmentation model accepting images as input and outputting masks.
        test_loader (torch.utils.data.dataloader.DataLoader): DataLoader for test data.

    Returns:
        float: F1-Score.
        float: Intersection over Union (IoU) score.
    """
    threshold = trial.suggest_float("threshold", 0.1, 0.9)
    model.eval()
    tn, fp, fn, tp = [0] * 4
    with torch.no_grad():
        for image, mask in test_loader:
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            outputs = model(image).cpu().squeeze()
            preds = (outputs > threshold).int().numpy().flatten()
            ground_truth = mask.cpu().squeeze().numpy().flatten()
            conf = confusion_matrix(ground_truth, preds)
            if len(conf[0]) == 2:
                fp += conf[0,1]
                fn += conf[1,0]
                tp += conf[1,1]
            tn += conf[0,0]
    
    iou = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return  f1, iou

# ----------------------------------------------- UNet -------------------------------------------------

def optimize_unet_hyperparameters(
    train_loader: DataLoader, val_loader: DataLoader, n_trials: int = 10
) -> dict:
    """
    Optimize hyperparameters of a U-Net model using Optuna, namely learning rate, number of
    epochs, number of layers and kernel size.
    
    Args:
        train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for validation data.
        n_trials (int): Number of trials to run.
        
    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: unet_objective(trial, train_loader, val_loader))
    return study.best_params


def unet_objective(
    trial: optuna.trial.Trial, train_loader: DataLoader, val_loader: DataLoader
) -> float:
    """
    Objective function for Optuna to optimize hyperparameters of a U-Net model, i.e learning rate,
    number of epoches, number of layers and kernel size.

    Args:
        trial (optuna.trial.Trial): Optuna's trial object.
        train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for validation data.

    Returns:
        float: Validation loss.
    """
    # Define a U-Net model using Optuna's trial object
    model = unet(trial).to(DEVICE)
    criterion = nn.BCELoss()
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 20, 40, step=5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping params
    patience = 3
    best_val_loss = float("inf")
    patience_cnt = 0

    for _ in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        # Early stopping
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    return val_loss


def unet(trial: optuna.trial.Trial) -> nn.Module:
    """
    Define a U-Net model with nb. of layers and kernel size optimized by Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna's trial object.

    Returns:
        nn.Module: A U-Net model.
    """
    nb_layers = trial.suggest_int("encoder_layers", 2, 3)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    encoder = []
    for i in range(nb_layers):
        in_channels, out_channels = 3 if i == 0 else 2 ** (5 + i), 2 ** (6 + i)
        padding = kernel_size // 2
        encoder.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            )
        )
        encoder += [nn.ReLU(), nn.MaxPool2d(2)]

    decoder = []
    for i in range(nb_layers - 1):
        in_channels, out_channels = 2 ** (5 + nb_layers - i), 2 ** (4 + nb_layers - i)
        decoder.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
        decoder.append(nn.ReLU())
        decoder.append(nn.ConvTranspose2d(2**6, 1, kernel_size=2, stride=2))
        decoder.append(nn.Sigmoid())

    return nn.Sequential(*encoder, *decoder)
